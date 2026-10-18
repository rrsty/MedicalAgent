"""API 路由：consult / consult/stream(SSE) / health / stats / conversations。

- POST /api/consult          同步问诊（Semaphore 并发限制 + 超时处理）
- POST /api/consult/stream   SSE 流式：progress（节点进度）→ answer（完整响应）→ done
- GET  /api/health           健康检查
- GET  /api/stats            系统统计
- GET/DELETE /api/conversations*  对话历史（读 MySQL，删除同步清 Redis）
"""
import asyncio
import json
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from config.settings import settings
from src.api.models import (
    ConsultationRequest,
    ConsultationResponse,
    ConversationSummary,
    HealthResponse,
    LinkedEntity,
    MedicationItem,
    MessageItem,
    StatsResponse,
)
from src.common.logger import setup_logger
from src.history import store as history_store
from src.knowledge_graph.graph_queries import GraphQueries
from src.knowledge_graph.schema_manager import verify_connectivity
from src.vector_store.search import get_shared_retriever

logger = setup_logger("api.routes", "api.log")

router = APIRouter(prefix="/api")

# 并发限制（防止系统过载）+ 全局状态引用（main.py lifespan 注入）
_semaphore: asyncio.Semaphore | None = None
_app_state: dict = {}


def init_router(app_state: dict, semaphore: asyncio.Semaphore) -> None:
    global _semaphore, _app_state
    _app_state = app_state
    _semaphore = semaphore


_ANSWER_CACHE_TTL_HOURS = 6


def _answer_cache_key(query: str, history: list) -> str:
    """答案缓存键：归一化问句 + 最近 4 轮历史（多轮指代不同则不命中，安全）。"""
    import hashlib

    raw = query.strip() + "||" + json.dumps(history[-4:], ensure_ascii=False)
    return "anscache:" + hashlib.md5(raw.encode("utf-8")).hexdigest()


def _state_to_response(state: dict, session_id: str, duration_ms: int) -> ConsultationResponse:
    """AgentState → ConsultationResponse（字段对齐 R3 契约）。"""
    return ConsultationResponse(
        answer=state.get("final_answer", "") or state.get("knowledge_answer", ""),
        intent=state.get("intent", "general"),
        symptoms=state.get("symptoms", []),
        severity=state.get("severity", ""),
        departments=state.get("departments", []),
        medications=[
            m if isinstance(m, MedicationItem) else MedicationItem(**m)
            for m in state.get("medications", [])
        ],
        advice=state.get("advice", []),
        disclaimers=state.get("disclaimers", []),
        warnings=state.get("warnings", []),
        linked_entities=[
            le if isinstance(le, LinkedEntity) else LinkedEntity(**le)
            for le in state.get("linked_entities", [])
        ],
        trace=state.get("trace", []),
        duration_ms=duration_ms,
        session_id=session_id,
    )


async def _process_consult(request: ConsultationRequest) -> ConsultationResponse:
    """共用处理链：记忆读取 → Agent 图执行 → 双写历史。"""
    start = time.perf_counter()
    graph_app = _app_state.get("graph_app")
    if graph_app is None:
        raise HTTPException(status_code=503, detail="Agent 系统尚未就绪，请稍后再试")

    history = await asyncio.to_thread(history_store.load_context, request.session_id)
    logger.info("consult session=%s | history turns: %d", request.session_id, len(history))

    # 接入层语义缓存：相同问句+相同上下文直接返回（TTL 6h）
    akey = _answer_cache_key(request.query, history)
    cached_raw = await asyncio.to_thread(history_store.cache_get, akey)
    if cached_raw:
        try:
            cached = ConsultationResponse.model_validate_json(cached_raw)
            cached.duration_ms = int((time.perf_counter() - start) * 1000)
            logger.info("consult 缓存命中 session=%s (%dms)", request.session_id, cached.duration_ms)
            await asyncio.to_thread(
                history_store.save_turn,
                request.session_id, request.query, cached.answer, cached.intent,
            )
            return cached
        except Exception:  # noqa: BLE001
            pass

    try:
        state = await asyncio.wait_for(
            asyncio.to_thread(graph_app.invoke, {
                "query": request.query,
                "history": list(history),
                "disclaimers": [],
                "warnings": [],
                "retrieved_entities": [],
                "retrieved_contexts": [],
                "linked_entities": [],
                "trace": [],
            }),
            timeout=settings.api_timeout_seconds,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"处理超时（>{settings.api_timeout_seconds}s），请稍后重试或简化问题",
        )

    duration_ms = int((time.perf_counter() - start) * 1000)
    response = _state_to_response(state, request.session_id, duration_ms)

    # 答案缓存回填（仅有效回答）
    if response.answer:
        await asyncio.to_thread(
            history_store.cache_set, akey, response.model_dump_json(),
            _ANSWER_CACHE_TTL_HOURS,
        )

    # 双写历史（先 MySQL 后 Redis，失败降级不影响响应）
    await asyncio.to_thread(
        history_store.save_turn,
        request.session_id, request.query, response.answer, response.intent,
    )

    logger.info(
        "consult 完成 session=%s intent=%s duration=%dms disclaimer=%s",
        request.session_id, response.intent, duration_ms, bool(response.disclaimers),
    )
    return response


# ---------- 同步问诊 ----------
@router.post("/consult", response_model=ConsultationResponse)
async def consult(request: ConsultationRequest) -> ConsultationResponse:
    async with _semaphore:
        return await _process_consult(request)


# ---------- SSE 流式问诊 ----------
@router.post("/consult/stream")
async def consult_stream(request: ConsultationRequest) -> StreamingResponse:
    async def event_generator():
        """SSE 事件协议：progress{node,label,facts} → answer{完整JSON} → done [DONE]。"""
        yield f"event: progress\ndata: {json.dumps({'node': 'start', 'label': '开始处理', 'facts': []}, ensure_ascii=False)}\n\n"

        graph_app = _app_state.get("graph_app")
        if graph_app is None:
            yield f"event: error\ndata: {json.dumps({'message': 'Agent 系统尚未就绪'}, ensure_ascii=False)}\n\n"
            yield "event: done\ndata: [DONE]\n\n"
            return

        start = time.perf_counter()
        history = await asyncio.to_thread(history_store.load_context, request.session_id)

        # 语义缓存命中：直接吐 answer 帧（跳过 Agent 执行）
        akey = _answer_cache_key(request.query, history)
        cached_raw = await asyncio.to_thread(history_store.cache_get, akey)
        if cached_raw:
            try:
                cached = ConsultationResponse.model_validate_json(cached_raw)
                cached.duration_ms = int((time.perf_counter() - start) * 1000)
                await asyncio.to_thread(
                    history_store.save_turn,
                    request.session_id, request.query, cached.answer, cached.intent,
                )
                yield (
                    "event: answer\ndata: "
                    + json.dumps(cached.model_dump(), ensure_ascii=False)
                    + "\n\n"
                )
                yield "event: done\ndata: [DONE]\n\n"
                return
            except Exception:  # noqa: BLE001
                pass

        # LangGraph stream 双模式：updates 取节点进度、values 取终态
        # 同步生成器经 to_thread + 哨兵桥接（避免 StopIteration 穿越线程边界，PEP 479）
        sentinel = object()
        input_state = {
            "query": request.query,
            "history": list(history),
            "disclaimers": [],
            "warnings": [],
            "retrieved_entities": [],
            "retrieved_contexts": [],
            "linked_entities": [],
            "trace": [],
        }
        gen = graph_app.stream(input_state, stream_mode=["updates", "values"])

        async def next_chunk():
            return await asyncio.to_thread(lambda: next(gen, sentinel))

        final_state: dict = {}
        try:
            while True:
                chunk = await asyncio.wait_for(next_chunk(), timeout=settings.api_timeout_seconds)
                if chunk is sentinel:
                    break
                mode, payload = chunk
                if mode == "updates":
                    for node_name, update in (payload or {}).items():
                        if not update:
                            continue
                        facts = []
                        if node_name == "intent_classifier":
                            facts = [f"意图: {update.get('intent', '')}"]
                        elif node_name == "symptom_detector":
                            facts = [f"症状: {('、'.join(update.get('symptoms', []))) or '未检出'}"]
                        elif node_name == "department_recommender":
                            facts = [f"科室: {('、'.join(update.get('departments', []))) or '-'}"]
                        elif node_name == "medication_advisor":
                            facts = [f"药物候选: {len(update.get('medications', []))} 个"]
                        elif node_name == "medical_knowledge":
                            facts = [f"检索命中: {len(update.get('retrieved_contexts', []))} 条"]
                        label = _app_state.get("node_labels", {}).get(node_name, node_name)
                        data = json.dumps(
                            {"node": node_name, "label": label, "facts": facts},
                            ensure_ascii=False,
                        )
                        yield f"event: progress\ndata: {data}\n\n"
                elif mode == "values":
                    final_state = payload
        except asyncio.TimeoutError:
            yield f"event: error\ndata: {json.dumps({'message': '处理超时'}, ensure_ascii=False)}\n\n"
            yield "event: done\ndata: [DONE]\n\n"
            return
        except Exception as exc:  # noqa: BLE001
            logger.error("SSE 流处理失败: %s", exc)
            yield f"event: error\ndata: {json.dumps({'message': f'处理失败: {exc}'}, ensure_ascii=False)}\n\n"
            yield "event: done\ndata: [DONE]\n\n"
            return

        duration_ms = int((time.perf_counter() - start) * 1000)
        response = _state_to_response(final_state, request.session_id, duration_ms)
        await asyncio.to_thread(
            history_store.save_turn,
            request.session_id, request.query, response.answer, response.intent,
        )

        yield (
            "event: answer\ndata: "
            + json.dumps(response.model_dump(), ensure_ascii=False)
            + "\n\n"
        )
        yield "event: done\ndata: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------- 健康检查 / 统计 ----------
@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    retriever = get_shared_retriever()
    return HealthResponse(
        status="healthy" if _app_state.get("graph_app") is not None else "degraded",
        neo4j_connected=await asyncio.to_thread(verify_connectivity),
        vector_index_loaded=retriever.is_ready,
        agent_system_ready=_app_state.get("graph_app") is not None,
        version="1.0.0",
    )


_STATS_TTL_SECONDS = 30
_stats_cache: tuple[float, StatsResponse] | None = None


@router.get("/stats", response_model=StatsResponse)
async def stats() -> StatsResponse:
    """系统统计（30s TTL 缓存：图谱统计要打 Neo4j 多次查询，无需每次实时）。"""
    global _stats_cache
    import time as _time

    now = _time.monotonic()
    if _stats_cache is not None and now - _stats_cache[0] < _STATS_TTL_SECONDS:
        return _stats_cache[1]

    queries = GraphQueries()
    kg_stats = await asyncio.to_thread(queries.get_stats)
    retriever = get_shared_retriever()
    resp = StatsResponse(
        total_nodes=kg_stats.get("total_nodes", 0),
        total_relations=kg_stats.get("total_relations", 0),
        graph_density=kg_stats.get("graph_density", 0.0),
        node_stats=kg_stats.get("node_stats", {}),
        vector_index_size=retriever.index_size,
        agent_ready=_app_state.get("graph_app") is not None,
    )
    _stats_cache = (now, resp)
    return resp


# ---------- 对话历史 ----------
def _fmt(value) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S") if hasattr(value, "strftime") else str(value or "")


@router.get("/conversations", response_model=list[ConversationSummary])
def list_conversations() -> list[ConversationSummary]:
    return [
        ConversationSummary(
            id=row["id"], title=row["title"],
            created_at=_fmt(row.get("created_at")), updated_at=_fmt(row.get("updated_at")),
        )
        for row in history_store.list_conversations(limit=50)
    ]


@router.get("/conversations/{conversation_id}", response_model=list[MessageItem])
def get_conversation(conversation_id: str) -> list[MessageItem]:
    messages = history_store.get_messages(conversation_id)
    if not messages:
        raise HTTPException(status_code=404, detail="对话不存在")
    return [
        MessageItem(
            role=m["role"], content=m["content"],
            intent=m.get("intent", ""), created_at=_fmt(m.get("created_at")),
        )
        for m in messages
    ]


@router.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str) -> dict:
    ok = history_store.delete_conversation(conversation_id)
    if not ok:
        raise HTTPException(status_code=500, detail="删除失败")
    return {"deleted": True, "id": conversation_id}
