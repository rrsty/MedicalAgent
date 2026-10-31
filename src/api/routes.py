"""API 路由：consult / consult/stream(SSE) / health / stats / conversations。

- POST /api/consult          同步问诊（Semaphore 并发限制 + 超时处理）
- POST /api/consult/stream   SSE 流式：progress（节点进度）→ answer（完整响应）→ done
- GET  /api/health           健康检查
- GET  /api/stats            系统统计
- GET/DELETE /api/conversations*  对话历史（读 MySQL，删除同步清 Redis）
"""
import asyncio
import hashlib
import json
import time

from fastapi import APIRouter, HTTPException, Request
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


class TimeoutBudget:
    """基于单调时钟的全链路超时预算管理器。"""

    def __init__(self, timeout_seconds: float):
        self.timeout_seconds = float(timeout_seconds)
        self.start_monotonic = time.monotonic()
        self.deadline = self.start_monotonic + self.timeout_seconds

    @property
    def remaining(self) -> float:
        """获取当前剩余超时秒数（下限为 0.0）。"""
        return max(0.0, self.deadline - time.monotonic())

    def check(self) -> float:
        """检查并返回剩余秒数；若已耗尽则抛出 asyncio.TimeoutError。"""
        rem = self.remaining
        if rem <= 0.0:
            raise asyncio.TimeoutError("全链路处理超时（预算已耗尽）")
        return rem


class SafeStreamingResponse(StreamingResponse):
    """确保在客户端断开或流结束时必定执行 body_iterator.aclose() 的流式响应。"""

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            if hasattr(self.body_iterator, "aclose") and callable(self.body_iterator.aclose):
                try:
                    await self.body_iterator.aclose()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("SafeStreamingResponse aclose exception: %s", exc)


_ANSWER_CACHE_TTL_HOURS = 6
_SYSTEM_VERSION = "v1"


def _answer_cache_key(query: str, history: list, model_name: str | None = None) -> str:
    """精确内容缓存键：归一化问句 + 规范化全量历史上下文 + 模型/系统版本（SHA-256 哈希）。

    完全剔除 session_id，保证多会话安全共享，同时杜绝上下文截断造成的哈希碰撞。
    """
    clean_query = query.strip()
    norm_history: list[dict[str, str]] = []
    for m in (history or []):
        if isinstance(m, dict):
            role = str(m.get("role", "")).strip().lower()
            content = str(m.get("content", "")).strip()
        elif hasattr(m, "role") and hasattr(m, "content"):
            role = str(getattr(m, "role", "")).strip().lower()
            content = str(getattr(m, "content", "")).strip()
        else:
            continue
        if content:
            norm_history.append({"role": role, "content": content})

    payload = {
        "query": clean_query,
        "context": norm_history,
        "model": model_name or getattr(settings, "model_name", "default"),
        "version": _SYSTEM_VERSION,
    }
    raw_str = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(raw_str.encode("utf-8")).hexdigest()
    return f"anscache:{digest}"


_exact_answer_cache_key = _answer_cache_key


def _serialize_cache_payload(response: ConsultationResponse) -> str:
    """序列化可复用答案载荷（剥离 session_id 与 duration_ms）。"""
    data = response.model_dump(exclude={"session_id", "duration_ms"})
    return json.dumps(data, ensure_ascii=False)


def _deserialize_cache_payload(raw_json: str, session_id: str, duration_ms: int) -> ConsultationResponse:
    """反序列化并绑定当前请求者的 session_id 与当前实际延迟。"""
    data = json.loads(raw_json)
    data["session_id"] = session_id
    data["duration_ms"] = duration_ms
    return ConsultationResponse.model_validate(data)


def _state_to_response(state: dict, session_id: str, duration_ms: int) -> ConsultationResponse:
    """AgentState → ConsultationResponse（字段对齐 R3 契约）。"""
    return ConsultationResponse(
        answer=state.get("final_answer", "") or state.get("knowledge_answer", ""),
        intent=state.get("intent", "general"),
        symptoms=state.get("symptoms", []),
        severity=state.get("severity", ""),
        departments=state.get("departments", []),
        medications=[
            m if isinstance(m, MedicationItem) else MedicationItem.model_validate(m)
            for m in state.get("medications", [])
        ],
        advice=state.get("advice", []),
        disclaimers=state.get("disclaimers", []),
        warnings=state.get("warnings", []),
        linked_entities=[
            le if isinstance(le, LinkedEntity) else LinkedEntity.model_validate(le)
            for le in state.get("linked_entities", [])
        ],
        trace=state.get("trace", []),
        duration_ms=duration_ms,
        session_id=session_id,
        needs_clarification=state.get("needs_clarification", False),
        retrieved_contexts=state.get("retrieved_contexts", []),
    )


async def _process_consult(request: ConsultationRequest) -> ConsultationResponse:
    """共用处理链：记忆读取 → Agent 图执行 → 双写历史（受 TimeoutBudget 全程管控）。"""
    graph_app = _app_state.get("graph_app")
    if graph_app is None:
        raise HTTPException(status_code=503, detail="Agent 系统尚未就绪，请稍后再试")

    budget = TimeoutBudget(settings.api_timeout_seconds)
    start_mono = budget.start_monotonic

    try:
        # 1. 预算内加载上下文（优先注入历史）
        if request.history is not None:
            history = request.history
        else:
            history = await asyncio.wait_for(
                asyncio.to_thread(history_store.load_context, request.session_id),
                timeout=budget.check(),
            )
        logger.info("consult session=%s | history turns: %d", request.session_id, len(history))

        # 2. 精确缓存检查
        akey = _answer_cache_key(request.query, history)
        cached_raw = None
        try:
            cached_raw = await asyncio.wait_for(
                asyncio.to_thread(history_store.cache_get, akey),
                timeout=budget.check(),
            )
        except asyncio.TimeoutError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("缓存读取异常: %s", exc)

        if cached_raw:
            try:
                duration_ms = int((time.monotonic() - start_mono) * 1000)
                cached_resp = _deserialize_cache_payload(cached_raw, request.session_id, duration_ms)
                logger.info("consult 缓存命中 session=%s (%dms)", request.session_id, cached_resp.duration_ms)
                try:
                    await asyncio.to_thread(
                        history_store.save_turn,
                        request.session_id, request.query, cached_resp.answer, cached_resp.intent,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("缓存命中保存历史失败: %s", exc)
                return cached_resp
            except Exception as exc:  # noqa: BLE001
                logger.warning("缓存反序列化失败: %s", exc)

        # 3. 剩余预算内执行 Agent 状态图
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
            timeout=budget.check(),
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"处理超时（>{settings.api_timeout_seconds}s），请稍后重试或简化问题",
        )

    duration_ms = int((time.monotonic() - start_mono) * 1000)
    response = _state_to_response(state, request.session_id, duration_ms)

    # 4. 回填缓存（脱敏 session_id）与双写历史
    if response.answer:
        payload_str = _serialize_cache_payload(response)
        try:
            await asyncio.to_thread(
                history_store.cache_set, akey, payload_str, _ANSWER_CACHE_TTL_HOURS,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("答案缓存写入失败: %s", exc)

    try:
        await asyncio.to_thread(
            history_store.save_turn,
            request.session_id, request.query, response.answer, response.intent,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("历史记录保存失败: %s", exc)

    logger.info(
        "consult 完成 session=%s intent=%s duration=%dms disclaimer=%s",
        request.session_id, response.intent, duration_ms, bool(response.disclaimers),
    )
    return response


# ---------- 同步问诊 ----------
@router.post("/consult", response_model=ConsultationResponse)
async def consult(request: ConsultationRequest) -> ConsultationResponse:
    if _semaphore is None:
        raise HTTPException(status_code=503, detail="系统未初始化")
    async with _semaphore:
        return await _process_consult(request)


# ---------- SSE 流式问诊 ----------
@router.post("/consult/stream")
async def consult_stream(
    request: Request,
    body: ConsultationRequest | None = None,
) -> StreamingResponse:
    if _semaphore is None:
        raise HTTPException(status_code=503, detail="系统未初始化")

    if isinstance(request, ConsultationRequest):
        actual_body = request
        http_request = None
    else:
        actual_body = body
        http_request = request

    if actual_body is None:
        raise HTTPException(status_code=422, detail="缺少请求体数据")

    async def event_generator():
        """SSE 事件协议：progress{node,label,facts} → answer{完整JSON} → done [DONE]。"""
        # 并发信号量许可覆盖流式执行全生命周期
        async with _semaphore:
            budget = TimeoutBudget(settings.api_timeout_seconds)
            start_mono = budget.start_monotonic

            yield f"event: progress\ndata: {json.dumps({'node': 'start', 'label': '开始处理', 'facts': []}, ensure_ascii=False)}\n\n"

            graph_app = _app_state.get("graph_app")
            if graph_app is None:
                yield f"event: error\ndata: {json.dumps({'message': 'Agent 系统尚未就绪'}, ensure_ascii=False)}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
                return

            # 1. 预算内加载上下文
            try:
                if actual_body.history is not None:
                    history = actual_body.history
                else:
                    history = await asyncio.wait_for(
                        asyncio.to_thread(history_store.load_context, actual_body.session_id),
                        timeout=budget.check(),
                    )
            except asyncio.TimeoutError:
                yield f"event: error\ndata: {json.dumps({'message': '加载上下文超时'}, ensure_ascii=False)}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
                return
            except Exception as exc:  # noqa: BLE001
                logger.error("加载上下文失败: %s", exc)
                yield f"event: error\ndata: {json.dumps({'message': f'加载上下文失败: {exc}'}, ensure_ascii=False)}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
                return

            # 2. 精确缓存命中检查
            akey = _answer_cache_key(actual_body.query, history)
            cached_raw = None
            try:
                cached_raw = await asyncio.wait_for(
                    asyncio.to_thread(history_store.cache_get, akey),
                    timeout=budget.check(),
                )
            except asyncio.TimeoutError:
                yield f"event: error\ndata: {json.dumps({'message': '缓存查询超时'}, ensure_ascii=False)}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("流式缓存读取异常: %s", exc)

            if cached_raw:
                try:
                    duration_ms = int((time.monotonic() - start_mono) * 1000)
                    cached_resp = _deserialize_cache_payload(cached_raw, actual_body.session_id, duration_ms)
                    try:
                        await asyncio.to_thread(
                            history_store.save_turn,
                            actual_body.session_id, actual_body.query, cached_resp.answer, cached_resp.intent,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("缓存命中保存历史失败: %s", exc)

                    logger.info("consult_stream 缓存命中 session=%s (%dms)", actual_body.session_id, cached_resp.duration_ms)
                    yield (
                        "event: answer\ndata: "
                        + json.dumps(cached_resp.model_dump(), ensure_ascii=False)
                        + "\n\n"
                    )
                    yield "event: done\ndata: [DONE]\n\n"
                    return
                except Exception as exc:  # noqa: BLE001
                    logger.warning("流式缓存反序列化失败: %s", exc)

            # 3. LangGraph stream 双模式：updates 取节点进度、values 取终态
            sentinel = object()
            input_state = {
                "query": actual_body.query,
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
            disconnected = False
            try:
                while True:
                    if http_request is not None and await http_request.is_disconnected():
                        logger.warning("客户端断开连接，提前终止流式执行: session=%s", actual_body.session_id)
                        disconnected = True
                        break

                    step_timeout = budget.check()
                    chunk = await asyncio.wait_for(next_chunk(), timeout=step_timeout)
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
            finally:
                try:
                    if hasattr(gen, "close"):
                        gen.close()
                except Exception:
                    pass

            if disconnected:
                return

            # 4. 正常完成处理：格式化响应、回填缓存与保存历史
            duration_ms = int((time.monotonic() - start_mono) * 1000)
            response = _state_to_response(final_state, actual_body.session_id, duration_ms)

            # 对齐普通接口：成功生成后回填答案缓存！
            if response.answer:
                payload_str = _serialize_cache_payload(response)
                try:
                    await asyncio.to_thread(
                        history_store.cache_set, akey, payload_str, _ANSWER_CACHE_TTL_HOURS,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("SSE 答案缓存写入失败: %s", exc)

            try:
                await asyncio.to_thread(
                    history_store.save_turn,
                    actual_body.session_id, actual_body.query, response.answer, response.intent,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("SSE 历史记录保存失败: %s", exc)

            yield (
                "event: answer\ndata: "
                + json.dumps(response.model_dump(), ensure_ascii=False)
                + "\n\n"
            )
            yield "event: done\ndata: [DONE]\n\n"

    return SafeStreamingResponse(
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
