"""医学知识 Agent —— grounded-ReAct（强制检索打底 + 可选 ReAct 多跳 + 分风险兜底）。

三段式（企业级终态）：
  Step 1 强制混合检索（模型无法跳过）：FAISS 语义召回 → Neo4j 按实体类型扩展事实；
  Step 2 可选 ReAct 多跳：显式工具调用预算循环复用 tools.py，单步检索够不到的多跳补充取证
        （settings.enable_react_layer 开关，关闭即纯强制检索、更快更省）；
  Step 3 分风险兜底：检索为空时——一般知识题用 LLM 通用知识并标注「非本系统知识库内容」；
        高风险（相互作用/剂量/禁忌/过敏/饮酒）绝不臆测，直接建议咨询医生/药师。

只写 state["knowledge_answer"]（不写 final_answer），下游 safety_checker → answer_fusion 不动，
免责声明不丢。实体链接在检索后精修（锚点=检索命中实体名+症状+问题）。
"""
import json

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from config.settings import settings
from src.agents.state import AgentState
from src.common.context_window import compact_text, format_context_items, format_history_block
from src.common.llm import get_llm
from src.common.logger import setup_logger
from src.common.provenance import format_context_with_provenance
from src.knowledge_graph.graph_queries import GraphQueries
from src.vector_store.search import get_shared_retriever

logger = setup_logger("agent.knowledge", "agents.log")

SAFE_DEFERRAL = (
    "这个问题涉及用药安全的高风险信息（相互作用/剂量/禁忌/过敏/饮酒），"
    "本系统知识库中没有可靠的检索依据，为避免误导，请咨询医生或药师，"
    "或查阅权威药物数据库（如药品说明书）。切勿自行判断用药。"
)

_GENERAL_NO_KB_NOTICE = "（以下内容来自模型通用知识，非本系统知识库检索结果，仅供参考）"

HIGH_RISK_KEYWORDS = [
    "相互作用", "一起吃", "合用", "联用", "剂量", "用量多少", "吃多少",
    "过量", "禁忌", "过敏", "饮酒", "喝酒", "怀孕", "孕妇", "哺乳",
]

_GROUNDED_SYSTEM_PROMPT = """你是一个严谨的医学科普专家。请**仅依据**下面提供的检索到的知识库内容回答用户问题。

规则：
1. 只使用知识库内容中的事实，不添加知识库之外的信息
2. 检索内容不足以完整回答时，如实说明并只给出能确认的部分
3. 用通俗易懂的中文，适当使用 Markdown 列表组织
4. 涉及用药时提醒"遵医嘱"
5. 不提供诊断结论"""

_REACT_SYSTEM_PROMPT = """你是一个严谨的医学科普专家。回答事实性问题前，应先调用工具取证再作答；
工具查不到时如实说明，绝不编造。用通俗中文回答；涉及用药提醒"遵医嘱"。
可用工具：search_knowledge_graph（图谱实体查询）、semantic_search（语义检索）、
get_medication_info（药物详情）、natural_language_graph_query（自然语言多跳查图谱）。

下面是系统已强制检索到的知识库内容（优先使用）：
{contexts}"""


class MedicalKnowledgeAgent:
    """医学知识节点（graph.py 中节点名 medical_knowledge 保持不变）。"""

    def __init__(self):
        self.queries = GraphQueries()

    # ---------- Step 1 强制混合检索 ----------
    def _retrieve(self, query: str, symptoms: list[str]) -> tuple[list[dict], list[str]]:
        """FAISS 语义召回 → Neo4j 结构化扩展。全程 try/except，查空返回 ([], [])。"""
        try:
            hybrid = get_shared_retriever().hybrid_search(query)
            entities, contexts = hybrid["entities"], list(hybrid["contexts"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("强制检索失败（不阻断）: %s", exc)
            entities, contexts = [], []

        # 症状词补充检索（问题里没提到的症状线索）
        for symptom in symptoms[:3]:
            try:
                info = self.queries.get_disease_info(symptom)
                if not info:
                    diseases = self.queries.get_diseases_for_symptom(symptom, limit=5)
                    if diseases:
                        contexts.append(f"{symptom}（症状）可能提示: " + "、".join(diseases))
                else:
                    contexts.append(
                        f"{symptom}：科室 {'、'.join(info.get('departments', []))}；"
                        f"药物 {'、'.join(info.get('medications', [])[:6])}"
                    )
            except Exception as exc:  # noqa: BLE001
                logger.debug("症状补充检索失败: %s", exc)

        return entities, [c if "[数据源:" in c else format_context_with_provenance(c)
                          for c in contexts]

    # ---------- Step 2 可选 ReAct 多跳 ----------
    def _react_answer(
        self, query: str, contexts: list[str], history: list | None = None
    ) -> tuple[str, list[str], list[str]]:
        """在检索结果之上跑 ReAct；返回 (answer, tool_names, tool_messages_contexts)。"""
        from src.agents.tools import get_react_tools

        tools = {tool.name: tool for tool in get_react_tools()}
        model = get_llm(0.1).bind_tools(list(tools.values()))
        system = _REACT_SYSTEM_PROMPT.format(
            contexts=format_context_items(contexts) or "（无）"
        )
        history_block = _history_block(history)
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=f"{history_block}用户问题：{query}"),
        ]
        tool_names: list[str] = []
        tool_contexts: list[str] = []
        # Count every requested tool, including parallel and repeated calls.
        # A model may produce a final answer after the last permitted tool.
        for _ in range(settings.react_max_tool_calls + 1):
            message = model.invoke(messages)
            calls = message.tool_calls or []
            if not calls:
                return str(message.content or ""), tool_names, tool_contexts
            if len(tool_names) + len(calls) > settings.react_max_tool_calls:
                raise RuntimeError("ReAct tool-call budget exhausted; use grounded fallback")
            messages.append(message)
            for call in calls:
                name = call["name"]
                tool_names.append(name)
                tool = tools.get(name)
                content = (str(tool.invoke(call["args"])) if tool
                           else f"查询失败：未知工具 {name}")
                messages.append(ToolMessage(content=content, tool_call_id=call["id"]))
                if content and "查询失败" not in content and "未找到" not in content:
                    tool_contexts.append(content)
        raise RuntimeError("ReAct step budget exhausted; use grounded fallback")

    # ---------- Step 3 分风险兜底 ----------
    @staticmethod
    def _is_high_risk(query: str) -> bool:
        return any(kw in query for kw in HIGH_RISK_KEYWORDS)

    def _general_knowledge_fallback(self, query: str) -> str:
        try:
            resp = get_llm(0.3).invoke([
                (
                    "system",
                    "你是医学科普专家。知识库检索没有命中，请用你自己的通用医学知识谨慎回答。"
                    "不提供诊断结论，涉及用药提醒遵医嘱。用通俗中文回答。",
                ),
                ("human", query),
            ])
            return f"{_GENERAL_NO_KB_NOTICE}\n\n{resp.content}"
        except Exception as exc:  # noqa: BLE001
            logger.error("通用知识兜底失败: %s", exc)
            return "抱歉，本系统暂时无法回答该问题。涉及健康问题请及时就医咨询专业医生。"

    # ---------- 节点入口 ----------
    def __call__(self, state: AgentState) -> dict:
        query = state.get("query", "")
        symptoms = state.get("symptoms", [])
        history = state.get("history", [])
        trace: list[dict] = []

        # 多轮指代软消解（P1-9）：把最近一条用户消息拼进检索词，
        # 让「它该用什么药」能借上一轮「什么是糖尿病」命中图谱实体
        last_user = next(
            (t.get("content", "") for t in reversed(history) if t.get("role") == "user"),
            "",
        )
        retrieval_query = query
        if last_user and last_user != query:
            previous_user = compact_text(last_user, 240, tail_ratio=0.35)
            retrieval_query = f"{query} {previous_user}"
            trace.append({"node": "medical_knowledge", "step": "context_merge",
                          "context_from": compact_text(last_user, 80, tail_ratio=0.35)})

        # Step 1 强制混合检索（grounding 底线，模型无法跳过）
        entities, contexts = self._retrieve(retrieval_query, symptoms)
        trace.append({"node": "medical_knowledge", "step": "forced_retrieval",
                      "entities": len(entities), "contexts": len(contexts)})

        # Step 0 实体链接（检索后精修：锚点 = 检索命中实体名 + 症状 + 问题）
        linked: list[dict] = []
        try:
            anchor_terms = [e["name"] for e in entities] + symptoms + [query]
            linked = get_shared_retriever().link_entities(anchor_terms)
        except Exception as exc:  # noqa: BLE001
            logger.debug("实体链接失败（不阻断）: %s", exc)

        # 生成回答
        if contexts:
            # 有检索结果 → grounded；可选叠加 ReAct 多跳
            if settings.enable_react_layer:
                try:
                    react_answer, tool_names, tool_contexts = self._react_answer(
                        query, contexts, history
                    )
                    contexts.extend(tool_contexts)
                    if react_answer.strip():
                        answer = react_answer
                        trace.append({"node": "medical_knowledge", "step": "react",
                                      "tools": tool_names})
                    else:
                        answer = self._grounded_static_answer(query, contexts, history)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("ReAct 层失败，退回纯 grounded 生成: %s", exc)
                    trace.append({"node": "medical_knowledge", "step": "react_fallback",
                                  "error_type": type(exc).__name__})
                    answer = self._grounded_static_answer(query, contexts, history)
            else:
                answer = self._grounded_static_answer(query, contexts, history)
                trace.append({"node": "medical_knowledge", "step": "grounded_only"})
        elif self._is_high_risk(query):
            # 高风险 + 检索空 → 绝不臆测，直接拒答引导就医
            answer = SAFE_DEFERRAL
            trace.append({"node": "medical_knowledge", "step": "high_risk_refusal"})
        else:
            # 一般知识题 + 检索空 → LLM 通用知识并标注
            answer = self._general_knowledge_fallback(query)
            trace.append({"node": "medical_knowledge", "step": "general_fallback"})

        logger.info("医学知识回答完成（contexts=%d, linked=%d）", len(contexts), len(linked))
        return {
            "knowledge_answer": answer,
            "retrieved_entities": entities,
            "retrieved_contexts": contexts,
            "linked_entities": linked,
            "trace": trace,
        }

    # ---------- grounded 静态生成（ReAct 关闭/失败时的兜底）----------
    def _grounded_static_answer(
        self, query: str, contexts: list[str], history: list | None = None
    ) -> str:
        try:
            context_text = format_context_items(contexts)
            resp = get_llm(0.3).invoke([
                ("system", _GROUNDED_SYSTEM_PROMPT),
                (
                    "human",
                    f"{_history_block(history)}知识库内容：\n{context_text}\n\n用户问题：{query}",
                ),
            ])
            return resp.content
        except Exception as exc:  # noqa: BLE001
            logger.error("grounded 生成失败: %s", exc)
            summary = "\n".join(f"- {c}" for c in contexts[:5])
            return f"根据知识库检索到的信息：\n{summary}\n\n（回答生成暂时不可用，请以医生意见为准）"


def _history_block(history: list | None) -> str:
    """把最近对话拼成提示词片段（供指代消解）；空历史返回空串。"""
    return format_history_block(
        history,
        max_messages=4,
        per_message_chars=180,
        total_chars=720,
        title="最近对话（用于理解指代）：",
    )


# 解析器共享：把 trace 序列化成可读文本（answer_fusion / SSE 调试用）
def format_trace(trace: list[dict]) -> str:
    try:
        return json.dumps(trace, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return ""
