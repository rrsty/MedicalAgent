"""LangGraph 状态图编排 —— 系统中枢（9 个 Agent + 条件路由）。

路由表：
  appointment : 症状检测 → 科室推荐 → 就医指导 → 安全检查 → 答案融合
  medication  : 症状检测 → 用药建议（无症状转医学知识）→ 安全检查 → 答案融合
  knowledge   : 医学知识(grounded-ReAct) → 安全检查 → 答案融合
  emergency   : 安全检查（急症警告）→ 答案融合
  general     : 通用对话 → 直接结束（不经安全检查）

关键设计：除通用对话外，所有医疗流程最终汇聚「安全检查」，强制附加免责声明。
「追问」不是独立路由——信息不足由科室推荐置 needs_clarification，答案融合输出追问话术。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langgraph.graph import END, START, StateGraph

from src.agents.answer_fusion import AnswerFusion
from src.agents.department_recommender import DepartmentRecommender
from src.agents.general_chat import general_chat
from src.agents.intent_classifier import intent_classifier, is_potential_medical_query
from src.agents.medical_knowledge import MedicalKnowledgeAgent
from src.agents.medication_advisor import MedicationAdvisor
from src.agents.pre_visit_advisor import pre_visit_advisor
from src.agents.safety_checker import EMERGENCY_SYMPTOM_KEYWORDS, SafetyCheckerAgent
from src.agents.state import AgentState, IntentType
from src.agents.symptom_detector import symptom_detector
from src.common.logger import setup_logger

logger = setup_logger("agents.graph", "agents.log")

# SSE progress 帧用的节点中文名（R1：只在此定义）
NODE_LABELS_ZH = {
    "intent_classifier": "意图识别",
    "symptom_detector": "症状检测",
    "department_recommender": "科室推荐",
    "medication_advisor": "用药建议",
    "medical_knowledge": "医学知识",
    "pre_visit_advisor": "就医指导",
    "safety_checker": "安全检查",
    "answer_fusion": "答案融合",
    "general_chat": "通用对话",
}


# ---------- 条件路由 ----------
def route_by_intent(state: AgentState) -> str:
    """意图识别后的一级路由（含急症强制升级门：纯规则，不依赖 LLM）。

    安全兜底：无论 LLM 把意图判成什么（含 general），只要问题命中急症词表
    （R1 唯一来源 safety_checker.EMERGENCY_SYMPTOM_KEYWORDS），一律强制升级到
    安全检查——general 分支不过安全检查，这条门堵住「寒暄式回答急症」的缺口。
    """
    intent = state.get("intent", IntentType.GENERAL.value)
    query = state.get("query", "")

    if intent != IntentType.EMERGENCY.value and any(
        kw in query for kw in EMERGENCY_SYMPTOM_KEYWORDS
    ):
        logger.warning("急症强制升级门触发（intent=%s）: %s", intent, query[:50])
        return "safety_checker"

    if intent == IntentType.APPOINTMENT.value:
        return "symptom_detector"      # 挂号：先检测症状
    if intent == IntentType.MEDICATION.value:
        return "symptom_detector"      # 用药：也先检测症状
    if intent == IntentType.KNOWLEDGE.value:
        return "medical_knowledge"     # 知识：直接强制检索
    if intent == IntentType.EMERGENCY.value:
        return "safety_checker"        # 急症：直奔安全检查

    # 医疗兜底拦截门：若被判为 general，但在流入 general_chat 前检出潜在医疗健康特征，强制拦截回流
    if is_potential_medical_query(query):
        logger.warning("触发路由层医疗兜底拦截门: general -> medical_knowledge | query=%s", query[:50])
        return "medical_knowledge"

    return "general_chat"              # 其他：通用对话


def route_after_symptom(state: AgentState) -> str:
    """症状检测后的二级路由（appointment vs medication，medication 无症状转知识）。"""
    intent = state.get("intent", "")
    symptoms = state.get("symptoms", [])

    if state.get("severity") == "emergency":
        return "safety_checker"        # 症状检测升级急症 → 安全检查

    if intent == IntentType.APPOINTMENT.value:
        return "department_recommender"
    if intent == IntentType.MEDICATION.value:
        # medication 无症状锚点 → 医学知识 grounded-ReAct（迭代收窄追问策略）
        return "medication_advisor" if symptoms else "medical_knowledge"
    return "safety_checker"


# ---------- 组装状态图 ----------
def build_app():
    graph = StateGraph(AgentState)

    graph.add_node("intent_classifier", intent_classifier)
    graph.add_node("symptom_detector", symptom_detector)
    graph.add_node("department_recommender", DepartmentRecommender())
    graph.add_node("medication_advisor", MedicationAdvisor())
    graph.add_node("medical_knowledge", MedicalKnowledgeAgent())
    graph.add_node("pre_visit_advisor", pre_visit_advisor)
    graph.add_node("safety_checker", SafetyCheckerAgent())
    graph.add_node("answer_fusion", AnswerFusion())
    graph.add_node("general_chat", general_chat)

    graph.add_edge(START, "intent_classifier")
    graph.add_conditional_edges(
        "intent_classifier",
        route_by_intent,
        {
            "symptom_detector": "symptom_detector",
            "medical_knowledge": "medical_knowledge",
            "safety_checker": "safety_checker",
            "general_chat": "general_chat",
        },
    )
    graph.add_conditional_edges(
        "symptom_detector",
        route_after_symptom,
        {
            "department_recommender": "department_recommender",
            "medication_advisor": "medication_advisor",
            "medical_knowledge": "medical_knowledge",
            "safety_checker": "safety_checker",
        },
    )
    graph.add_edge("department_recommender", "pre_visit_advisor")
    graph.add_edge("pre_visit_advisor", "safety_checker")
    graph.add_edge("medication_advisor", "safety_checker")
    graph.add_edge("medical_knowledge", "safety_checker")
    graph.add_edge("safety_checker", "answer_fusion")
    graph.add_edge("answer_fusion", END)
    graph.add_edge("general_chat", END)

    return graph.compile()


app = build_app()


def run(query: str, history: list | None = None) -> AgentState:
    """API 层调用入口（history 由 API 层按记忆机制读取后注入）。"""
    if history is None:
        history = []
    initial: AgentState = {
        "query": query,
        "history": list(history),
        "disclaimers": [],
        "warnings": [],
        "retrieved_entities": [],
        "retrieved_contexts": [],
        "linked_entities": [],
        "trace": [],
    }
    return app.invoke(initial)


def get_mermaid_diagram() -> str:
    """静态 Mermaid 流程图（离线文档/前端展示用）。"""
    return """
flowchart TD
    START([START]) --> IC[意图识别 · LLM]
    IC -->|急症关键词强制升级门 · 纯规则| SC[安全检查 · 规则]
    IC -->|appointment / medication| SD[症状检测 · LLM+多轮上下文]
    IC -->|knowledge| MK[医学知识 · grounded-ReAct<br/>强制检索 FAISS+Neo4j<br/>可选ReAct多跳+Text2Cypher]
    IC -->|general| GC[通用对话 · LLM] --> END([END])

    SD -->|appointment| DR[科室推荐 · Neo4j→LLM→默认]
    SD -->|medication · 有症状| MA[用药建议 · 图谱候选+LLM过滤]
    SD -->|medication · 无症状| MK
    SD -->|急症关键词| SC

    DR --> PV[就医指导 · LLM+预设兜底] --> SC
    MA --> SC
    MK --> SC
    SC --> AF[答案融合 · LLM+确定性兜底] --> END
"""


if __name__ == "__main__":
    # 简单自检：无 API Key 时仅验证图编译与路由
    print(get_mermaid_diagram())
    if len(sys.argv) > 1 and sys.argv[1] == "--ask":
        question = sys.argv[2] if len(sys.argv) > 2 else "我最近头痛、头晕，应该挂什么科？"
        result = run(question)
        print(f"intent     : {result.get('intent')}")
        print(f"symptoms   : {result.get('symptoms')}")
        print(f"departments: {result.get('departments')}")
        print(f"disclaimers: {len(result.get('disclaimers', []))} 条")
        print(f"warnings   : {result.get('warnings')}")
        print(f"final_answer:\n{result.get('final_answer')}")
