"""意图识别 Agent：LLM 分类 + Few-shot，输出 intent + confidence。"""
from config.settings import settings
from src.agents.safety_checker import EMERGENCY_SYMPTOM_KEYWORDS
from src.agents.state import AgentState, IntentType
from src.common.llm import get_llm
from src.common.utils import parse_llm_json
from src.common.logger import setup_logger

logger = setup_logger("agent.intent", "agents.log")

SYSTEM_PROMPT = """你是一个医疗问诊意图识别专家。请判断用户的问题属于以下哪种意图：
- appointment：看病挂号，询问应该看什么科、挂什么号
- medication：用药建议，询问吃什么药、怎么用药、药物副作用
- knowledge：医学知识，询问疾病定义、病因、预防、检查等知识
- emergency：急症，胸痛、呼吸困难、意识模糊、大出血等需要立即就医的紧急症状
- general：非医疗问题（问候、闲聊、天气等）

只输出 JSON：{"intent": "...", "confidence": 0.0~1.0}

示例：
1. "我最近头痛，应该挂什么科？" → {"intent": "appointment", "confidence": 0.95}
2. "高血压吃什么药好？" → {"intent": "medication", "confidence": 0.95}
3. "什么是糖尿病？" → {"intent": "knowledge", "confidence": 0.95}
4. "我突然胸痛、呼吸困难！" → {"intent": "emergency", "confidence": 0.98}
5. "今天天气怎么样？" → {"intent": "general", "confidence": 0.9}
6. "阿司匹林有什么副作用？" → {"intent": "medication", "confidence": 0.9}
7. "高血压是怎么引起的？" → {"intent": "knowledge", "confidence": 0.92}
8. "你好" → {"intent": "general", "confidence": 0.9}"""

_VALID_INTENTS = {i.value for i in IntentType}


def intent_classifier(state: AgentState) -> dict:
    """意图识别节点（LLM 分类 + 关键词兜底，解析失败降级 general）。"""
    query = state.get("query", "")

    # 急症关键词先行短路（R1 词表唯一来源 safety_checker；安全优先：LLM 误判也有兜底）
    if any(kw in query for kw in EMERGENCY_SYMPTOM_KEYWORDS):
        return {"intent": IntentType.EMERGENCY.value, "confidence": 0.99}

    try:
        messages = [("system", SYSTEM_PROMPT)]
        # 拼接最近对话作为上下文（多轮指代消解："它/这个病"）
        for turn in state.get("history", [])[-4:]:
            role = "user" if turn.get("role") == "user" else "assistant"
            messages.append((role, turn.get("content", "")[:300]))
        messages.append(("human", query))

        resp = get_llm(temperature=0.1).invoke(messages)
        data = parse_llm_json(resp.content) or {}
        intent = str(data.get("intent", "general")).lower()
        if intent not in _VALID_INTENTS:
            intent = IntentType.GENERAL.value
        confidence = float(data.get("confidence", 0.5))
    except Exception as exc:  # noqa: BLE001
        logger.warning("意图识别 LLM 失败，降级 general: %s", exc)
        intent, confidence = IntentType.GENERAL.value, 0.3

    logger.info("意图识别: %s (confidence=%.2f) | query=%s", intent, confidence, query[:50])
    return {"intent": intent, "confidence": confidence}
