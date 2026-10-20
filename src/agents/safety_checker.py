"""安全检查 Agent —— 最重要的 Agent：纯规则引擎，不依赖 LLM（架构级兜底）。

设计依据：安全层必须确定、可审计——LLM 是被防护的对象，不能自己看守自己。
- 急症召回主信号 = intent==emergency（不用 answer 子串判断，避免欢迎语里的
  「120/急诊」误触发），症状关键词作兜底；
- 用药意图/药物列表 → 强制追加「遵医嘱/药师」提醒；
- 特殊人群（孕妇/儿童/老人）→ 特殊提示。
"""
from src.agents.state import AgentState, IntentType
from src.common.logger import setup_logger

logger = setup_logger("agent.safety", "agents.log")

DISCLAIMER = "🔴 重要声明：本建议仅供参考，不能替代专业医疗诊断。请及时就医，遵医嘱用药！"

EMERGENCY_WARNING = "🚨 您的症状可能属于急症，请立即拨打 120 或前往急诊科就医，不要延误！"

MEDICATION_SAFETY_REMINDER = "💊 用药安全提醒：请务必遵医嘱用药，用法用量与药物联用请咨询医生或药师。"

SPECIAL_POPULATION_REMINDERS = {
    "孕妇": "🤰 您提到的情况如涉及孕妇，用药需特别谨慎，请务必在产科医生指导下进行。",
    "怀孕": "🤰 您提到的情况如涉及孕妇，用药需特别谨慎，请务必在产科医生指导下进行。",
    "儿童": "👶 涉及儿童用药时，剂量与药物选择与成人不同，请务必遵儿科医嘱。",
    "老人": "👴 老年人用药需考虑肝肾功能，建议在医生指导下调整剂量。",
}

# 急症关键词 —— R1 单一事实来源：全项目唯一的急症词表。
# intent_classifier（短路升级）、symptom_detector（severity 判定）、
# graph.route_by_intent（急症强制升级门）一律 import 此列表，禁止再复制。
EMERGENCY_SYMPTOM_KEYWORDS = [
    "胸痛", "呼吸困难", "意识模糊", "昏迷", "抽搐", "大出血", "大量出血", "血流不止",
    "窒息", "中毒", "剧烈头痛", "晕厥", "心跳骤停", "高热惊厥",
]


class SafetyCheckerAgent:
    """安全检查（纯规则，无 LLM）。"""

    DISCLAIMER = DISCLAIMER

    @staticmethod
    def has_emergency_symptoms(symptoms: list[str], query: str = "") -> bool:
        text = query + "".join(symptoms)
        return any(kw in text for kw in EMERGENCY_SYMPTOM_KEYWORDS)

    def __call__(self, state: AgentState) -> dict:
        intent = state.get("intent", "")
        query = state.get("query", "")
        symptoms = state.get("symptoms", [])
        medications = state.get("medications", [])

        disclaimers: list[str] = [DISCLAIMER]
        warnings: list[str] = []

        # 急症：intent==emergency 为主信号（emit one warning），关键词兜底
        is_emergency = intent == IntentType.EMERGENCY.value or self.has_emergency_symptoms(
            symptoms, query
        )
        if is_emergency:
            warnings.append(EMERGENCY_WARNING)

        # 用药安全：用药意图或已有药物列表时强制提醒
        if intent == IntentType.MEDICATION.value or medications:
            disclaimers.append(MEDICATION_SAFETY_REMINDER)

        # 特殊人群提示：检查当前 query 以及多轮对话历史（保持特殊人群生理状态连续性）
        history = state.get("history", [])
        combined_text = query + " " + " ".join(str(t.get("content", "")) for t in history[-4:])
        for keyword, reminder in SPECIAL_POPULATION_REMINDERS.items():
            if keyword in combined_text:
                disclaimers.append(reminder)
                break

        logger.info(
            "安全检查: intent=%s emergency=%s disclaimers=%d warnings=%d",
            intent, is_emergency, len(disclaimers), len(warnings),
        )
        return {"disclaimers": disclaimers, "warnings": warnings}
