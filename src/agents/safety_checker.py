"""安全检查 Agent —— 最重要的 Agent：纯规则引擎，不依赖 LLM（架构级兜底）。

设计依据：安全层必须确定、可审计——LLM 是被防护的对象，不能自己看守自己。
- 急症召回主信号 = intent==emergency（不用 answer 子串判断，避免欢迎语里的
  「120/急诊」误触发），症状关键词作兜底；
- 用药意图/药物列表 → 强制追加「遵医嘱/药师」提醒；
- 特殊人群（孕妇/儿童/老人）→ 特殊提示。
"""
from src.agents.emergency_rules import (
    EMERGENCY_RULES,
    EMERGENCY_SYMPTOM_KEYWORDS,
    EmergencyEvaluationResult,
    emergency_engine,
)
from src.agents.state import AgentState, IntentType, Severity
from src.common.logger import setup_logger

logger = setup_logger("agent.safety", "agents.log")

DISCLAIMER = "🔴 重要声明：本建议仅供参考，不能替代专业医疗诊断。请及时就医，遵医嘱用药！"

EMERGENCY_WARNING = "🚨 您的症状可能属于急症，请立即拨打 120 或前往急诊科就医，不要延误！"

MEDICATION_SAFETY_REMINDER = "💊 用药安全提醒：请务必遵医嘱用药，用法用量与药物联用请咨询医生或药师。"

_PREGNANCY_REMINDER = "🤰 您提到的情况如涉及孕妇/孕哺期，用药需特别谨慎，请务必在产科医生指导下进行。"
_PEDIATRIC_REMINDER = "👶 涉及儿童/婴幼儿用药时，剂量与药物选择与成人不同，请务必遵儿科医嘱。"
_ELDERLY_REMINDER = "👴 老年人用药需考虑肝肾功能，建议在医生指导下调整剂量。"

SPECIAL_POPULATION_REMINDERS = {
    "孕妇": _PREGNANCY_REMINDER,
    "怀孕": _PREGNANCY_REMINDER,
    "孕期": _PREGNANCY_REMINDER,
    "孕周": _PREGNANCY_REMINDER,
    "哺乳期": _PREGNANCY_REMINDER,
    "哺乳": _PREGNANCY_REMINDER,
    "备孕": _PREGNANCY_REMINDER,
    "产后": _PREGNANCY_REMINDER,
    "儿童": _PEDIATRIC_REMINDER,
    "婴儿": _PEDIATRIC_REMINDER,
    "幼儿": _PEDIATRIC_REMINDER,
    "宝宝": _PEDIATRIC_REMINDER,
    "小孩": _PEDIATRIC_REMINDER,
    "老人": _ELDERLY_REMINDER,
    "老年人": _ELDERLY_REMINDER,
    "高龄": _ELDERLY_REMINDER,
}

# 急症规则与关键词 —— 单一事实来源从 emergency_rules 导入并对外提供
# intent_classifier（短路升级）、symptom_detector（severity 判定）、
# graph.route_by_intent（急症强制升级门）一律 import 此列表或使用 emergency_engine


class SafetyCheckerAgent:
    """安全检查（纯规则引擎，无 LLM）。基于演示规则库与局部否定过滤审计。"""

    DISCLAIMER = DISCLAIMER

    @staticmethod
    def evaluate_emergency(symptoms: list[str], query: str = "") -> EmergencyEvaluationResult:
        """调用急症规则引擎进行演示规则匹配与否定过滤分析。"""
        return emergency_engine.evaluate(query, symptoms)

    @classmethod
    def has_emergency_symptoms(cls, symptoms: list[str], query: str = "") -> bool:
        """向下兼容方法：判断是否存在未被否定排除的急症症状。"""
        res = cls.evaluate_emergency(symptoms, query)
        return res.is_emergency

    def __call__(self, state: AgentState) -> dict:
        intent = state.get("intent", "")
        query = state.get("query", "")
        symptoms = state.get("symptoms", [])
        medications = state.get("medications", [])

        disclaimers: list[str] = [DISCLAIMER]
        warnings: list[str] = []

        # 急症判定：显式急症意图 OR 症状检测升级 severity==emergency OR 规则引擎命中
        eval_result = self.evaluate_emergency(symptoms, query)
        is_emergency = (
            intent == IntentType.EMERGENCY.value
            or state.get("severity") == Severity.EMERGENCY.value
            or eval_result.is_emergency
        )
        if is_emergency:
            warnings.append(EMERGENCY_WARNING)

        # 用药安全：用药意图或已有药物列表时强制提醒
        if intent == IntentType.MEDICATION.value or medications:
            disclaimers.append(MEDICATION_SAFETY_REMINDER)

        # 特殊人群提示：仅检查当前 query 及历史中的**用户**消息，避免助手回复里的「孕妇慎用」误污染后续轮次
        history = state.get("history", [])
        user_history_text = " ".join(
            str(t.get("content", ""))
            for t in history[-6:]
            if isinstance(t, dict) and t.get("role") == "user"
        )
        combined_text = f"{query} {user_history_text}".strip()
        seen_reminders: set[str] = set()
        for keyword, reminder in SPECIAL_POPULATION_REMINDERS.items():
            if keyword in combined_text and reminder not in seen_reminders:
                disclaimers.append(reminder)
                seen_reminders.add(reminder)

        logger.info(
            "安全检查: intent=%s emergency=%s disclaimers=%d warnings=%d | %s",
            intent, is_emergency, len(disclaimers), len(warnings), eval_result.audit_summary,
        )
        result: dict = {
            "disclaimers": disclaimers,
            "warnings": warnings,
            "emergency_audit": {
                "is_emergency": is_emergency,
                "matched_rules": eval_result.matched_rules,
                "suppressed_rules": eval_result.suppressed_rules,
                "guidelines": eval_result.guideline_sources,
            },
        }
        if is_emergency:
            result["intent"] = IntentType.EMERGENCY.value
            result["severity"] = Severity.EMERGENCY.value
        return result
