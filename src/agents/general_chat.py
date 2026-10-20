"""通用对话 Agent：带上下文记忆的自然拟人回复（有温度、有常识、守底线）。"""
from src.agents.state import AgentState
from src.common.llm import get_llm
from src.common.logger import setup_logger

logger = setup_logger("agent.general", "agents.log")

SYSTEM_PROMPT = """你是一个充满人文关怀、专业严谨的「智慧问诊助手」。
当前用户处于通用交流或非典型问诊状态，请**结合最近的多轮对话历史**给出自然、流畅、得体的回复。

交互准则：
1. 【多轮承接】：若用户在礼貌道别、道谢或确认（如“好的/谢谢/明白了”），请真诚祝福其早日康复，叮嘱遵医嘱休息，切勿机械重复系统功能！
2. 【记忆回顾】：若用户在测试历史（如“刚才我说了什么/之前提到的病或症状”），请准确根据上下文总结回顾；
3. 【情绪共情】：若用户表达出对病情的焦虑、害怕、沮丧，先给予温暖安慰（“放轻松，身体不舒服容易让人焦虑，我们一步步来”），再温柔引导；
4. 【身份/能力问询】：诚实说明自己是辅助预问诊AI，能协助科室推荐和用药科普，但不能确诊和开处方；
5. 【日常闲聊】：日常问候或生活常识（天气/作息/饮食）可自然交谈，顺畅融入健康生活贴士；
6. 【医学红线】：涉及具体病情时可提供科学常识参考，但坚决不给出绝对的诊断结论，不替代医生开处方，并主动提醒“请以面诊医生建议为准”。"""


def general_chat(state: AgentState) -> dict:
    query = state.get("query", "")
    history = state.get("history", [])

    messages = [("system", SYSTEM_PROMPT)]
    # 注入最近 3 轮（6 条消息）上下文，确保多轮记忆与自然承接
    for turn in history[-6:]:
        role = "user" if turn.get("role") == "user" else "assistant"
        messages.append((role, str(turn.get("content", ""))[:400]))
    messages.append(("human", query))

    try:
        resp = get_llm(temperature=0.7).invoke(messages)
        answer = resp.content.strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("通用对话 LLM 失败: %s", exc)
        answer = "您好！我是智慧问诊助手，随时可以帮您梳理症状推荐科室、解答用药与健康常识。请问有什么不舒服吗？"

    logger.info("通用对话完成: query=%s", query[:30])
    return {"final_answer": answer}

