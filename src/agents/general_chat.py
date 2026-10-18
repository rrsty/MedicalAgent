"""通用对话 Agent：处理非医疗问题，友好回复 + 引导使用医疗功能（直接到 END，不经安全检查）。"""
from src.agents.state import AgentState
from src.common.llm import get_llm
from src.common.logger import setup_logger

logger = setup_logger("agent.general", "agents.log")

SYSTEM_PROMPT = """你是「智慧问诊Agent系统」的助手。用户提出了非医疗相关的问题。

要求：
1. 友好、简短地回应用户
2. 引导用户使用医疗功能：看病挂号推荐、用药建议、医学知识解答
3. 不回答专业医疗问题（引导用户提出医疗相关问题）
4. 回答控制在 80 字以内"""


def general_chat(state: AgentState) -> dict:
    query = state.get("query", "")
    try:
        resp = get_llm(0.7).invoke([
            ("system", SYSTEM_PROMPT),
            ("human", query),
        ])
        answer = resp.content.strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("通用对话 LLM 失败: %s", exc)
        answer = (
            "您好！我是智慧问诊助手，可以帮您：\n- 🏥 看病挂号：根据症状推荐科室\n"
            "- 💊 用药建议：查询药物信息\n- 📚 医学知识：解答健康疑问\n\n"
            "请问有什么可以帮您？"
        )

    logger.info("通用对话完成")
    return {"final_answer": answer}
