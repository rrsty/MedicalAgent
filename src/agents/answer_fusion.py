"""答案融合 Agent：整合所有 Agent 输出为一条连贯回答（LLM 生成 + 确定性兜底）。

- needs_clarification 置位时输出追问话术（「追问」不是独立路由）；
- 确保免责声明在末尾；LLM 失败时用确定性模板拼装，绝不丢信息。
"""
import json

from src.agents.state import AgentState
from src.common.llm import get_llm
from src.common.logger import setup_logger

logger = setup_logger("agent.fusion", "agents.log")

FUSION_SYSTEM_PROMPT = """你是一个专业、友善的医疗客服。请把各个环节的结构化结果整合为一条连贯、自然的中文回答。

要求：
1. 保留所有关键信息（症状/科室/药物/建议），用 Markdown 组织（分节 + 列表）
2. 语气专业温和，不提供诊断结论
3. 急症警告放在最前面醒目位置
4. 免责声明固定放在最后
5. 直接输出回答正文，不要多余解释"""


class AnswerFusion:
    @staticmethod
    def _clarification_answer(state: AgentState) -> str:
        question = state.get(
            "clarification_question", "请问您有哪些具体症状或不适？"
        )
        disclaimer = "\n\n".join(state.get("disclaimers", []))
        return (
            "为了给您更准确的建议，我需要了解更多信息：\n\n"
            f"**{question}**\n\n"
            "您可以描述症状的部位、持续时间、严重程度等信息。\n\n"
            + (f"---\n{disclaimer}" if disclaimer else "")
        )

    @staticmethod
    def _deterministic_fallback(state: AgentState) -> str:
        """确定性拼装（LLM 失败兜底，绝不丢信息）。"""
        parts: list[str] = []
        intent = state.get("intent", "")

        if intent == "appointment":
            if state.get("symptoms"):
                parts.append(f"**检测到的症状**：{'、'.join(state['symptoms'])}")
            if state.get("departments"):
                parts.append(
                    "**推荐就诊科室**：" + " → ".join(
                        f"{d}（第{i+1}优先）" for i, d in enumerate(state["departments"])
                    )
                )
            if state.get("advice"):
                parts.append("**就医前建议**：\n" + "\n".join(f"- {a}" for a in state["advice"]))
        elif intent == "medication":
            for med in state.get("medications", []):
                line = f"- **{med.get('name', '')}**"
                if med.get("purpose"):
                    line += f"：{med['purpose']}"
                if med.get("side_effects"):
                    line += f"（常见副作用：{med['side_effects']}）"
                parts.append(line)
        elif state.get("knowledge_answer"):
            parts.append(state["knowledge_answer"])

        if state.get("warnings"):
            parts.insert(0, "\n".join(state["warnings"]))

        if not parts:
            parts.append("抱歉，暂时无法处理您的问题，请尝试换一种提问方式。")

        disclaimers = state.get("disclaimers", [])
        if disclaimers:
            parts.append("---\n" + "\n\n".join(disclaimers))
        return "\n\n".join(parts)

    def __call__(self, state: AgentState) -> dict:
        if state.get("needs_clarification"):
            return {"final_answer": self._clarification_answer(state)}

        # emergency：安全警告已足够，直接确定性拼装（快，且绝不稀释警告）
        if state.get("intent") == "emergency":
            return {"final_answer": self._deterministic_fallback(state)}

        try:
            context_sections = {
                "intent": state.get("intent", ""),
                "symptoms": state.get("symptoms", []),
                "departments": state.get("departments", []),
                "medications": state.get("medications", []),
                "advice": state.get("advice", []),
                "knowledge_answer": state.get("knowledge_answer", ""),
                "warnings": state.get("warnings", []),
                "disclaimers": state.get("disclaimers", []),
            }
            resp = get_llm(0.5).invoke([
                ("system", FUSION_SYSTEM_PROMPT),
                ("human", json.dumps(context_sections, ensure_ascii=False)),
            ])
            final_answer = resp.content.strip()
            if not final_answer:
                raise ValueError("LLM 返回空回答")
            # 确保免责声明在末尾（LLM 可能漏掉）
            disclaimers = state.get("disclaimers", [])
            if disclaimers and not any(d[:12] in final_answer for d in disclaimers):
                final_answer += "\n\n---\n" + "\n\n".join(disclaimers)
        except Exception as exc:  # noqa: BLE001
            logger.warning("答案融合 LLM 失败，使用确定性拼装兜底: %s", exc)
            final_answer = self._deterministic_fallback(state)

        logger.info("答案融合完成（%d 字）", len(final_answer))
        return {"final_answer": final_answer}
