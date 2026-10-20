"""通用对话 Agent 单测：验证多轮上下文注入、情绪与礼貌承接。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unittest.mock import MagicMock, patch
from src.agents.general_chat import general_chat


def test_general_chat_injects_history():
    captured = {}

    class MockLLM:
        def invoke(self, messages):
            captured["messages"] = messages
            return MagicMock(content="不客气！祝您早日康复。")

    state = {
        "query": "谢谢医生，我知道了",
        "history": [
            {"role": "user", "content": "我头痛吃什么药好？"},
            {"role": "assistant", "content": "建议按医嘱使用布洛芬等药物，注意休息。"},
        ]
    }

    with patch("src.agents.general_chat.get_llm", return_value=MockLLM()):
        res = general_chat(state)
        assert "final_answer" in res
        assert "不客气" in res["final_answer"]

        # 验证 messages 中包含了历史消息与当前问句
        msgs = captured["messages"]
        assert len(msgs) == 4  # system + 2 history + human
        assert msgs[1][0] == "user" and "头痛" in msgs[1][1]
        assert msgs[2][0] == "assistant" and "布洛芬" in msgs[2][1]
        assert msgs[3][0] == "human" and "谢谢医生" in msgs[3][1]


def test_general_chat_fallback():
    mock_llm_fail = MagicMock()
    mock_llm_fail.invoke.side_effect = Exception("Network timeout")

    with patch("src.agents.general_chat.get_llm", return_value=mock_llm_fail):
        res = general_chat({"query": "你好"})
        assert "final_answer" in res
        assert len(res["final_answer"]) > 0


if __name__ == "__main__":
    test_general_chat_injects_history()
    test_general_chat_fallback()
    print("ALL GENERAL CHAT TESTS PASSED SUCCESSFULLY!")

