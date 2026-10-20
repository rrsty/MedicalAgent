import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unittest.mock import MagicMock, patch
from src.agents.intent_classifier import intent_classifier


def run_tests():
    # 1. 模拟 LLM 误判 general 时，医药概念兜底纠偏为 knowledge
    mock_llm_general = MagicMock()
    mock_llm_general.invoke.return_value = MagicMock(content='{"intent": "general", "confidence": 0.5}')

    with patch("src.agents.intent_classifier.get_llm", return_value=mock_llm_general):
        r1 = intent_classifier({"query": "司美格鲁肽是什么"})
        assert r1["intent"] == "knowledge", f"Expected knowledge, got {r1}"
        assert r1["confidence"] >= 0.85

        r2 = intent_classifier({"query": "司美格鲁肽"})
        assert r2["intent"] == "knowledge", f"Expected knowledge, got {r2}"

        r3 = intent_classifier({"query": "阿司匹林有什么用"})
        assert r3["intent"] == "knowledge", f"Expected knowledge, got {r3}"

        r4 = intent_classifier({"query": "今天天气怎么样"})
        assert r4["intent"] == "general", f"Expected general, got {r4}"

        r5 = intent_classifier({"query": "刚才我说的是什么症状？"})
        assert r5["intent"] == "general", f"Expected general, got {r5}"

        # 验证多轮生理状态/病史补充纠偏（绝不落入 general）
        r8 = intent_classifier({
            "query": "我说了我是孕妇",
            "history": [{"role": "user", "content": "头晕吃什么药"}, {"role": "assistant", "content": "..."}]
        })
        assert r8["intent"] == "medication", f"Expected medication, got {r8}"

        r9 = intent_classifier({"query": "怀孕前就有的糖尿病，目前孕5周"})
        assert r9["intent"] in ("knowledge", "medication"), f"Expected medical intent, got {r9}"

        r10 = intent_classifier({
            "query": "我说的是我有糖尿病",
            "history": [{"role": "user", "content": "吃什么药"}, {"role": "assistant", "content": "..."}]
        })
        assert r10["intent"] == "medication", f"Expected medication, got {r10}"

    # 2. 模拟 LLM 正常识别
    mock_llm_normal = MagicMock()
    mock_llm_normal.invoke.return_value = MagicMock(content='{"intent": "knowledge", "confidence": 0.95}')

    with patch("src.agents.intent_classifier.get_llm", return_value=mock_llm_normal):
        r6 = intent_classifier({"query": "司美格鲁肽是什么"})
        assert r6["intent"] == "knowledge"
        assert r6["confidence"] == 0.95

    # 3. 急症短路验证
    r7 = intent_classifier({"query": "我突然剧烈胸痛呼吸困难"})
    assert r7["intent"] == "emergency"
    assert r7["confidence"] == 0.99

    print("ALL INTENT CLASSIFIER TESTS PASSED!")


if __name__ == "__main__":
    run_tests()
