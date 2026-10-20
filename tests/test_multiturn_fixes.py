"""多轮修复验证单测：
1. 生理状态与病史补充（"我说了我是孕妇"、"怀孕前就有的糖尿病，目前孕5周"）不掉入 general
2. 多轮特殊人群（孕妇）免责提示跨轮次保持
3. 用药建议精准命中问句中的疾病（糖尿病），不被历史症状（头晕）混淆
4. 孕期用药高风险阻断与警示提醒
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unittest.mock import MagicMock, patch
from src.agents.intent_classifier import intent_classifier
from src.agents.safety_checker import SafetyCheckerAgent
from src.agents.medication_advisor import MedicationAdvisor


def test_intent_clarifications():
    # 模拟 LLM 误判 general 时，规则引擎纠偏
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content='{"intent": "general", "confidence": 0.5}')

    with patch("src.agents.intent_classifier.get_llm", return_value=mock_llm):
        # 1. 用户强调孕妇身份
        r1 = intent_classifier({
            "query": "我说了我是孕妇",
            "history": [{"role": "user", "content": "头晕该吃什么药"}, {"role": "assistant", "content": "..."}]
        })
        assert r1["intent"] == "medication", f"Expected medication, got {r1}"

        # 2. 用户陈述既往病史与孕周
        r2 = intent_classifier({
            "query": "怀孕前就有的糖尿病，目前孕5周",
            "history": []
        })
        assert r2["intent"] in ("knowledge", "medication"), f"Expected knowledge/medication, got {r2}"

        # 3. 用户纠正自身疾病
        r3 = intent_classifier({
            "query": "我说的是我有糖尿病",
            "history": [{"role": "user", "content": "头晕吃什么药"}]
        })
        assert r3["intent"] == "medication", f"Expected medication, got {r3}"


def test_safety_checker_history_persistence():
    checker = SafetyCheckerAgent()
    # 当前 query 没有"孕妇"，但 history 中明确是孕妇
    state = {
        "intent": "medication",
        "query": "我有糖尿病，我应该吃什么药",
        "history": [
            {"role": "user", "content": "我是孕妇，但我头有点晕，我该吃什么药"},
            {"role": "assistant", "content": "建议注意休息..."},
        ],
        "symptoms": [],
        "medications": [],
    }
    out = checker(state)
    assert any("孕妇" in d for d in out["disclaimers"]), "多轮历史中的孕妇身份提示必须在后续轮次中持久保持！"


def test_medication_advisor_targets_disease_and_pregnancy():
    advisor = MedicationAdvisor()

    # Mock GraphQueries
    mock_queries = MagicMock()
    mock_queries.match_disease_in_text.return_value = "糖尿病"
    mock_queries.get_disease_info.return_value = {
        "name": "糖尿病",
        "medications": ["二甲双胍", "格列美脲", "胰岛素"]
    }
    advisor.queries = mock_queries

    state = {
        "query": "我有糖尿病，我应该吃什么药",
        "symptoms": ["头晕"],  # 模拟上文遗留的历史症状
        "history": [
            {"role": "user", "content": "我是孕妇，头晕吃什么药"},
            {"role": "assistant", "content": "..."}
        ]
    }

    # Mock LLM
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content='{"medications": [{"name": "胰岛素", "purpose": "降糖", "side_effects": "低血糖"}]}')

    with patch("src.agents.medication_advisor.get_llm", return_value=mock_llm):
        out = advisor(state)
        meds = out["medications"]
        assert len(meds) > 0
        assert meds[0]["name"] == "胰岛素", f"Expected 胰岛素 for 糖尿病, got {meds}"
        assert "孕" in meds[0]["reminder"], f"Expected pregnancy warning in reminder, got {meds[0]['reminder']}"


if __name__ == "__main__":
    test_intent_clarifications()
    test_safety_checker_history_persistence()
    test_medication_advisor_targets_disease_and_pregnancy()
    print("ALL MULTI-TURN FIXES TESTS PASSED SUCCESSFULLY!")

