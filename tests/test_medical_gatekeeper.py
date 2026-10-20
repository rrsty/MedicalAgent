"""医疗复检门禁单测：
验证口语化症状、隐晦表达、检查化验指标被大模型误判为 general 时，
能够成功触发医疗拦截门，回流至专业医疗分支，绝不落入闲聊。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unittest.mock import MagicMock, patch
from src.agents.intent_classifier import intent_classifier, is_potential_medical_query
from src.agents.graph import route_by_intent


def test_is_potential_medical_query_lexicons():
    # 1. 常见口语化、隐晦或非典型医疗问题（必须判为 True）
    assert is_potential_medical_query("胃里像火烧一样难受")
    assert is_potential_medical_query("心口窝总觉得憋屈")
    assert is_potential_medical_query("体检尿酸520怎么降下来")
    assert is_potential_medical_query("医生让我做胃镜我很害怕有必要做吗")
    assert is_potential_medical_query("骨头缝里透风是怎么回事")
    assert is_potential_medical_query("转氨酶偏高严重吗")
    assert is_potential_medical_query("脖子转不动咔咔响")

    # 2. 纯闲聊与历史测试（必须判为 False）
    assert not is_potential_medical_query("今天天气怎么样")
    assert not is_potential_medical_query("你好")
    assert not is_potential_medical_query("你是谁")
    assert not is_potential_medical_query("讲个笑话")
    assert not is_potential_medical_query("刚才我说的是什么症状？")


def test_intent_classifier_gatekeeper_recovery():
    # 模拟 LLM 犹豫或误判，返回 general (confidence=0.5)
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content='{"intent": "general", "confidence": 0.5}')

    with patch("src.agents.intent_classifier.get_llm", return_value=mock_llm):
        # 口语化症状纠偏为 knowledge
        r1 = intent_classifier({"query": "胃里像火烧一样难受"})
        assert r1["intent"] == "knowledge", f"Expected knowledge, got {r1}"
        assert r1["confidence"] >= 0.85

        # 检查指标纠偏为 knowledge
        r2 = intent_classifier({"query": "体检尿酸520严重吗"})
        assert r2["intent"] == "knowledge", f"Expected knowledge, got {r2}"

        # 挂号诉求纠偏为 appointment
        r3 = intent_classifier({"query": "脖子转不动去哪个科室看"})
        assert r3["intent"] == "appointment", f"Expected appointment, got {r3}"

        # 用药诉求纠偏为 medication
        r4 = intent_classifier({"query": "血糖偏高吃什么药能降下来"})
        assert r4["intent"] == "medication", f"Expected medication, got {r4}"

        # 纯闲聊保持 general
        r5 = intent_classifier({"query": "今天天气真好啊"})
        assert r5["intent"] == "general", f"Expected general, got {r5}"


def test_route_by_intent_gatekeeper():
    # 路由层终审拦截门：即使 intent 被漏标为 general，带医学特征也绝不流进 general_chat
    state_medical = {"intent": "general", "query": "心口窝总觉得堵得慌"}
    assert route_by_intent(state_medical) == "medical_knowledge"

    state_chat = {"intent": "general", "query": "今天天气怎么样"}
    assert route_by_intent(state_chat) == "general_chat"


if __name__ == "__main__":
    test_is_potential_medical_query_lexicons()
    test_intent_classifier_gatekeeper_recovery()
    test_route_by_intent_gatekeeper()
    print("ALL MEDICAL GATEKEEPER TESTS PASSED SUCCESSFULLY!")

