"""安全规则单测：免责声明 / 急症关键词 / 用药提醒 / 特殊人群（纯规则，无 LLM）。"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MODEL_API_KEY", "test-placeholder")

from src.agents.safety_checker import (  # noqa: E402
    DISCLAIMER,
    EMERGENCY_SYMPTOM_KEYWORDS,
    SafetyCheckerAgent,
)
from src.agents.state import IntentType  # noqa: E402

checker = SafetyCheckerAgent()


def test_disclaimer_always_added_for_medical_intent():
    out = checker({"intent": IntentType.APPOINTMENT.value, "query": "头痛", "symptoms": ["头痛"]})
    assert DISCLAIMER in out["disclaimers"]
    assert out["warnings"] == []


def test_emergency_intent_fires_warning():
    out = checker({"intent": IntentType.EMERGENCY.value, "query": "难受", "symptoms": []})
    assert out["warnings"], "急症意图必须拉警告"
    assert DISCLAIMER in out["disclaimers"]


def test_emergency_keyword_fallback_even_if_intent_wrong():
    """急症升级兜底：LLM 误判意图时，关键词仍要拉警告（含昏迷/中毒等全量词表）。"""
    out = checker({"intent": IntentType.GENERAL.value, "query": "家人昏迷不醒了", "symptoms": []})
    assert out["warnings"], "急症关键词必须触发警告"


def test_emergency_keyword_list_is_full():
    """词表单一来源必须覆盖评审指出的缺口词。"""
    for kw in ("昏迷", "抽搐", "中毒", "窒息", "心跳骤停"):
        assert kw in EMERGENCY_SYMPTOM_KEYWORDS


def test_medication_reminder_for_medication_intent():
    out = checker({"intent": IntentType.MEDICATION.value, "query": "吃什么药", "symptoms": []})
    assert any("遵医嘱" in d for d in out["disclaimers"])


def test_medication_reminder_for_meds_in_state():
    out = checker({"intent": IntentType.KNOWLEDGE.value, "query": "高血压",
                   "symptoms": [], "medications": [{"name": "氨氯地平"}]})
    assert any("遵医嘱" in d for d in out["disclaimers"])


def test_special_population_prompt():
    out = checker({"intent": IntentType.MEDICATION.value, "query": "孕妇感冒吃什么药",
                   "symptoms": [], "medications": []})
    assert any("孕妇" in d for d in out["disclaimers"])


def test_calm_general_case_no_warning():
    """普通问候不该拉急症警告（避免 120 字样误触发的回归）。"""
    out = checker({"intent": IntentType.GENERAL.value, "query": "你好，请问挂什么科要多少钱",
                   "symptoms": []})
    assert out["warnings"] == []
