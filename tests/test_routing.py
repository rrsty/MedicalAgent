"""路由逻辑单测：急症强制升级门（纯规则，不依赖 LLM；langgraph 缺失时跳过）。"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MODEL_API_KEY", "test-placeholder")

pytest.importorskip("langgraph")

from src.agents.graph import route_by_intent  # noqa: E402


def test_emergency_keyword_escalates_even_when_intent_is_general():
    """P0-3 回归：LLM 误判 general 时，急症关键词必须强制升级到安全检查。"""
    state = {"intent": "general", "query": "家人昏迷不醒了怎么办"}
    assert route_by_intent(state) == "safety_checker"


def test_emergency_keyword_escalates_medication_case():
    state = {"intent": "medication", "query": "误服大量安眠药中毒了该吃什么缓解"}
    assert route_by_intent(state) == "safety_checker"


def test_normal_routing_unchanged():
    assert route_by_intent({"intent": "appointment", "query": "头痛挂什么科"}) == "symptom_detector"
    assert route_by_intent({"intent": "knowledge", "query": "什么是糖尿病"}) == "medical_knowledge"
    assert route_by_intent({"intent": "general", "query": "你好"}) == "general_chat"
    assert route_by_intent({"intent": "emergency", "query": "难受"}) == "safety_checker"
