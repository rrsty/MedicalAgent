"""Adversarial stress tests for M1:
- SSE stream parsing robustness in evals/runner.py
- ConsultationRequest & ConsultationResponse schema validation in src/api/models.py
- Contract checker execution across varied environment encodings
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from evals.runner import UNIFIED_SAMPLE_SCHEMA_FIELDS, evaluate_case
from src.api.models import (
    ConsultationRequest,
    ConsultationResponse,
    LinkedEntity,
    MedicationItem,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class MockSSEResponse:
    """Mock requests.Response for SSE stream testing."""

    def __init__(self, lines: list[str], status_code: int = 200):
        self.lines = lines
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"HTTP {self.status_code}")

    def iter_lines(self, decode_unicode=True):
        for line in self.lines:
            yield line

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def dummy_case():
    return {
        "id": "adv-case-01",
        "query": "头痛怎么办",
        "history": [],
        "gold": {
            "intent": "general",
            "is_emergency": False,
            "must_clarify": False,
            "must_disclaimer": True,
        },
    }


# =====================================================================
# Target 1: SSE Stream Parsing Adversarial Tests
# =====================================================================


def test_sse_compact_and_split_prefixes(dummy_case):
    """Test compact event/data without spaces (event:answer instead of event: answer)."""
    lines = [
        "event:progress",
        'data:{"node":"start"}',
        "",
        "event:answer",
        'data:{"answer":"回复内容","intent":"general","disclaimers":["参考"]}',
        "",
        "event:done",
        "data:[DONE]",
        "",
    ]
    with patch("requests.post", return_value=MockSSEResponse(lines)):
        res = evaluate_case(dummy_case, runner_mode="sse", base_url="http://mock")
        assert res.get("status") == "success"
        assert res.get("intent_pred") == "general"
        assert res.get("disclaimer_ok") is True


def test_sse_unexpected_events_and_comments(dummy_case):
    """Test parser ignores unexpected event types, pings, and SSE comment lines."""
    lines = [
        ": keep-alive comment",
        "event: ping",
        "data: heartbeat",
        "",
        "event: custom_debug",
        'data: {"step": 1}',
        "",
        "event: answer",
        "data: " + json.dumps({"answer": "回答完成", "intent": "general", "disclaimers": ["免责"]}, ensure_ascii=False),
        "",
        "event: done",
        "data: [DONE]",
        "",
    ]
    with patch("requests.post", return_value=MockSSEResponse(lines)):
        res = evaluate_case(dummy_case, runner_mode="sse", base_url="http://mock")
        assert res.get("status") == "success"
        assert res.get("answer") == "回答完成"


def test_sse_truncated_after_answer_without_done(dummy_case):
    """Test truncated stream that drops connection immediately after answer (missing event: done)."""
    lines = [
        "event: progress",
        'data: {"node": "retrieve"}',
        "",
        "event: answer",
        "data: " + json.dumps({"answer": "截断前回答", "intent": "general", "disclaimers": ["免责"]}, ensure_ascii=False),
        # Connection closed without done event
    ]
    with patch("requests.post", return_value=MockSSEResponse(lines)):
        res = evaluate_case(dummy_case, runner_mode="sse", base_url="http://mock")
        assert res.get("status") == "success"
        assert res.get("answer") == "截断前回答"


def test_sse_truncated_before_answer_handled_safely(dummy_case):
    """Test truncated stream before answer: cleanly caught into unified error schema."""
    lines = [
        "event: progress",
        'data: {"node": "retrieve"}',
        "",
        # Connection closed before answer
    ]
    with patch("requests.post", return_value=MockSSEResponse(lines)):
        res = evaluate_case(dummy_case, runner_mode="sse", base_url="http://mock")
        assert res.get("status") == "error"
        assert UNIFIED_SAMPLE_SCHEMA_FIELDS.issubset(res.keys())
        assert res.get("intent_ok") is False


def test_sse_missing_data_for_answer(dummy_case):
    """Test answer event with missing data: line."""
    lines = [
        "event: answer",
        "",
        "event: done",
        "data: [DONE]",
    ]
    with patch("requests.post", return_value=MockSSEResponse(lines)):
        res = evaluate_case(dummy_case, runner_mode="sse", base_url="http://mock")
        assert res.get("status") == "error"
        assert UNIFIED_SAMPLE_SCHEMA_FIELDS.issubset(res.keys())


def test_sse_server_error_event(dummy_case):
    """Test error event sent from server."""
    lines = [
        "event: error",
        'data: {"message": "Rate limit exceeded by OpenAI API"}',
        "",
        "event: done",
        "data: [DONE]",
    ]
    with patch("requests.post", return_value=MockSSEResponse(lines)):
        res = evaluate_case(dummy_case, runner_mode="sse", base_url="http://mock")
        assert res.get("status") == "error"
        assert res.get("error_type") == "model_error"
        assert UNIFIED_SAMPLE_SCHEMA_FIELDS.issubset(res.keys())


def test_sse_malformed_json_in_answer(dummy_case):
    """Test corrupted JSON in answer data is caught safely into error sample."""
    lines = [
        "event: answer",
        "data: {broken_json: 123",
        "",
        "event: done",
        "data: [DONE]",
    ]
    with patch("requests.post", return_value=MockSSEResponse(lines)):
        res = evaluate_case(dummy_case, runner_mode="sse", base_url="http://mock")
        assert res.get("status") == "error"
        assert UNIFIED_SAMPLE_SCHEMA_FIELDS.issubset(res.keys())


def test_sse_malformed_progress_ignored(dummy_case):
    """Test malformed progress event payload is ignored without aborting the evaluation."""
    lines = [
        "event: progress",
        "data: {invalid_progress_json",
        "",
        "event: answer",
        "data: " + json.dumps({"answer": "正常回答", "intent": "general"}, ensure_ascii=False),
        "",
        "event: done",
        "data: [DONE]",
    ]
    with patch("requests.post", return_value=MockSSEResponse(lines)):
        res = evaluate_case(dummy_case, runner_mode="sse", base_url="http://mock")
        assert res.get("status") == "success"
        assert res.get("answer") == "正常回答"


def test_sse_null_answer_payload_handled_safely(dummy_case):
    """Test answer event payload containing null answer field."""
    lines = [
        "event: answer",
        'data: {"answer": null, "intent": "general"}',
        "",
        "event: done",
        "data: [DONE]",
    ]
    with patch("requests.post", return_value=MockSSEResponse(lines)):
        res = evaluate_case(dummy_case, runner_mode="sse", base_url="http://mock")
        assert res.get("status") == "error"
        assert UNIFIED_SAMPLE_SCHEMA_FIELDS.issubset(res.keys())


# =====================================================================
# Target 2: Model Schema Validation & Serialization Adversarial Tests
# =====================================================================


def test_consultation_request_extra_fields():
    """Test ConsultationRequest allows extra unknown fields and ignores them."""
    req = ConsultationRequest.model_validate({
        "query": "头痛怎么办",
        "session_id": "sess-test-123",
        "unexpected_field": "val",
        "extra_dict": {"k": 1},
    })
    assert req.query == "头痛怎么办"
    assert req.session_id == "sess-test-123"
    assert "unexpected_field" not in req.model_dump()


def test_consultation_response_extra_fields():
    """Test ConsultationResponse allows extra unknown fields and ignores them."""
    resp = ConsultationResponse.model_validate({
        "answer": "回答内容",
        "unexpected_field": "val",
    })
    assert resp.answer == "回答内容"
    assert "unexpected_field" not in resp.model_dump()


def test_consultation_request_history_variations():
    """Test ConsultationRequest handles empty list, None, and omitted history."""
    req_empty = ConsultationRequest(query="头痛", session_id="sess-001", history=[])
    assert req_empty.history == []

    req_none = ConsultationRequest(query="头痛", session_id="sess-001", history=None)
    assert req_none.history is None

    req_omitted = ConsultationRequest(query="头痛", session_id="sess-001")
    assert req_omitted.history is None


def test_consultation_request_large_history_performance():
    """Test ConsultationRequest validates 10,000 history items within 500ms."""
    large_history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"问答历史 {i}"}
        for i in range(10000)
    ]
    t0 = time.perf_counter()
    req = ConsultationRequest(query="最新问题", session_id="sess-perf-01", history=large_history)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert len(req.history) == 10000
    assert elapsed_ms < 500


def test_consultation_request_invalid_history_rejected():
    """Test ConsultationRequest rejects non-list and invalid nested history values."""
    with pytest.raises(ValidationError):
        ConsultationRequest(query="头痛", session_id="sess-001", history="not_a_list")

    with pytest.raises(ValidationError):
        ConsultationRequest(query="头痛", session_id="sess-001", history=[{"role": ["nested_list"]}])


def test_consultation_response_needs_clarification():
    """Test needs_clarification field handles booleans, None, and coerces 0/1/strings."""
    assert ConsultationResponse(answer="ok", needs_clarification=True).needs_clarification is True
    assert ConsultationResponse(answer="ok", needs_clarification=False).needs_clarification is False
    assert ConsultationResponse(answer="ok", needs_clarification=None).needs_clarification is None

    coerced_1 = ConsultationResponse.model_validate({"answer": "ok", "needs_clarification": 1})
    assert coerced_1.needs_clarification is True

    coerced_0 = ConsultationResponse.model_validate({"answer": "ok", "needs_clarification": 0})
    assert coerced_0.needs_clarification is False

    coerced_str = ConsultationResponse.model_validate({"answer": "ok", "needs_clarification": "true"})
    assert coerced_str.needs_clarification is True

    with pytest.raises(ValidationError):
        ConsultationResponse.model_validate({"answer": "ok", "needs_clarification": "invalid_non_bool"})

    with pytest.raises(ValidationError):
        ConsultationResponse.model_validate({"answer": "ok", "needs_clarification": [1, 2]})


def test_consultation_request_session_id_bounds():
    """Test session_id boundary conditions (length 6..64, regex pattern)."""
    # Too short (<6)
    with pytest.raises(ValidationError):
        ConsultationRequest(query="头痛", session_id="12345")

    # Too long (>64)
    with pytest.raises(ValidationError):
        ConsultationRequest(query="头痛", session_id="a" * 65)

    # Invalid characters
    with pytest.raises(ValidationError):
        ConsultationRequest(query="头痛", session_id="sess id with spaces")

    with pytest.raises(ValidationError):
        ConsultationRequest(query="头痛", session_id="sess@id!")

    # Boundary valid
    req_6 = ConsultationRequest(query="头痛", session_id="a" * 6)
    assert len(req_6.session_id) == 6

    req_64 = ConsultationRequest(query="头痛", session_id="a" * 64)
    assert len(req_64.session_id) == 64


def test_consultation_response_roundtrip():
    """Test full serialization and deserialization roundtrip for ConsultationResponse."""
    resp = ConsultationResponse(
        answer="建议前往门诊就诊。",
        intent="appointment",
        symptoms=["头痛"],
        severity="medium",
        departments=["神经内科"],
        medications=[MedicationItem(name="对乙酰氨基酚", purpose="止痛")],
        advice=["避免劳累"],
        disclaimers=["仅供参考"],
        warnings=[],
        linked_entities=[LinkedEntity(input_entity="头疼", matched_entity="头痛", similarity=0.99)],
        duration_ms=45,
        session_id="sess-roundtrip-01",
        needs_clarification=False,
        retrieved_contexts=["头痛病理特征说明"],
    )
    raw_json = resp.model_dump_json()
    loaded = ConsultationResponse.model_validate_json(raw_json)
    assert loaded.answer == resp.answer
    assert loaded.medications[0].name == "对乙酰氨基酚"
    assert loaded.linked_entities[0].similarity == 0.99
    assert loaded.retrieved_contexts == ["头痛病理特征说明"]


# =====================================================================
# Target 3: Contract Checker Multi-Encoding Stress Tests
# =====================================================================


@pytest.mark.parametrize(
    "env_vars",
    [
        {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
        {"PYTHONIOENCODING": "gbk", "PYTHONUTF8": "0"},
        {"PYTHONIOENCODING": "cp936", "PYTHONUTF8": "0"},
        {"PYTHONIOENCODING": "ascii", "PYTHONUTF8": "0"},
        {"PYTHONIOENCODING": "cp1252", "PYTHONUTF8": "0"},
        {"PYTHONIOENCODING": "latin-1", "PYTHONUTF8": "0"},
    ],
)
def test_contract_checker_encodings(env_vars):
    """Test check_contracts.py runs and succeeds without UnicodeEncodeError across varied encodings."""
    checker = PROJECT_ROOT / "scripts" / "check_contracts.py"
    env = os.environ.copy()
    env.update(env_vars)
    proc = subprocess.run(
        [sys.executable, str(checker)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.returncode == 0
    assert b"\xe5\x85\xa8\xe7\xbb\xbf" in proc.stdout  # "全绿" in UTF-8
