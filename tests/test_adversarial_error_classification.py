"""Adversarial stress tests for error classification and exception handling in evals/runner.py.

Stress Scenarios covered:
1. Custom exception classes inheriting from Exception with and without string representations.
2. Complex nested exceptions (e.__cause__, e.__context__), exceptions with Unicode/emojis in error message.
3. Combinations of status codes, keywords, and OOM messages.
4. End-to-end schema integrity, aggregation, report generation under adversarial exception cases.
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evals.runner import (
    UNIFIED_SAMPLE_SCHEMA_FIELDS,
    _build_error_sample,
    _classify_error,
    aggregate,
    evaluate_case,
    gates,
    write_report,
)


# =====================================================================
# Scenario 1: Custom Exception Classes (With and Without String Representations)
# =====================================================================


class PlainCustomException(Exception):
    """Custom exception with standard string argument."""
    pass


class EmptyCustomException(Exception):
    """Custom exception with no arguments / empty string representation."""
    pass


class ReprOnlyException(Exception):
    """Custom exception where __str__ returns empty string but __repr__ has details."""

    def __str__(self):
        return ""

    def __repr__(self):
        return "ReprOnlyException(detail='CUDA OOM on GPU:0')"


class SubclassedTimeoutException(TimeoutError):
    """Custom exception subclassing TimeoutError without explicit message."""
    pass


class SubclassedJSONDecodeException(json.JSONDecodeError):
    """Custom exception subclassing JSONDecodeError."""

    def __init__(self):
        super().__init__("Invalid token at pos 0", "doc", 0)


class StructuredTupleArgException(Exception):
    """Custom exception storing non-string tuple/dict in args."""
    pass


def test_custom_exception_with_string_representation():
    """1.1 Custom exception with keyword string representations."""
    assert _classify_error(PlainCustomException("Request timeout after 30s")) == "timeout"
    assert _classify_error(PlainCustomException("HTTP 502 Bad Gateway")) == "http_error"
    assert _classify_error(PlainCustomException("OpenAI API rate limit exceeded")) == "model_error"
    assert _classify_error(PlainCustomException("GPU CUDA out of memory")) == "model_error"
    assert _classify_error(PlainCustomException("empty answer received")) == "empty_response"
    assert _classify_error(PlainCustomException("json parse validation failed")) == "parse_error"


def test_custom_exception_without_string_representation():
    """1.2 Custom exception with no arguments (empty string) defaults gracefully to unknown."""
    exc = EmptyCustomException()
    assert str(exc) == ""
    assert _classify_error(exc) == "unknown"

    sample = _build_error_sample({"id": "adv_empty", "query": "测试"}, exc, 15)
    assert sample["status"] == "error"
    assert sample["error_type"] == "unknown"
    assert sample["error"] == ""
    assert UNIFIED_SAMPLE_SCHEMA_FIELDS.issubset(sample.keys())


def test_custom_exception_repr_only():
    """1.3 Custom exception where __str__ is empty is safely handled without crash."""
    exc = ReprOnlyException()
    assert str(exc) == ""
    assert _classify_error(exc) == "unknown"

    sample = _build_error_sample({"id": "adv_repr", "query": "测试"}, exc, 20)
    assert sample["error_type"] == "unknown"
    assert UNIFIED_SAMPLE_SCHEMA_FIELDS.issubset(sample.keys())


def test_custom_exception_subclassing_builtin_errors():
    """1.4 Custom exceptions subclassing TimeoutError and JSONDecodeError inherit type checks."""
    assert _classify_error(SubclassedTimeoutException()) == "timeout"
    assert _classify_error(SubclassedJSONDecodeException()) == "parse_error"


def test_custom_exception_with_structured_args():
    """1.5 Custom exception with structured tuple args (e.g. status code, payload dict)."""
    exc = StructuredTupleArgException(500, {"error": "internal crash", "gpu": True})
    # str(exc) becomes "(500, {'error': 'internal crash', 'gpu': True})"
    # Contains 500 and gpu -> classified into http_error or model_error safely
    classified = _classify_error(exc)
    assert classified in ("http_error", "model_error")

    sample = _build_error_sample({"id": "adv_struct"}, exc, 50)
    assert sample["status"] == "error"
    assert UNIFIED_SAMPLE_SCHEMA_FIELDS.issubset(sample.keys())


# =====================================================================
# Scenario 2: Complex Nested Exceptions & Unicode/Emojis Handling
# =====================================================================


def test_nested_exceptions_explicit_cause():
    """2.1 Exceptions chained explicitly via 'raise ... from cause'."""
    try:
        try:
            raise TimeoutError("connection to LLM service timed out")
        except Exception as inner:
            raise RuntimeError("High-level pipeline execution failed") from inner
    except Exception as outer:
        # Outer message doesn't match keywords, so falls to unknown
        assert _classify_error(outer) == "unknown"
        # Inner cause is identifiable
        assert _classify_error(outer.__cause__) == "timeout"

        sample = _build_error_sample({"id": "nested_explicit"}, outer, 300)
        assert sample["status"] == "error"
        assert UNIFIED_SAMPLE_SCHEMA_FIELDS.issubset(sample.keys())


def test_nested_exceptions_implicit_context():
    """2.2 Exceptions chained implicitly via 'except ...: raise'."""
    try:
        try:
            raise ConnectionResetError("HTTP 502 connection reset by peer")
        except Exception:
            raise RuntimeError("Task execution failed")
    except Exception as outer:
        assert _classify_error(outer) == "unknown"
        assert _classify_error(outer.__context__) == "http_error"


def test_deeply_nested_exception_chain():
    """2.3 Deeply nested multi-level exception chain (A -> B -> C)."""
    try:
        try:
            try:
                raise TimeoutError("socket timed out")
            except Exception as e1:
                raise ValueError("JSON decode error during parse") from e1
        except Exception as e2:
            raise RuntimeError("Workflow node failed") from e2
    except Exception as e3:
        assert _classify_error(e3) == "unknown"
        assert _classify_error(e3.__cause__) == "parse_error"
        assert _classify_error(e3.__cause__.__cause__) == "timeout"


@pytest.mark.parametrize(
    ("err_msg", "expected_type"),
    [
        ("【超时警报】上游模型服务响应耗时过长，已中断 ⏳", "timeout"),
        ("🚨 HTTP 500 服务器内部故障 💥 [Pool exhausted]", "http_error"),
        ("🌐 502 Bad Gateway 遇到网关异常 ❌", "http_error"),
        ("🛑 OpenAI API Key 鉴权失败，额度耗尽 (rate limit)", "model_error"),
        ("🔥 CUDA out of memory on GPU:0 💥 显存溢出", "model_error"),
        ("⚠️ empty response: 无有效回答返回 🚫", "empty_response"),
        ("📦 JSON 数据格式校验失败 parse error 格式不符", "parse_error"),
        ("عطل في الخادم 500 خطأ داخلي", "http_error"),  # Arabic RTL + 500
        ("Соединение прервано: read timeout ⌛", "timeout"),  # Cyrillic + timeout
        ("🌟 未知异常: 系统完全正常但抛出了未知错误 🤖", "unknown"),
    ],
)
def test_unicode_and_emojis_in_error_message(err_msg, expected_type):
    """2.4 Messages with CJK, Cyrillic, Arabic RTL, and emojis are classified and serialized cleanly."""
    exc = RuntimeError(err_msg)
    classified = _classify_error(exc)
    assert classified == expected_type

    sample = _build_error_sample({"id": "emoji_case", "query": "测试Unicode"}, exc, 120)
    assert sample["error_type"] == expected_type
    assert sample["error"] == err_msg
    assert UNIFIED_SAMPLE_SCHEMA_FIELDS.issubset(sample.keys())


def test_unicode_and_emojis_report_generation(tmp_path):
    """2.5 Verify write_report and aggregate survive and format messages with emojis and CJK."""
    cases = [
        RuntimeError("🚨 500 Internal Crash 💥"),
        TimeoutError("⏳ 接口响应超时 timeout 5000ms"),
        ValueError("⚠️ empty response returned ❌"),
        RuntimeError("🔥 GPU CUDA OOM 显存耗尽"),
    ]
    results = [
        _build_error_sample({"id": f"emoji_{i}", "query": f"查询_{i}", "gold": {"intent": "general"}}, exc, 50 * i)
        for i, exc in enumerate(cases, start=1)
    ]
    agg = aggregate(results)
    assert agg["count"] == 4
    assert agg["failed_count"] == 4
    assert set(agg["errors_by_type"].keys()) == {"http_error", "timeout", "empty_response", "model_error"}

    with patch("evals.runner.EVALS_DIR", tmp_path):
        report_path = write_report(agg, results, "core")
        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert "🚨 500 Internal Crash" in content
        assert "⏳ 接口响应超时" in content
        assert (tmp_path / "last_results.json").exists()


# =====================================================================
# Scenario 3: Combinations of Status Codes, Keywords, and OOM Messages
# =====================================================================


@pytest.mark.parametrize(
    ("msg", "expected_type", "rationale"),
    [
        # Timeout priority over HTTP status codes (e.g. 504 Gateway Timeout)
        ("504 Gateway Timeout", "timeout", "timeout rule evaluated before status code"),
        ("HTTP 500: request timed out waiting for worker", "timeout", "timeout keyword takes precedence"),
        ("Connection timeout with status_code=502", "timeout", "timeout keyword takes precedence"),

        # HTTP error priority over model error when status code is present
        ("HTTP 500 Server Error: GPU out of memory during inference", "http_error", "500/http matched first in HTTP responses"),
        ("503 Service Unavailable: OpenAI model rate limit exceeded", "http_error", "503 matched before rate/model"),
        ("404 Not Found: model endpoint missing", "http_error", "404 matched before model"),
        ("status_code: 502, error: memory allocation failed", "http_error", "status_code/502 matched before memory"),

        # Pure model errors without HTTP status markers
        ("Model inference crashed: CUDA out of memory on device 0", "model_error", "cuda/oom keywords"),
        ("OpenAI API authentication failed: invalid api key", "model_error", "openai/api key keywords"),
        ("Rate limit exceeded for organization", "model_error", "rate keyword"),
        ("PyTorch GPU OOM: Tried to allocate 2048MiB", "model_error", "gpu/oom keywords"),

        # Empty response combinations
        ("Service returned empty payload without error", "empty_response", "empty keyword"),
        ("SSE stream closed without returning an answer event", "empty_response", "without returning an answer keyword"),
        ("Backend completed execution with no answer", "empty_response", "no answer keyword"),

        # Parse error combinations
        ("JSONDecodeError: Unterminated string starting at line 1", "parse_error", "json/parse keyword"),
        ("Pydantic validation error: missing field 'answer'", "parse_error", "validation keyword"),
        ("Parse error while decoding server chunk", "parse_error", "parse keyword"),
    ],
)
def test_keyword_and_status_combinations(msg, expected_type, rationale):
    """3.1 Verify deterministic classification precedence across competing keywords."""
    classified = _classify_error(msg)
    assert classified == expected_type, f"Failed for '{msg}' (rationale: {rationale})"


# =====================================================================
# Scenario 4: Full Pipeline Integration under Adversarial Failure Scenarios
# =====================================================================


def test_evaluate_case_catches_adversarial_exceptions():
    """4.1 evaluate_case catches unexpected arbitrary exceptions in core runner mode."""
    case = {"id": "adv_crash", "query": "测试异常", "gold": {"intent": "general"}}

    with patch("src.agents.graph.run", side_effect=RuntimeError("🔥 Sudden CUDA Out of Memory crash")):
        res = evaluate_case(case, runner_mode="core")
        assert res["status"] == "error"
        assert res["error_type"] == "model_error"
        assert "CUDA Out of Memory" in res["error"]
        assert UNIFIED_SAMPLE_SCHEMA_FIELDS.issubset(res.keys())
        assert res["session_id"].startswith("eval-")
        assert res["latency_ms"] >= 0


def test_evaluate_case_catches_adversarial_empty_custom_exception():
    """4.2 evaluate_case catches custom empty exception safely without raising."""
    case = {"id": "adv_empty_exc", "query": "测试空异常"}

    with patch("src.agents.graph.run", side_effect=EmptyCustomException()):
        res = evaluate_case(case, runner_mode="core")
        assert res["status"] == "error"
        assert res["error_type"] == "unknown"
        assert UNIFIED_SAMPLE_SCHEMA_FIELDS.issubset(res.keys())


def test_substring_collisions_empirically_documented():
    """4.3 Document empirical behavior of substring matching rules:
    - 'generate' / 'generated' contains 'rate' -> classifies as 'model_error'
    - 'chatroom' contains 'oom' -> classifies as 'model_error'
    - 'sparse' contains 'parse' -> classifies as 'parse_error'
    """
    assert _classify_error("no answer generated") == "model_error"
    assert _classify_error("chatroom disconnected") == "model_error"
    assert _classify_error("sparse index computation error") == "parse_error"


def test_broken_str_exception_empirically_documented():
    """4.4 Document empirical behavior when an exception's __str__ method raises."""
    class BrokenStrException(Exception):
        def __str__(self):
            raise ValueError("custom __str__ failed")

    exc = BrokenStrException()
    with pytest.raises(ValueError, match="custom __str__ failed"):
        _classify_error(exc)

    with pytest.raises(ValueError, match="custom __str__ failed"):
        _build_error_sample({"id": "broken"}, exc, 10)


def test_nested_exception_cause_not_traversed_empirically_documented():
    """4.5 Document empirical behavior that outer generic exception masks inner __cause__."""
    try:
        try:
            raise TimeoutError("connection timed out")
        except Exception as inner:
            raise RuntimeError("Generic wrapper error without keywords") from inner
    except Exception as outer:
        assert _classify_error(outer) == "unknown"
        assert _classify_error(outer.__cause__) == "timeout"

