"""Adversarial stress testing suite for Milestone 2 (src/api/routes.py).

Empirical scenarios:
1. Concurrency limit testing:
   - Simulate multiple concurrent requests to /api/consult and /api/consult/stream beyond
     settings.max_concurrent_requests (10 simultaneous requests against semaphore=3).
   - Strict throttling verification: peak concurrent active executions strictly <= 3.
   - Mixed sync + SSE streaming workload throttling.
   - Zero permit leakage after concurrent bursts.
2. TimeoutBudget testing:
   - Remaining budget reaches 0 -> TimeoutBudget.check() raises TimeoutError and immediately aborts.
   - Monotonic clock accuracy, zero floor, cumulative multi-step budget exhaustion.
   - Timeout handling in sync /api/consult (HTTP 504).
   - Timeout handling in SSE /api/consult/stream (event: error -> event: done [DONE]).
3. Client disconnect testing:
   - SafeStreamingResponse aclose() execution on client disconnect.
   - Immediate semaphore release upon client disconnection (no lingering permits).
   - Unblocking waiting queue immediately after a client drops.
   - request.is_disconnected() early break and downstream LangGraph generator close.
"""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Request

from config.settings import settings
from src.api.models import ConsultationRequest, ConsultationResponse
from src.api.routes import (
    SafeStreamingResponse,
    TimeoutBudget,
    _answer_cache_key,
    _deserialize_cache_payload,
    _serialize_cache_payload,
    consult,
    consult_stream,
    init_router,
)


# =====================================================================
# Fixtures & Helpers
# =====================================================================

@pytest.fixture(autouse=True)
def reset_routes_state():
    """Ensure clean router state for every test."""
    dummy_state = {
        "graph_app": None,
        "node_labels": {
            "intent_classifier": "意图识别",
            "symptom_detector": "症状提取",
            "medical_knowledge": "知识检索",
        },
    }
    sem = asyncio.Semaphore(settings.max_concurrent_requests)
    init_router(dummy_state, sem)
    yield
    init_router({}, asyncio.Semaphore(settings.max_concurrent_requests))


def create_mock_graph_app(delay_seconds: float = 0.05):
    """Create a mock graph_app for invoke and stream."""
    mock = MagicMock()

    def fake_invoke(input_data):
        time.sleep(delay_seconds)
        return {
            "final_answer": "测试回答：请多喝水并注意休息",
            "intent": "qa",
            "symptoms": ["头痛"],
            "severity": "mild",
            "departments": ["全科"],
            "medications": [],
            "advice": ["休息"],
            "disclaimers": ["仅供参考"],
            "warnings": [],
            "linked_entities": [],
            "trace": [{"node": "qa", "result": "ok"}],
        }

    mock.invoke.side_effect = fake_invoke

    def fake_stream(input_data, stream_mode):
        time.sleep(delay_seconds)
        yield ("updates", {"intent_classifier": {"intent": "qa"}})
        time.sleep(delay_seconds)
        yield ("updates", {"symptom_detector": {"symptoms": ["头痛"]}})
        time.sleep(delay_seconds)
        yield (
            "values",
            {
                "final_answer": "流式测试回答：请注意休息",
                "intent": "qa",
                "symptoms": ["头痛"],
                "severity": "mild",
                "departments": ["全科"],
                "medications": [],
                "advice": ["多休息"],
                "disclaimers": ["仅供参考"],
                "warnings": [],
                "linked_entities": [],
                "trace": [{"node": "stream_qa", "result": "ok"}],
            },
        )

    mock.stream.side_effect = fake_stream
    return mock


async def consume_safe_stream(response: SafeStreamingResponse) -> list[str]:
    """Helper to consume chunks from a SafeStreamingResponse directly from body_iterator."""
    chunks = []
    async for chunk in response.body_iterator:
        if isinstance(chunk, bytes):
            chunks.append(chunk.decode("utf-8"))
        elif isinstance(chunk, str):
            chunks.append(chunk)
    return chunks


# =====================================================================
# Scenario 1: Concurrency Limit Testing (10 simultaneous vs semaphore=3)
# =====================================================================

def test_concurrency_sync_requests_throttled_to_semaphore():
    """Verify 10 concurrent /api/consult requests strictly respect semaphore=3."""
    async def _run():
        concurrency_limit = 3
        sem = asyncio.Semaphore(concurrency_limit)
        mock_app = create_mock_graph_app(delay_seconds=0.08)

        state = {"graph_app": mock_app, "node_labels": {}}
        init_router(state, sem)

        active_executions = 0
        max_active_observed = 0

        orig_invoke = mock_app.invoke.side_effect

        def instrumented_invoke(input_data):
            nonlocal active_executions, max_active_observed
            active_executions += 1
            if active_executions > max_active_observed:
                max_active_observed = active_executions
            try:
                return orig_invoke(input_data)
            finally:
                active_executions -= 1

        mock_app.invoke.side_effect = instrumented_invoke

        with patch("src.history.store.load_context", return_value=[]), \
             patch("src.history.store.cache_get", return_value=None), \
             patch("src.history.store.cache_set", return_value=True), \
             patch("src.history.store.save_turn", return_value=True):

            tasks = [
                consult(
                    ConsultationRequest(
                        session_id=f"sess-sync-{i:03d}",
                        query=f"问诊问题 {i}",
                    )
                )
                for i in range(10)
            ]

            responses = await asyncio.gather(*tasks)

        # 1. Peak active executions must never exceed the semaphore limit
        assert max_active_observed <= concurrency_limit, (
            f"Active executions peaked at {max_active_observed}, exceeding limit {concurrency_limit}!"
        )
        assert max_active_observed == concurrency_limit, (
            f"Expected max concurrent executions to reach {concurrency_limit}, got {max_active_observed}"
        )

        # 2. All 10 requests must succeed
        assert len(responses) == 10
        for i, resp in enumerate(responses):
            assert resp.session_id == f"sess-sync-{i:03d}"
            assert "请多喝水" in resp.answer

        # 3. Semaphore must be completely free afterwards (no leaked permits)
        assert sem._value == concurrency_limit

    asyncio.run(_run())


def test_concurrency_stream_requests_throttled_to_semaphore():
    """Verify 10 concurrent /api/consult/stream requests strictly respect semaphore=3."""
    async def _run():
        concurrency_limit = 3
        sem = asyncio.Semaphore(concurrency_limit)
        mock_app = create_mock_graph_app(delay_seconds=0.05)

        state = {"graph_app": mock_app, "node_labels": {}}
        init_router(state, sem)

        active_executions = 0
        max_active_observed = 0

        orig_stream = mock_app.stream.side_effect

        def instrumented_stream(input_data, stream_mode):
            nonlocal active_executions, max_active_observed
            active_executions += 1
            if active_executions > max_active_observed:
                max_active_observed = active_executions
            try:
                for item in orig_stream(input_data, stream_mode):
                    yield item
            finally:
                active_executions -= 1

        mock_app.stream.side_effect = instrumented_stream

        with patch("src.history.store.load_context", return_value=[]), \
             patch("src.history.store.cache_get", return_value=None), \
             patch("src.history.store.cache_set", return_value=True), \
             patch("src.history.store.save_turn", return_value=True):

            async def run_single_stream(idx: int):
                req_body = ConsultationRequest(
                    session_id=f"sess-stream-{idx:03d}",
                    query=f"流式提问 {idx}",
                )
                mock_req = MagicMock(spec=Request)
                mock_req.is_disconnected = AsyncMock(return_value=False)

                resp = await consult_stream(mock_req, req_body)
                chunks = await consume_safe_stream(resp)
                full_text = "".join(chunks)
                return full_text

            tasks = [run_single_stream(i) for i in range(10)]
            results = await asyncio.gather(*tasks)

        # 1. Peak active executions must never exceed the semaphore limit
        assert max_active_observed <= concurrency_limit, (
            f"Streaming active executions peaked at {max_active_observed}, exceeding limit {concurrency_limit}!"
        )
        assert max_active_observed == concurrency_limit

        # 2. All 10 requests completed with answer and [DONE]
        assert len(results) == 10
        for res in results:
            assert "event: answer" in res
            assert "event: done" in res
            assert "[DONE]" in res

        # 3. Semaphore restored cleanly
        assert sem._value == concurrency_limit

    asyncio.run(_run())


def test_concurrency_mixed_sync_and_stream_requests():
    """Verify mixed 5 sync and 5 streaming requests throttled together against semaphore=3."""
    async def _run():
        concurrency_limit = 3
        sem = asyncio.Semaphore(concurrency_limit)
        mock_app = create_mock_graph_app(delay_seconds=0.06)

        state = {"graph_app": mock_app, "node_labels": {}}
        init_router(state, sem)

        active_executions = 0
        max_active_observed = 0

        orig_invoke = mock_app.invoke.side_effect
        orig_stream = mock_app.stream.side_effect

        def instrumented_invoke(input_data):
            nonlocal active_executions, max_active_observed
            active_executions += 1
            if active_executions > max_active_observed:
                max_active_observed = active_executions
            try:
                return orig_invoke(input_data)
            finally:
                active_executions -= 1

        def instrumented_stream(input_data, stream_mode):
            nonlocal active_executions, max_active_observed
            active_executions += 1
            if active_executions > max_active_observed:
                max_active_observed = active_executions
            try:
                for item in orig_stream(input_data, stream_mode):
                    yield item
            finally:
                active_executions -= 1

        mock_app.invoke.side_effect = instrumented_invoke
        mock_app.stream.side_effect = instrumented_stream

        with patch("src.history.store.load_context", return_value=[]), \
             patch("src.history.store.cache_get", return_value=None), \
             patch("src.history.store.cache_set", return_value=True), \
             patch("src.history.store.save_turn", return_value=True):

            async def run_sync(i):
                return await consult(
                    ConsultationRequest(session_id=f"sess-mix-s-{i}", query=f"同步问题 {i}")
                )

            async def run_stream(i):
                mock_req = MagicMock(spec=Request)
                mock_req.is_disconnected = AsyncMock(return_value=False)
                req_body = ConsultationRequest(session_id=f"sess-mix-st-{i}", query=f"流式问题 {i}")
                resp = await consult_stream(mock_req, req_body)
                return await consume_safe_stream(resp)

            tasks = []
            for i in range(5):
                tasks.append(run_sync(i))
                tasks.append(run_stream(i))

            results = await asyncio.gather(*tasks)

        assert max_active_observed <= concurrency_limit
        assert max_active_observed == concurrency_limit
        assert len(results) == 10
        assert sem._value == concurrency_limit

    asyncio.run(_run())


# =====================================================================
# Scenario 2: TimeoutBudget Testing
# =====================================================================

def test_timeout_budget_unit_behavior():
    """Adversarial unit test of TimeoutBudget monotonic clock and budget checking."""
    # 1. Budget of 0.1s
    budget = TimeoutBudget(0.1)
    assert 0.0 < budget.remaining <= 0.1
    rem = budget.check()
    assert 0.0 < rem <= 0.1

    # Sleep past deadline
    time.sleep(0.12)
    assert budget.remaining == 0.0

    # check() must raise asyncio.TimeoutError
    with pytest.raises(asyncio.TimeoutError, match="预算已耗尽"):
        budget.check()

    # Zero or negative budget
    zero_budget = TimeoutBudget(0.0)
    assert zero_budget.remaining == 0.0
    with pytest.raises(asyncio.TimeoutError):
        zero_budget.check()

    neg_budget = TimeoutBudget(-5.0)
    assert neg_budget.remaining == 0.0
    with pytest.raises(asyncio.TimeoutError):
        neg_budget.check()


def test_timeout_budget_aborts_sync_consult_with_504():
    """Verify TimeoutBudget raises TimeoutError and translates to 504 in /api/consult."""
    async def _run():
        sem = asyncio.Semaphore(1)
        mock_app = MagicMock()

        # Simulate invoke taking 0.3s
        def slow_invoke(input_data):
            time.sleep(0.3)
            return {"final_answer": "slow"}

        mock_app.invoke.side_effect = slow_invoke
        state = {"graph_app": mock_app, "node_labels": {}}
        init_router(state, sem)

        with patch("src.history.store.load_context", return_value=[]), \
             patch("src.history.store.cache_get", return_value=None), \
             patch.object(settings, "api_timeout_seconds", 0.1):

            t0 = time.monotonic()
            with pytest.raises(HTTPException) as exc_info:
                await consult(
                    ConsultationRequest(session_id="sess-timeout-sync", query="超长查询")
                )
            elapsed = time.monotonic() - t0

            assert exc_info.value.status_code == 504
            assert "处理超时" in exc_info.value.detail
            assert elapsed < 0.25

        # Semaphore permit must be released immediately after 504
        assert sem._value == 1

    asyncio.run(_run())


def test_timeout_budget_aborts_stream_consult_cleanly():
    """Verify TimeoutBudget raises TimeoutError and yields error event in /api/consult/stream."""
    async def _run():
        sem = asyncio.Semaphore(1)
        mock_app = MagicMock()

        def slow_stream(input_data, stream_mode):
            yield ("updates", {"intent_classifier": {"intent": "qa"}})
            time.sleep(0.25)
            yield ("updates", {"symptom_detector": {"symptoms": ["头痛"]}})

        mock_app.stream.side_effect = slow_stream
        state = {"graph_app": mock_app, "node_labels": {}}
        init_router(state, sem)

        with patch("src.history.store.load_context", return_value=[]), \
             patch("src.history.store.cache_get", return_value=None), \
             patch.object(settings, "api_timeout_seconds", 0.15):

            mock_req = MagicMock(spec=Request)
            mock_req.is_disconnected = AsyncMock(return_value=False)
            req_body = ConsultationRequest(session_id="sess-timeout-stream", query="流式超时测试")

            resp = await consult_stream(mock_req, req_body)
            chunks = await consume_safe_stream(resp)
            full_text = "".join(chunks)

            assert "event: progress" in full_text
            assert "event: error" in full_text
            assert "处理超时" in full_text
            assert "event: done" in full_text
            assert "[DONE]" in full_text

        assert sem._value == 1

    asyncio.run(_run())


def test_timeout_budget_multi_step_cumulative_exhaustion():
    """Verify multi-step cumulative time triggers TimeoutBudget.check() without resetting."""
    async def _run():
        sem = asyncio.Semaphore(1)
        mock_app = MagicMock()

        # Step 1 takes 0.08s, Step 2 takes 0.08s (total 0.16s > budget 0.12s)
        def multi_step_stream(input_data, stream_mode):
            time.sleep(0.08)
            yield ("updates", {"intent_classifier": {"intent": "qa"}})
            time.sleep(0.08)
            yield ("updates", {"symptom_detector": {"symptoms": ["头晕"]}})
            yield ("values", {"final_answer": "ok"})

        mock_app.stream.side_effect = multi_step_stream
        state = {"graph_app": mock_app, "node_labels": {}}
        init_router(state, sem)

        with patch("src.history.store.load_context", return_value=[]), \
             patch("src.history.store.cache_get", return_value=None), \
             patch.object(settings, "api_timeout_seconds", 0.12):

            mock_req = MagicMock(spec=Request)
            mock_req.is_disconnected = AsyncMock(return_value=False)
            req_body = ConsultationRequest(session_id="sess-cum-timeout", query="累积超时测试")

            resp = await consult_stream(mock_req, req_body)
            chunks = await consume_safe_stream(resp)
            full_text = "".join(chunks)

            assert "event: error" in full_text
            assert "处理超时" in full_text
            assert "event: answer" not in full_text

        assert sem._value == 1

    asyncio.run(_run())


# =====================================================================
# Scenario 3: Client Disconnect Testing
# =====================================================================

def test_client_disconnect_immediately_releases_semaphore_and_aclose():
    """Simulate client dropping connection after 1st chunk: verify aclose() and semaphore release."""
    async def _run():
        sem = asyncio.Semaphore(1)
        mock_app = MagicMock()

        langgraph_gen_closed = False

        class ClosableLanggraphGen:
            def __init__(self):
                self.step = 0

            def __iter__(self):
                return self

            def __next__(self):
                self.step += 1
                if self.step == 1:
                    return ("updates", {"intent_classifier": {"intent": "qa"}})
                time.sleep(0.2)
                return ("updates", {"symptom_detector": {"symptoms": ["咳嗽"]}})

            def close(self):
                nonlocal langgraph_gen_closed
                langgraph_gen_closed = True

        closable_gen = ClosableLanggraphGen()
        mock_app.stream.return_value = closable_gen

        state = {"graph_app": mock_app, "node_labels": {}}
        init_router(state, sem)

        with patch("src.history.store.load_context", return_value=[]), \
             patch("src.history.store.cache_get", return_value=None), \
             patch("src.history.store.cache_set", return_value=True), \
             patch("src.history.store.save_turn", return_value=True):

            mock_req = MagicMock(spec=Request)
            disconnect_calls = 0

            async def fake_is_disconnected():
                nonlocal disconnect_calls
                disconnect_calls += 1
                return disconnect_calls >= 2

            mock_req.is_disconnected = fake_is_disconnected
            req_body = ConsultationRequest(session_id="sess-disc-01", query="客户端掉线测试")

            resp = await consult_stream(mock_req, req_body)

            scope = {"type": "http", "method": "POST", "path": "/api/consult/stream"}
            received_chunks = []

            disconnect_event = asyncio.Event()

            async def fake_receive():
                await disconnect_event.wait()
                return {"type": "http.disconnect"}

            async def fake_send(message):
                if message["type"] == "http.response.body":
                    body_bytes = message.get("body", b"")
                    received_chunks.append(body_bytes)
                    if b"intent_classifier" in body_bytes:
                        disconnect_event.set()

            await resp(scope, fake_receive, fake_send)

        assert langgraph_gen_closed is True
        assert sem._value == 1

        acquired_next = False
        async with sem:
            acquired_next = True
        assert acquired_next is True

    asyncio.run(_run())


def test_client_disconnect_during_stream_send_exception():
    """Simulate client abrupt connection drop (ClientDisconnect exception in ASGI send)."""
    async def _run():
        sem = asyncio.Semaphore(1)
        mock_app = create_mock_graph_app(delay_seconds=0.04)

        state = {"graph_app": mock_app, "node_labels": {}}
        init_router(state, sem)

        with patch("src.history.store.load_context", return_value=[]), \
             patch("src.history.store.cache_get", return_value=None):

            mock_req = MagicMock(spec=Request)
            mock_req.is_disconnected = AsyncMock(return_value=False)
            req_body = ConsultationRequest(session_id="sess-abrupt-drop", query="连接突发中断")

            resp = await consult_stream(mock_req, req_body)

            scope = {"type": "http", "method": "POST", "path": "/api/consult/stream"}

            disconnect_event = asyncio.Event()

            async def fake_receive():
                await disconnect_event.wait()
                return {"type": "http.disconnect"}

            send_count = 0

            async def fake_send_raising(message):
                nonlocal send_count
                send_count += 1
                if send_count >= 2:
                    raise ConnectionResetError("Client disconnected prematurely")

            with pytest.raises(ConnectionResetError):
                await resp(scope, fake_receive, fake_send_raising)

        assert sem._value == 1
        assert sem.locked() is False
        async with sem:
            pass

    asyncio.run(_run())


def test_unblock_waiting_request_when_streaming_client_disconnects():
    """Verify that a waiting queued request gets the semaphore as soon as an active stream disconnects."""
    async def _run():
        concurrency_limit = 1
        sem = asyncio.Semaphore(concurrency_limit)
        mock_app = MagicMock()

        class HangingGenerator:
            def __iter__(self):
                return self

            def __next__(self):
                time.sleep(0.02)
                return ("updates", {"intent_classifier": {"intent": "qa"}})

            def close(self):
                pass

        mock_app.stream.return_value = HangingGenerator()

        def quick_invoke(input_data):
            return {"final_answer": "立即响应", "intent": "qa"}

        mock_app.invoke.side_effect = quick_invoke

        state = {"graph_app": mock_app, "node_labels": {}}
        init_router(state, sem)

        with patch("src.history.store.load_context", return_value=[]), \
             patch("src.history.store.cache_get", return_value=None), \
             patch("src.history.store.cache_set", return_value=True), \
             patch("src.history.store.save_turn", return_value=True):

            mock_req = MagicMock(spec=Request)
            mock_req.is_disconnected = AsyncMock(return_value=False)
            req_body1 = ConsultationRequest(session_id="sess-hang-1", query="挂起请求")

            resp1 = await consult_stream(mock_req, req_body1)

            req_body2 = ConsultationRequest(session_id="sess-queue-2", query="排队请求")

            req2_finished = False

            async def do_req2():
                nonlocal req2_finished
                r = await consult(req_body2)
                req2_finished = True
                return r

            scope = {"type": "http", "method": "POST", "path": "/api/consult/stream"}

            disconnect_event = asyncio.Event()

            async def fake_receive():
                await disconnect_event.wait()
                return {"type": "http.disconnect"}

            stream_started = asyncio.Event()

            async def fake_send(message):
                if message["type"] == "http.response.body":
                    stream_started.set()
                    raise ConnectionAbortedError("Client dropped connection")

            req1_task = asyncio.create_task(resp1(scope, fake_receive, fake_send))

            await stream_started.wait()

            req2_task = asyncio.create_task(do_req2())

            with pytest.raises(ConnectionAbortedError):
                await req1_task

            resp2_val = await asyncio.wait_for(req2_task, timeout=1.0)
            assert req2_finished is True
            assert resp2_val.session_id == "sess-queue-2"
            assert resp2_val.answer == "立即响应"

        assert sem._value == 1

    asyncio.run(_run())


# =====================================================================
# Additional Adversarial Checks: Exact Cache Key & Cross-Session Isolation
# =====================================================================

def test_adversarial_cache_collision_resistance():
    """Verify exact cache key cannot be collided even when recent turns match."""
    h1 = [
        {"role": "user", "content": "患者确诊2型糖尿病五年"},
        {"role": "assistant", "content": "收到，糖尿病史"},
        {"role": "user", "content": "最近头晕目眩"},
        {"role": "assistant", "content": "建议测量血压血糖"},
        {"role": "user", "content": "血压正常"},
    ]
    h2 = [
        {"role": "user", "content": "患者左腿骨折术后两周"},
        {"role": "assistant", "content": "收到，骨折术后"},
        {"role": "user", "content": "最近头晕目眩"},
        {"role": "assistant", "content": "建议测量血压血糖"},
        {"role": "user", "content": "血压正常"},
    ]

    key1 = _answer_cache_key("头晕该挂哪个科室", h1)
    key2 = _answer_cache_key("头晕该挂哪个科室", h2)

    assert key1 != key2, "Cache key collision detected between diabetes and fracture histories!"


def test_adversarial_session_leak_in_cached_payload():
    """Verify serialized cache payload never leaks original session_id across users."""
    original_resp = ConsultationResponse(
        answer="口服二甲双胍治疗",
        intent="medication",
        symptoms=["多饮", "多尿"],
        session_id="secret-patient-session-uuid-9999",
        duration_ms=450,
    )

    serialized = _serialize_cache_payload(original_resp)
    assert "secret-patient-session-uuid-9999" not in serialized
    assert "duration_ms" not in serialized

    new_patient_session = "patient-carol-uuid-1111"
    restored = _deserialize_cache_payload(serialized, new_patient_session, duration_ms=12)

    assert restored.session_id == "patient-carol-uuid-1111"
    assert restored.duration_ms == 12
    assert restored.answer == "口服二甲双胍治疗"
