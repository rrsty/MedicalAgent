"""Adversarial stress tests for Milestone M2 (src/api/routes.py):
1. TimeoutBudget monotonic clock and end-to-end deadline enforcement
2. Concurrency semaphore lifecycle coverage in consult_stream
3. SafeStreamingResponse generator finalization (aclose) and client disconnect termination
4. Symmetric answer cache backfill between sync (/consult) and SSE (/consult/stream)
5. Exact context cache key generation and multi-session isolation
"""
import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Request

from src.api.models import ConsultationRequest, ConsultationResponse
from src.api.routes import (
    SafeStreamingResponse,
    TimeoutBudget,
    _answer_cache_key,
    _deserialize_cache_payload,
    _process_consult,
    _serialize_cache_payload,
    consult,
    consult_stream,
    init_router,
)


# ============================================================================
# 1. TimeoutBudget Monotonicity and Enforcement
# ============================================================================

def test_timeout_budget_monotonicity():
    """Verify TimeoutBudget uses monotonic clock and computes remaining timeout."""
    budget = TimeoutBudget(0.2)
    assert budget.timeout_seconds == 0.2
    assert budget.remaining <= 0.2
    assert budget.remaining > 0.0

    rem1 = budget.check()
    time.sleep(0.05)
    rem2 = budget.check()
    assert rem1 > rem2, "Remaining time must decrease monotonically"

    time.sleep(0.16)
    assert budget.remaining == 0.0
    with pytest.raises(asyncio.TimeoutError, match="全链路处理超时"):
        budget.check()


def test_timeout_budget_zero_or_negative():
    """Verify TimeoutBudget with zero or negative timeout raises immediately."""
    b0 = TimeoutBudget(0)
    assert b0.remaining == 0.0
    with pytest.raises(asyncio.TimeoutError):
        b0.check()

    b_neg = TimeoutBudget(-1.0)
    assert b_neg.remaining == 0.0
    with pytest.raises(asyncio.TimeoutError):
        b_neg.check()


def test_consult_timeout_in_context_load():
    """Verify consult raises 504 if context loading exceeds TimeoutBudget."""
    async def _run():
        sem = asyncio.Semaphore(5)
        mock_app = MagicMock()
        init_router({"graph_app": mock_app}, sem)

        def slow_load(sid):
            time.sleep(0.15)
            return []

        req = ConsultationRequest(query="头晕", session_id="s_timeout_ctx")
        with patch("src.api.routes.history_store.load_context", side_effect=slow_load), \
             patch("config.settings.settings.api_timeout_seconds", 0.05):
            with pytest.raises(HTTPException) as exc_info:
                await consult(req)
            assert exc_info.value.status_code == 504
            assert "处理超时" in exc_info.value.detail
    asyncio.run(_run())


def test_consult_timeout_in_graph_invoke():
    """Verify consult raises 504 if graph execution exceeds remaining TimeoutBudget."""
    async def _run():
        sem = asyncio.Semaphore(5)
        mock_app = MagicMock()

        def slow_invoke(state):
            time.sleep(0.15)
            return {"final_answer": "done", "intent": "general"}

        mock_app.invoke = slow_invoke
        init_router({"graph_app": mock_app}, sem)

        req = ConsultationRequest(query="头晕", session_id="s_timeout_graph")
        with patch("src.api.routes.history_store.load_context", return_value=[]), \
             patch("src.api.routes.history_store.cache_get", return_value=None), \
             patch("config.settings.settings.api_timeout_seconds", 0.08):
            with pytest.raises(HTTPException) as exc_info:
                await consult(req)
            assert exc_info.value.status_code == 504
    asyncio.run(_run())


def test_consult_stream_timeout_in_event_generator():
    """Verify consult_stream yields error event and closes when budget expires during chunks."""
    async def _run():
        sem = asyncio.Semaphore(5)
        mock_app = MagicMock()

        def slow_stream(state, stream_mode):
            yield ("updates", {"intent_classifier": {"intent": "qa"}})
            time.sleep(0.15)
            yield ("values", {"final_answer": "late"})

        mock_app.stream = slow_stream
        init_router({"graph_app": mock_app}, sem)

        req = ConsultationRequest(query="感冒发烧", session_id="s_stream_timeout")
        with patch("src.api.routes.history_store.load_context", return_value=[]), \
             patch("src.api.routes.history_store.cache_get", return_value=None), \
             patch("config.settings.settings.api_timeout_seconds", 0.08):
            response = await consult_stream(req)
            assert isinstance(response, SafeStreamingResponse)

            events = []
            async for chunk in response.body_iterator:
                events.append(chunk)

            full_output = "".join(events)
            assert "event: progress" in full_output
            assert "event: error" in full_output
            assert "处理超时" in full_output
            assert "event: done\ndata: [DONE]" in full_output
            assert "event: answer" not in full_output, "No answer should be yielded on timeout"
    asyncio.run(_run())


# ============================================================================
# 2. Concurrency Semaphore Lifecycle in Streaming
# ============================================================================

def test_concurrency_semaphore_covers_streaming_lifecycle():
    """Verify _semaphore limits active streaming generators and is released upon completion."""
    async def _run():
        sem = asyncio.Semaphore(2)
        mock_app = MagicMock()

        active_streams = 0
        max_active_streams = 0
        lock = asyncio.Lock()

        def concurrent_stream(state, stream_mode):
            nonlocal active_streams, max_active_streams
            # simulate step 1
            yield ("updates", {"intent_classifier": {"intent": "general"}})
            # simulate latency
            time.sleep(0.04)
            yield ("values", {"final_answer": f"ans-{state['query']}"})

        mock_app.stream = concurrent_stream
        init_router({"graph_app": mock_app}, sem)

        async def run_client(idx: int):
            nonlocal active_streams, max_active_streams
            req = ConsultationRequest(query=f"q_{idx}", session_id=f"session_client_{idx}")
            with patch("src.api.routes.history_store.load_context", return_value=[]), \
                 patch("src.api.routes.history_store.cache_get", return_value=None), \
                 patch("src.api.routes.history_store.cache_set"), \
                 patch("src.api.routes.history_store.save_turn"), \
                 patch("config.settings.settings.api_timeout_seconds", 10.0):
                resp = await consult_stream(req)
                async for chunk in resp.body_iterator:
                    async with lock:
                        active_streams += 1
                        if active_streams > max_active_streams:
                            max_active_streams = active_streams
                    await asyncio.sleep(0.02)
                    async with lock:
                        active_streams -= 1

        # Run 5 concurrent clients with semaphore limit = 2
        tasks = [asyncio.create_task(run_client(i)) for i in range(5)]
        await asyncio.gather(*tasks)

        assert max_active_streams <= 2, f"Expected concurrency <= 2, got {max_active_streams}"
        assert sem._value == 2, f"Semaphore permit leaked! Final value: {sem._value}"
    asyncio.run(_run())


def test_semaphore_release_on_stream_exception():
    """Verify _semaphore is released even if an unhandled exception occurs inside stream."""
    async def _run():
        sem = asyncio.Semaphore(1)
        mock_app = MagicMock()

        def error_stream(state, stream_mode):
            yield ("updates", {"intent_classifier": {"intent": "qa"}})
            raise RuntimeError("Unexpected LLM error")

        mock_app.stream = error_stream
        init_router({"graph_app": mock_app}, sem)

        req = ConsultationRequest(query="报错测试", session_id="sid_err")
        with patch("src.api.routes.history_store.load_context", return_value=[]), \
             patch("src.api.routes.history_store.cache_get", return_value=None), \
             patch("config.settings.settings.api_timeout_seconds", 5.0):
            resp = await consult_stream(req)
            events = []
            async for chunk in resp.body_iterator:
                events.append(chunk)

            full_output = "".join(events)
            assert "event: error" in full_output
            assert "Unexpected LLM error" in full_output
            assert sem._value == 1, "Semaphore permit leaked on generator error!"
    asyncio.run(_run())


# ============================================================================
# 3. Client Disconnect & SafeStreamingResponse Finalization
# ============================================================================

def test_safe_streaming_response_calls_aclose():
    """Verify SafeStreamingResponse calls body_iterator.aclose() in finally block."""
    async def _run():
        aclose_called = False

        async def fake_gen():
            nonlocal aclose_called
            try:
                yield "data: chunk 1\n\n"
                yield "data: chunk 2\n\n"
            finally:
                aclose_called = True

        response = SafeStreamingResponse(fake_gen(), media_type="text/event-stream")

        # Simulate ASGI call lifecycle that exits early (e.g. client disconnect)
        async def mock_receive():
            return {"type": "http.disconnect"}

        async def mock_send(message):
            if message.get("type") == "http.response.body" and message.get("body"):
                raise ConnectionResetError("Client dropped connection")

        scope = {"type": "http", "method": "POST", "path": "/api/consult/stream"}
        try:
            await response(scope, mock_receive, mock_send)
        except ConnectionResetError:
            pass

        assert aclose_called, "SafeStreamingResponse failed to trigger aclose() on disconnect!"
    asyncio.run(_run())


def test_stream_terminates_on_is_disconnected():
    """Verify consult_stream terminates immediately when request.is_disconnected() returns True."""
    async def _run():
        sem = asyncio.Semaphore(1)
        mock_app = MagicMock()
        nodes_executed = []

        def long_stream(state, stream_mode):
            for i in range(5):
                nodes_executed.append(f"node_{i}")
                yield ("updates", {f"node_{i}": {"intent": f"val_{i}"}})

        mock_app.stream = long_stream
        init_router({"graph_app": mock_app}, sem)

        # Mock Request where is_disconnected returns True on 2nd check
        mock_request = MagicMock(spec=Request)
        disconnect_calls = 0

        async def mock_is_disconnected():
            nonlocal disconnect_calls
            disconnect_calls += 1
            return disconnect_calls >= 2

        mock_request.is_disconnected = mock_is_disconnected
        body = ConsultationRequest(query="断开测试", session_id="sid_disc")

        with patch("src.api.routes.history_store.load_context", return_value=[]), \
             patch("src.api.routes.history_store.cache_get", return_value=None), \
             patch("src.api.routes.history_store.cache_set") as mock_cache_set, \
             patch("src.api.routes.history_store.save_turn") as mock_save_turn, \
             patch("config.settings.settings.api_timeout_seconds", 5.0):

            resp = await consult_stream(mock_request, body=body)
            events = []
            async for chunk in resp.body_iterator:
                events.append(chunk)

            assert len(nodes_executed) < 5, "Stream should have terminated early without running all nodes"
            assert not mock_cache_set.called, "Must not backfill cache when disconnected early"
            assert not mock_save_turn.called, "Must not save turn when disconnected early"
            assert sem._value == 1, "Semaphore permit must be restored after disconnect"
    asyncio.run(_run())


# ============================================================================
# 4. Symmetric Cache Backfill (Sync <-> SSE)
# ============================================================================

def test_symmetric_cache_backfill_sse_to_sync():
    """Verify that SSE stream populates cache symmetrically so /consult hits it."""
    async def _run():
        cache_store = {}
        history_turns = []

        def fake_cache_set(k, v, ttl):
            cache_store[k] = v

        def fake_cache_get(k):
            return cache_store.get(k)

        def fake_save_turn(sid, q, a, i, **kw):
            history_turns.append((sid, q, a, i))
            return True

        sem = asyncio.Semaphore(5)
        mock_app = MagicMock()

        def mock_stream(state, stream_mode):
            yield ("updates", {"intent_classifier": {"intent": "medication"}})
            yield ("values", {
                "final_answer": "阿莫西林胶囊每日三次，每次一颗。",
                "intent": "medication",
                "departments": ["内科"],
                "disclaimers": ["请遵医嘱"],
            })

        mock_app.stream = mock_stream
        init_router({"graph_app": mock_app}, sem)

        # 1. Call SSE stream with session_1
        req_sse = ConsultationRequest(query="阿莫西林怎么吃", session_id="user_alice")
        with patch("src.api.routes.history_store.load_context", return_value=[]), \
             patch("src.api.routes.history_store.cache_get", side_effect=fake_cache_get), \
             patch("src.api.routes.history_store.cache_set", side_effect=fake_cache_set), \
             patch("src.api.routes.history_store.save_turn", side_effect=fake_save_turn), \
             patch("config.settings.settings.api_timeout_seconds", 5.0):

            sse_resp = await consult_stream(req_sse)
            events = []
            async for chunk in sse_resp.body_iterator:
                events.append(chunk)

        assert len(cache_store) == 1, "SSE stream must backfill answer cache"
        cached_key = list(cache_store.keys())[0]

        # 2. Call sync /consult with session_2 and same query
        req_sync = ConsultationRequest(query="阿莫西林怎么吃", session_id="user_bob")
        with patch("src.api.routes.history_store.load_context", return_value=[]), \
             patch("src.api.routes.history_store.cache_get", side_effect=fake_cache_get), \
             patch("src.api.routes.history_store.save_turn", side_effect=fake_save_turn), \
             patch("config.settings.settings.api_timeout_seconds", 5.0):

            sync_resp = await consult(req_sync)

        # Verify sync hit the cache created by SSE:
        assert sync_resp.answer == "阿莫西林胶囊每日三次，每次一颗。"
        assert sync_resp.intent == "medication"
        assert sync_resp.session_id == "user_bob", "Must rebind to current caller's session_id!"
        assert "user_alice" not in cache_store[cached_key], "Session ID must be excluded from cached payload"
    asyncio.run(_run())


def test_symmetric_cache_backfill_sync_to_sse():
    """Verify that sync /consult populates cache symmetrically so /consult/stream hits it."""
    async def _run():
        cache_store = {}

        def fake_cache_set(k, v, ttl):
            cache_store[k] = v

        def fake_cache_get(k):
            return cache_store.get(k)

        sem = asyncio.Semaphore(5)
        mock_app = MagicMock()
        mock_app.invoke.return_value = {
            "final_answer": "布洛芬缓释胶囊用于缓解轻至中度疼痛。",
            "intent": "medication",
        }
        init_router({"graph_app": mock_app}, sem)

        # 1. Call sync /consult with session_alice
        req_sync = ConsultationRequest(query="布洛芬怎么吃", session_id="user_alice")
        with patch("src.api.routes.history_store.load_context", return_value=[]), \
             patch("src.api.routes.history_store.cache_get", side_effect=fake_cache_get), \
             patch("src.api.routes.history_store.cache_set", side_effect=fake_cache_set), \
             patch("src.api.routes.history_store.save_turn"), \
             patch("config.settings.settings.api_timeout_seconds", 5.0):

            await consult(req_sync)

        assert len(cache_store) == 1

        # 2. Call SSE /consult/stream with session_charlie
        req_sse = ConsultationRequest(query="布洛芬怎么吃", session_id="user_charlie")
        with patch("src.api.routes.history_store.load_context", return_value=[]), \
             patch("src.api.routes.history_store.cache_get", side_effect=fake_cache_get), \
             patch("src.api.routes.history_store.save_turn"), \
             patch("config.settings.settings.api_timeout_seconds", 5.0):

            sse_resp = await consult_stream(req_sse)
            events = []
            async for chunk in sse_resp.body_iterator:
                events.append(chunk)

        full_output = "".join(events)
        assert "event: answer" in full_output
        # Parse answer event data
        for line in full_output.split("\n"):
            if line.startswith("data: ") and "布洛芬" in line:
                data = json.loads(line[6:])
                assert data["session_id"] == "user_charlie"
                assert data["answer"] == "布洛芬缓释胶囊用于缓解轻至中度疼痛。"
                break
        else:
            pytest.fail("Answer event not found or does not contain expected data")
    asyncio.run(_run())


# ============================================================================
# 5. Exact Context Cache Key & Cross-Session Isolation
# ============================================================================

def test_exact_cache_key_context_differentiation():
    """Verify that different dialogue histories produce distinct cache keys (no truncation collision)."""
    # 6 turns where turns 3-6 are identical, but turn 1 differs (diabetes vs fracture)
    history_diabetes = [
        {"role": "user", "content": "我今年50岁，有10年糖尿病史"},
        {"role": "assistant", "content": "好的，请问血糖平时控制如何？"},
        {"role": "user", "content": "平时空腹7左右"},
        {"role": "assistant", "content": "了解。请问还有其他不适吗？"},
        {"role": "user", "content": "最近经常头晕"},
        {"role": "assistant", "content": "请问是早晨还是饭后头晕？"},
    ]
    history_fracture = [
        {"role": "user", "content": "我前天打球小腿骨折了"},
        {"role": "assistant", "content": "好的，请问已经打石膏了吗？"},
        {"role": "user", "content": "平时空腹7左右"},
        {"role": "assistant", "content": "了解。请问还有其他不适吗？"},
        {"role": "user", "content": "最近经常头晕"},
        {"role": "assistant", "content": "请问是早晨还是饭后头晕？"},
    ]
    query = "头晕要挂哪个科？"

    key_diab = _answer_cache_key(query, history_diabetes)
    key_frac = _answer_cache_key(query, history_fracture)

    assert key_diab != key_frac, "Cache key collision! History before last 4 turns must not be ignored."


def test_cache_payload_session_stripping_and_rebinding():
    """Verify that serialized payload strips session_id and deserialize binds fresh session_id and duration."""
    orig_resp = ConsultationResponse(
        answer="测试回答内容",
        intent="qa",
        session_id="sensitive_user_session_12345",
        duration_ms=456,
        disclaimers=["测试免责声明"],
    )

    payload = _serialize_cache_payload(orig_resp)
    assert "sensitive_user_session_12345" not in payload
    assert "456" not in payload

    # Rebind to another user session
    rebound = _deserialize_cache_payload(payload, "new_user_session_99999", 35)
    assert rebound.session_id == "new_user_session_99999"
    assert rebound.duration_ms == 35
    assert rebound.answer == "测试回答内容"
    assert rebound.intent == "qa"
    assert rebound.disclaimers == ["测试免责声明"]
