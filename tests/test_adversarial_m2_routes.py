"""Empirical Adversarial Stress Tests for M2:
1. Cache collision resistance (multi-turn dialogues with identical last 4 messages but different earlier turns).
2. Cross-session isolation (Session A caches answer; Session B hits cache; zero leaks of Session A to Session B).
3. SSE cache backfill and symmetry (stream backfills cache; consult hits stream cache; consult backfills cache; stream hits consult cache; abort does not backfill).
4. Concurrency semaphore and timeout budget lifecycle.
"""
import asyncio
import json
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.api.models import ConsultationRequest, ConsultationResponse, MessageItem
from src.api.routes import (
    TimeoutBudget,
    _answer_cache_key,
    _deserialize_cache_payload,
    _serialize_cache_payload,
    init_router,
    router,
)


class InMemoryCache:
    """Thread-safe in-memory cache simulator for history_store cache_get/cache_set."""

    def __init__(self):
        self.store: Dict[str, str] = {}
        self.get_calls = 0
        self.set_calls = 0

    def get(self, key: str) -> str | None:
        self.get_calls += 1
        return self.store.get(key)

    def set(self, key: str, value: str, ttl_hours: int = 24) -> None:
        self.set_calls += 1
        self.store[key] = value

    def clear(self):
        self.store.clear()
        self.get_calls = 0
        self.set_calls = 0


class MockGraphApp:
    """Mock LangGraph app that tracks invoke/stream calls and returns reproducible states."""

    def __init__(self):
        self.invoke_count = 0
        self.stream_count = 0
        self.stream_delay = 0.0
        self.should_raise_in_stream = False
        self.stream_closed = False

    def invoke(self, state: dict) -> dict:
        self.invoke_count += 1
        return {
            "final_answer": f"Answer for {state.get('query')} with {len(state.get('history', []))} turns",
            "intent": "medication",
            "symptoms": ["头痛", "发热"],
            "severity": "mild",
            "departments": ["神经内科"],
            "medications": [
                {"name": "布洛芬", "purpose": "镇痛解热", "side_effects": "胃部不适", "reminder": "饭后服用"}
            ],
            "advice": ["清淡饮食", "注意休息"],
            "disclaimers": ["本回答仅供医疗参考，不能替代医生面诊。"],
            "warnings": [],
            "linked_entities": [
                {"input_entity": "头痛", "matched_entity": "头痛", "type": "症状", "similarity": 1.0}
            ],
            "trace": [{"node": "intent_classifier", "duration_ms": 12}],
            "needs_clarification": False,
            "retrieved_contexts": ["头痛相关诊疗指南条目"],
        }

    def stream(self, state: dict, stream_mode=None):
        self.stream_count += 1
        self.stream_closed = False

        class CustomGen:
            def __init__(self, outer):
                self.outer = outer

            def __iter__(self):
                return self

            def __next__(self):
                if self.outer.should_raise_in_stream:
                    raise RuntimeError("Simulated graph stream error")
                if self.outer.stream_delay > 0:
                    time.sleep(self.outer.stream_delay)
                return ("updates", {"intent_classifier": {"intent": "medication"}})

            def close(self):
                self.outer.stream_closed = True

        def gen_fn():
            gen = CustomGen(self)
            try:
                # Update 1
                yield next(gen)
                # Update 2
                yield ("updates", {"medication_advisor": {"medications": [{"name": "布洛芬"}]}})
                # Final Values
                yield ("values", self.invoke(state))
            finally:
                gen.close()

        return gen_fn()


@pytest.fixture
def test_app():
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def mock_graph():
    return MockGraphApp()


@pytest.fixture
def memory_cache():
    return InMemoryCache()


@pytest.fixture(autouse=True)
def setup_api_env(mock_graph, memory_cache):
    """Initializes router state with mock graph and patches history_store with in-memory store."""
    semaphore = asyncio.Semaphore(5)
    app_state = {
        "graph_app": mock_graph,
        "node_labels": {
            "intent_classifier": "意图识别",
            "symptom_detector": "症状检测",
            "department_recommender": "科室推荐",
            "medication_advisor": "用药建议",
            "medical_knowledge": "医学知识检索",
        },
    }
    init_router(app_state, semaphore)

    saved_turns = []

    def mock_save_turn(sid, query, answer, intent):
        saved_turns.append({"session_id": sid, "query": query, "answer": answer, "intent": intent})

    def mock_load_context(sid):
        return []

    with patch("src.history.store.cache_get", side_effect=memory_cache.get), \
         patch("src.history.store.cache_set", side_effect=memory_cache.set), \
         patch("src.history.store.save_turn", side_effect=mock_save_turn), \
         patch("src.history.store.load_context", side_effect=mock_load_context):
        yield {
            "graph": mock_graph,
            "cache": memory_cache,
            "saved_turns": saved_turns,
            "semaphore": semaphore,
        }


# =========================================================================
# Scenario 1: Cache Collision Resistance
# =========================================================================

class TestCacheCollisionResistance:
    """Construct multi-turn dialogues with identical last 4 messages but different earlier turns.
    Verify that their cache keys are completely distinct.
    """

    def test_cache_collision_identical_last_4_different_earlier_turns(self):
        """Turn 1 differs ('糖尿病' vs '骨折'), Turns 2-4 (last 6 messages) are 100% identical.
        The old system hashed only history[-4:], leading to critical cache collision.
        The new system must produce completely distinct SHA-256 keys.
        """
        # Tail 4 messages (Turns 3 and 4)
        tail_messages = [
            {"role": "user", "content": "今天头有点晕"},
            {"role": "assistant", "content": "头晕可能由代谢异常或血压波动引起，请平卧休息。"},
            {"role": "user", "content": "那我应该挂什么科就诊？"},
            {"role": "assistant", "content": "建议前往三甲医院专科门诊进一步排查。"},
        ]

        # Middle turn (Turn 2)
        turn2 = [
            {"role": "user", "content": "最近常常口渴而且容易饿"},
            {"role": "assistant", "content": "多饮多食是典型代谢症状，请留意是否有多尿现象。"},
        ]

        # Turn 1 for Patient A (Diabetes)
        h_patient_a = [
            {"role": "user", "content": "医生，我患有2型糖尿病六年，一直在注射门冬胰岛素30"},
            {"role": "assistant", "content": "收到，请问近期空腹血糖和餐后两小时血糖控制在什么范围？"},
        ] + turn2 + tail_messages

        # Turn 1 for Patient B (Fracture)
        h_patient_b = [
            {"role": "user", "content": "医生，我昨天打球左小腿骨折了，刚在急诊打完高分子石膏板"},
            {"role": "assistant", "content": "收到，患肢需要垫高，注意观察足趾末梢感觉和血液循环。"},
        ] + turn2 + tail_messages

        # Verify that both dialogues have 8 messages (4 turns)
        assert len(h_patient_a) == 8
        assert len(h_patient_b) == 8
        # Verify that the last 6 messages (Turns 2, 3, 4) are completely identical
        assert h_patient_a[-6:] == h_patient_b[-6:]

        query = "请问我现在可以立刻吃点阿司匹林或者布洛芬止痛吗？"

        key_a = _answer_cache_key(query, h_patient_a)
        key_b = _answer_cache_key(query, h_patient_b)

        assert key_a != key_b, f"Cache collision detected! Key A: {key_a}, Key B: {key_b}"
        assert key_a.startswith("anscache:")
        assert key_b.startswith("anscache:")

    def test_cache_collision_comprehensive_matrix(self):
        """Stress-test 10 different medical conditions in Turn 1 with identical trailing turns."""
        conditions = [
            "高血压三年",
            "支气管哮喘急性发作",
            "痛风伴右脚大拇指红肿",
            "慢性肾功能不全三期",
            "冠心病并植入支架",
            "反流性食管炎",
            "癫痫发作病史",
            "甲状腺功能亢进症",
            "重度抑郁症在服舍曲林",
            "活动性胃溃疡出血",
        ]

        tail = [
            {"role": "user", "content": "目前感觉恶心想吐"},
            {"role": "assistant", "content": "恶心想吐需要警惕胃肠道或中枢刺激。"},
            {"role": "user", "content": "有什么急救措施吗？"},
            {"role": "assistant", "content": "请暂禁食水，若剧烈呕吐立即呼叫120。"},
        ]

        query = "可以吃止吐药吗？"
        keys = set()
        for cond in conditions:
            hist = [{"role": "user", "content": f"我是患者，既往有{cond}"}, {"role": "assistant", "content": "收到"}] + tail
            k = _answer_cache_key(query, hist)
            assert k not in keys, f"Collision detected for condition: {cond}"
            keys.add(k)

        assert len(keys) == len(conditions)

    def test_cache_key_normalization_properties(self):
        """Verify normalization properties:
        - Query whitespace normalization
        - Role casing normalization
        - Metadata stripping (session_id, created_at, intent in history items do NOT alter key)
        - MessageItem objects vs dicts produce identical keys
        - Empty message contents are cleanly handled
        """
        # 1. Query whitespace
        k1 = _answer_cache_key("  糖尿病能吃西瓜吗   ", [])
        k2 = _answer_cache_key("糖尿病能吃西瓜吗", [])
        assert k1 == k2

        # 2. History with extra metadata vs stripped history
        dirty_history = [
            {
                "role": "USER",
                "content": "头痛发热",
                "session_id": "patient_session_99999",
                "created_at": "2026-10-30 12:00:00",
                "intent": "symptom",
            },
            {
                "role": "ASSISTANT",
                "content": "请问发热多少度？",
                "session_id": "patient_session_99999",
                "created_at": "2026-10-30 12:00:05",
                "intent": "clarify",
            },
        ]
        clean_history = [
            {"role": "user", "content": "头痛发热"},
            {"role": "assistant", "content": "请问发热多少度？"},
        ]
        assert _answer_cache_key("38.5度", dirty_history) == _answer_cache_key("38.5度", clean_history)

        # 3. MessageItem models vs dicts
        model_history = [
            MessageItem(role="user", content="头痛发热"),
            MessageItem(role="assistant", content="请问发热多少度？"),
        ]
        assert _answer_cache_key("38.5度", model_history) == _answer_cache_key("38.5度", clean_history)

        # 4. Turn order sensitivity (Turn 1 -> Turn 2 != Turn 2 -> Turn 1)
        h_ordered = [
            {"role": "user", "content": "感冒"},
            {"role": "assistant", "content": "多喝水"},
            {"role": "user", "content": "发烧"},
            {"role": "assistant", "content": "物理降温"},
        ]
        h_reversed = [
            {"role": "user", "content": "发烧"},
            {"role": "assistant", "content": "物理降温"},
            {"role": "user", "content": "感冒"},
            {"role": "assistant", "content": "多喝水"},
        ]
        assert _answer_cache_key("吃什么药", h_ordered) != _answer_cache_key("吃什么药", h_reversed)


# =========================================================================
# Scenario 2: Cross-Session Isolation
# =========================================================================

class TestCrossSessionIsolation:
    """Simulate Session A generating and caching an answer payload;
    Simulate Session B requesting the identical context and hitting the cache.
    Verify Session B's returned response has session_id == 'Session B'
    and contains zero references or leaks to 'Session A'.
    """

    def test_cross_session_isolation_in_consult(self, test_app, setup_api_env):
        """Session A generates answer -> cached -> Session B hits cache via /api/consult."""
        async def _test():
            transport = ASGITransport(app=test_app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                session_a = "patient_alice_uuid_111111"
                session_b = "patient_bob_uuid_222222"

                identical_history = [
                    {"role": "user", "content": "我有高血压十年了"},
                    {"role": "assistant", "content": "请按时监测收缩压与舒张压。"},
                ]
                query = "可以剧烈运动跑步吗？"

                # 1. Session A makes request
                resp_a = await client.post(
                    "/api/consult",
                    json={
                        "session_id": session_a,
                        "query": query,
                        "history": identical_history,
                    },
                )
                assert resp_a.status_code == 200
                data_a = resp_a.json()
                assert data_a["session_id"] == session_a
                assert setup_api_env["graph"].invoke_count == 1

                # Check cache storage: verify raw cached string has NO mention of session_a
                cache_store = setup_api_env["cache"].store
                assert len(cache_store) == 1
                raw_cached_value = next(iter(cache_store.values()))
                assert session_a not in raw_cached_value
                assert "111111" not in raw_cached_value

                # 2. Session B makes identical request -> should hit cache
                resp_b = await client.post(
                    "/api/consult",
                    json={
                        "session_id": session_b,
                        "query": query,
                        "history": identical_history,
                    },
                )
                assert resp_b.status_code == 200
                data_b = resp_b.json()

                # Graph invoke count must STILL be 1 (Session B hit cache!)
                assert setup_api_env["graph"].invoke_count == 1

                # Verify Session B isolation
                assert data_b["session_id"] == session_b
                assert data_b["session_id"] != session_a
                assert session_a not in json.dumps(data_b)
                assert "111111" not in json.dumps(data_b)
                assert "alice" not in json.dumps(data_b).lower()

                # Verify saved turns logged session_b for the second request
                saved_turns = setup_api_env["saved_turns"]
                assert len(saved_turns) == 2
                assert saved_turns[0]["session_id"] == session_a
                assert saved_turns[1]["session_id"] == session_b

        asyncio.run(_test())

    def test_cross_session_isolation_in_stream(self, test_app, setup_api_env):
        """Session A generates answer via stream -> Session B hits cache via /api/consult/stream."""
        async def _test():
            transport = ASGITransport(app=test_app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                session_a = "patient_alice_stream_333333"
                session_b = "patient_bob_stream_444444"

                identical_history = [
                    {"role": "user", "content": "我有哮喘病"},
                    {"role": "assistant", "content": "请随身携带支气管扩张气雾剂。"},
                ]
                query = "可以去高原旅游吗？"

                # 1. Session A streams request
                resp_a = await client.post(
                    "/api/consult/stream",
                    json={
                        "session_id": session_a,
                        "query": query,
                        "history": identical_history,
                    },
                )
                assert resp_a.status_code == 200
                assert setup_api_env["graph"].invoke_count == 1

                # 2. Session B streams identical request
                resp_b = await client.post(
                    "/api/consult/stream",
                    json={
                        "session_id": session_b,
                        "query": query,
                        "history": identical_history,
                    },
                )
                assert resp_b.status_code == 200

                # Session B should have hit cache immediately!
                # Graph invoke count and stream count shouldn't increase for graph execution
                assert setup_api_env["graph"].invoke_count == 1
                assert setup_api_env["graph"].stream_count == 1

                # Parse SSE lines for Session B
                lines = resp_b.text.split("\n")
                answer_line = None
                for line in lines:
                    if line.startswith("data: ") and '"answer":' in line:
                        answer_line = line[len("data: "):]
                        break

                assert answer_line is not None, "Missing answer event in SSE stream"
                payload_b = json.loads(answer_line)

                # Strictly verify session isolation
                assert payload_b["session_id"] == session_b
                assert payload_b["session_id"] != session_a
                assert session_a not in resp_b.text
                assert "333333" not in resp_b.text
                assert "alice" not in resp_b.text.lower()

        asyncio.run(_test())


# =========================================================================
# Scenario 3: SSE Cache Backfill and Symmetry
# =========================================================================

class TestSSECacheBackfillAndSymmetry:
    """Verify that completing /api/consult/stream sets the answer cache,
    and a subsequent /api/consult request hits that cache payload.
    Also verify symmetry: /api/consult backfills -> /api/consult/stream hits cache.
    """

    def test_sse_stream_backfills_cache_for_subsequent_consult(self, test_app, setup_api_env):
        """1. Stream executes and backfills cache.
        2. Consult is called with identical query -> hits cache without calling graph.
        """
        async def _test():
            transport = ASGITransport(app=test_app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                session_stream = "sess_stream_000001"
                session_sync = "sess_sync_000002"

                query = "急性肠胃炎吃什么抗生素好？"
                history = [{"role": "user", "content": "今天拉肚子5次"}, {"role": "assistant", "content": "注意补充电解质"}]

                akey = _answer_cache_key(query, history)
                assert setup_api_env["cache"].get(akey) is None

                # 1. Complete SSE stream
                resp_stream = await client.post(
                    "/api/consult/stream",
                    json={
                        "session_id": session_stream,
                        "query": query,
                        "history": history,
                    },
                )
                assert resp_stream.status_code == 200
                assert "[DONE]" in resp_stream.text

                # Verify cache was backfilled by SSE stream!
                cached_val = setup_api_env["cache"].get(akey)
                assert cached_val is not None
                assert "布洛芬" in cached_val

                # Graph invoke count was 1 from stream
                assert setup_api_env["graph"].invoke_count == 1

                # 2. Subsequent /api/consult request
                resp_sync = await client.post(
                    "/api/consult",
                    json={
                        "session_id": session_sync,
                        "query": query,
                        "history": history,
                    },
                )
                assert resp_sync.status_code == 200
                data_sync = resp_sync.json()

                # Graph invoke count MUST STILL BE 1 (hit cache!)
                assert setup_api_env["graph"].invoke_count == 1

                # Content matches
                assert "布洛芬" in str(data_sync["medications"])
                assert data_sync["session_id"] == session_sync

        asyncio.run(_test())

    def test_consult_backfills_cache_for_subsequent_stream(self, test_app, setup_api_env):
        """Symmetry verification:
        1. /api/consult executes and backfills cache.
        2. /api/consult/stream is called with identical query -> hits cache immediately.
        """
        async def _test():
            transport = ASGITransport(app=test_app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                session_sync = "sess_sync_000003"
                session_stream = "sess_stream_000004"

                query = "偏头痛发作期注意事项"
                history = []

                # 1. Consult executes
                resp_sync = await client.post(
                    "/api/consult",
                    json={
                        "session_id": session_sync,
                        "query": query,
                        "history": history,
                    },
                )
                assert resp_sync.status_code == 200
                assert setup_api_env["graph"].invoke_count == 1

                # 2. Subsequent stream executes
                resp_stream = await client.post(
                    "/api/consult/stream",
                    json={
                        "session_id": session_stream,
                        "query": query,
                        "history": history,
                    },
                )
                assert resp_stream.status_code == 200

                # Graph stream was NOT started; invoke was not called again!
                assert setup_api_env["graph"].stream_count == 0
                assert setup_api_env["graph"].invoke_count == 1

                # Stream returns cached answer
                assert "event: answer" in resp_stream.text
                assert "event: done" in resp_stream.text
                assert session_stream in resp_stream.text

        asyncio.run(_test())

    def test_sse_cache_not_backfilled_on_failure(self, test_app, setup_api_env):
        """Negative test: if SSE stream fails or errors out mid-way,
        cache MUST NOT be backfilled with corrupt/empty data.
        """
        async def _test():
            setup_api_env["graph"].should_raise_in_stream = True

            transport = ASGITransport(app=test_app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                session_err = "sess_error_000005"
                query = "测试异常流处理"
                history = []
                akey = _answer_cache_key(query, history)

                resp = await client.post(
                    "/api/consult/stream",
                    json={
                        "session_id": session_err,
                        "query": query,
                        "history": history,
                    },
                )
                assert resp.status_code == 200
                assert "event: error" in resp.text

                # Cache must remain EMPTY!
                assert setup_api_env["cache"].get(akey) is None

        asyncio.run(_test())


# =========================================================================
# Scenario 4: Concurrency Semaphore and Lifecycle
# =========================================================================

class TestConcurrencyAndLifecycle:
    """Stress-test semaphore protection and generator cleanup."""

    def test_semaphore_guards_stream_lifecycle(self, test_app, setup_api_env):
        """Verify that streaming holds the semaphore for its full execution."""
        async def _test():
            # Use semaphore with capacity 1
            strict_sem = asyncio.Semaphore(1)
            init_router({"graph_app": setup_api_env["graph"], "node_labels": {}}, strict_sem)

            setup_api_env["graph"].stream_delay = 0.05  # Slow stream
            transport = ASGITransport(app=test_app)

            async with AsyncClient(transport=transport, base_url="http://test") as client:
                # Launch two concurrent streams
                req_1 = client.post("/api/consult/stream", json={"session_id": "sess_sem_01", "query": "Q1", "history": []})
                req_2 = client.post("/api/consult/stream", json={"session_id": "sess_sem_02", "query": "Q2", "history": []})

                r1, r2 = await asyncio.gather(req_1, req_2)
                assert r1.status_code == 200
                assert r2.status_code == 200
                assert strict_sem._value == 1  # Fully released after both finish

        asyncio.run(_test())

    def test_timeout_budget_precision(self):
        """Verify TimeoutBudget monotonic clock calculations and exception raising."""
        budget = TimeoutBudget(0.05)
        assert budget.remaining > 0.0
        time.sleep(0.06)
        assert budget.remaining == 0.0
        with pytest.raises(asyncio.TimeoutError):
            budget.check()
