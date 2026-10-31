"""Unit tests for History Store atomic transactions, idempotency, and compensation (Milestone M3, Features F11, F12, F13)."""
import json
from unittest.mock import patch, MagicMock
import pytest

from config.settings import settings
import src.history.store as store


class FakeMySQLCursor:
    def __init__(self, conn, dictionary=False):
        self.conn = conn
        self.dictionary = dictionary
        self.closed = False
        self._next_result = None

    def execute(self, query, params=None):
        self.conn.executed_queries.append((query.strip(), params))
        q = query.strip()

        if "CREATE TABLE" in q:
            pass
        elif "SELECT id, title FROM conversations" in q:
            cid = params[0] if params else None
            if cid in self.conn.conversations:
                row = self.conn.conversations[cid]
                self._next_result = (cid, row.get("title", "新对话")) if not self.dictionary else row
            else:
                self._next_result = None
        elif "SELECT request_id FROM request_dedup" in q:
            req_id = params[0] if params else None
            if req_id in self.conn.dedup_keys:
                self._next_result = (req_id,) if not self.dictionary else {"request_id": req_id}
            else:
                self._next_result = None
        elif "INSERT INTO request_dedup" in q:
            if self.conn.fail_on_dedup:
                raise RuntimeError("MySQL error inserting into request_dedup")
            self.conn.dedup_keys.add(params[0])
        elif "INSERT INTO conversations" in q:
            cid, title, c_at, u_at = params[0], params[1], params[2], params[3]
            self.conn.conversations[cid] = {"id": cid, "title": title, "created_at": c_at, "updated_at": u_at}
        elif "UPDATE conversations SET title" in q:
            cid = params[1]
            if cid in self.conn.conversations:
                self.conn.conversations[cid]["title"] = params[0]
        elif "UPDATE conversations SET updated_at" in q:
            cid = params[1]
            if cid in self.conn.conversations:
                self.conn.conversations[cid]["updated_at"] = params[0]
        elif "INSERT INTO messages" in q:
            cid, role, content, intent, c_at = params[0], params[1], params[2], params[3], params[4]
            if role == "user" and self.conn.fail_on_user:
                raise RuntimeError("MySQL error inserting user message")
            if role == "assistant" and self.conn.fail_on_assistant:
                raise RuntimeError("MySQL error inserting assistant message")
            self.conn.messages.append({
                "conversation_id": cid,
                "role": role,
                "content": content,
                "intent": intent,
                "created_at": c_at,
            })
        elif "DELETE FROM request_dedup" in q:
            cid = params[0]
            self.conn.deleted_dedup.append(cid)
        elif "DELETE FROM messages" in q:
            cid = params[0]
            self.conn.messages = [m for m in self.conn.messages if m["conversation_id"] != cid]
        elif "DELETE FROM conversations" in q:
            cid = params[0]
            self.conn.conversations.pop(cid, None)

    def fetchone(self):
        res = self._next_result
        self._next_result = None
        return res

    def fetchall(self):
        return []

    def close(self):
        self.closed = True


class FakeMySQLConnection:
    def __init__(self):
        self.autocommit = True
        self.committed = False
        self.rolled_back = False
        self.closed = False
        self.commit_count = 0
        self.rollback_count = 0
        self.executed_queries = []
        self.conversations = {}
        self.messages = []
        self.dedup_keys = set()
        self.deleted_dedup = []
        self.fail_on_user = False
        self.fail_on_assistant = False
        self.fail_on_commit = False
        self.fail_on_dedup = False

    def cursor(self, dictionary=False):
        return FakeMySQLCursor(self, dictionary=dictionary)

    def commit(self):
        if self.fail_on_commit:
            raise RuntimeError("MySQL error during commit")
        self.committed = True
        self.commit_count += 1

    def rollback(self):
        self.rolled_back = True
        self.rollback_count += 1

    def close(self):
        self.closed = True


class FakeRedisPipeline:
    def __init__(self, client):
        self.client = client
        self.commands = []

    def lpush(self, key, value):
        self.commands.append(("lpush", key, value))
        return self

    def ltrim(self, key, start, end):
        self.commands.append(("ltrim", key, start, end))
        return self

    def expire(self, key, ttl):
        self.commands.append(("expire", key, ttl))
        return self

    def execute(self):
        if self.client.fail_on_pipe_execute:
            raise RuntimeError("Redis connection broken during pipeline execute")
        for cmd in self.commands:
            if cmd[0] == "lpush":
                key, val = cmd[1], cmd[2]
                self.client.lists.setdefault(key, []).insert(0, val)
        self.commands = []
        return [1, True, True]


class FakeRedisClient:
    def __init__(self):
        self.lists = {}
        self.deleted_keys = []
        self.fail_on_pipe_execute = False

    def pipeline(self):
        return FakeRedisPipeline(self)

    def delete(self, key):
        self.deleted_keys.append(key)
        self.lists.pop(key, None)
        return 1

    def lpush(self, key, value):
        self.lists.setdefault(key, []).insert(0, value)
        return len(self.lists[key])

    def ltrim(self, key, start, end):
        if key in self.lists:
            self.lists[key] = self.lists[key][start : end + 1]

    def expire(self, key, seconds):
        return 1

    def lrange(self, key, start, end):
        if key not in self.lists:
            return []
        items = self.lists[key]
        if end == -1:
            return items[start:]
        return items[start : end + 1]


@pytest.fixture(autouse=True)
def clean_store_module():
    """Ensure src.history.store functions are isolated from leaky patches in other test suites."""
    import importlib
    importlib.reload(store)
    store._TABLES_INITIALIZED = True
    yield
    importlib.reload(store)


@pytest.fixture
def mock_storage():
    """Setup fake MySQL and Redis storage instances."""
    store._TABLES_INITIALIZED = True  # Avoid table creation DDL during test runs
    fake_conn = FakeMySQLConnection()
    fake_redis = FakeRedisClient()

    with patch("src.history.store._mysql_conn", return_value=fake_conn), \
         patch("src.history.store._get_redis", return_value=fake_redis):
        yield fake_conn, fake_redis


def test_save_turn_success(mock_storage):
    """Happy path: single DB transaction commits both messages, then Redis hot cache is written."""
    conn, redis = mock_storage
    sid = "sess-001"
    query = "感冒发烧吃什么药？"
    answer = "可遵医嘱服用对乙酰氨基酚或布洛芬退热。"
    intent = "medication"

    res = store.save_turn(sid, query, answer, intent)
    assert res is True
    assert conn.committed is True
    assert conn.commit_count == 1
    assert not conn.rolled_back

    # Verify MySQL data
    assert len(conn.messages) == 2
    assert conn.messages[0]["role"] == "user"
    assert conn.messages[0]["content"] == query
    assert conn.messages[1]["role"] == "assistant"
    assert conn.messages[1]["content"] == answer

    # Verify Redis pipeline hot cache data (in chronological order when read via get_hot)
    hot = store.get_hot(sid)
    assert hot is not None
    assert len(hot) == 2
    assert hot[0]["role"] == "user"
    assert hot[0]["content"] == query
    assert hot[1]["role"] == "assistant"
    assert hot[1]["content"] == answer


def test_save_turn_user_fail_rollbacks(mock_storage):
    """Failure inserting user message triggers DB rollback, leaves Redis untouched, returns False."""
    conn, redis = mock_storage
    conn.fail_on_user = True
    sid = "sess-fail-user"

    res = store.save_turn(sid, "头痛", "建议检查血压", "appointment")
    assert res is False
    assert conn.rolled_back is True
    assert conn.commit_count == 0
    assert not conn.committed

    # Redis must be completely untouched
    assert store.get_hot(sid) is None
    assert f"sess:{sid}" not in redis.lists


def test_save_turn_assistant_fail_rollbacks(mock_storage):
    """Failure inserting assistant message triggers DB rollback of entire turn, Redis untouched."""
    conn, redis = mock_storage
    conn.fail_on_assistant = True
    sid = "sess-fail-asst"

    res = store.save_turn(sid, "高血压怎么办", "按时服药", "medication")
    assert res is False
    assert conn.rolled_back is True
    assert conn.commit_count == 0
    assert not conn.committed

    # Redis must NOT contain user or assistant messages
    assert store.get_hot(sid) is None
    assert f"sess:{sid}" not in redis.lists


def test_save_turn_commit_fail_rollbacks(mock_storage):
    """Exception during conn.commit() triggers rollback and keeps Redis untouched."""
    conn, redis = mock_storage
    conn.fail_on_commit = True
    sid = "sess-fail-commit"

    res = store.save_turn(sid, "胸闷", "请立即就医", "emergency")
    assert res is False
    assert conn.rolled_back is True
    assert store.get_hot(sid) is None


def test_save_turn_redis_fail_triggers_invalidation_compensation(mock_storage):
    """When DB succeeds but Redis pipeline throws an error, delete_hot(sid) compensation is called."""
    conn, redis = mock_storage
    redis.fail_on_pipe_execute = True
    sid = "sess-redis-fail"

    # Pre-populate some dirty data in redis
    redis.lists[f"sess:{sid}"] = [json.dumps({"role": "user", "content": "旧缓存"})]

    res = store.save_turn(sid, "嗓子疼", "多喝温水", "advice")
    # DB succeeded so save_turn returns True
    assert res is True
    assert conn.committed is True

    # Invalidation compensation must have deleted the hot key
    assert f"sess:{sid}" in redis.deleted_keys
    assert f"sess:{sid}" not in redis.lists


def test_save_turn_idempotent_with_request_id(mock_storage):
    """Duplicate request_id calls are idempotent and do not double insert messages."""
    conn, redis = mock_storage
    sid = "sess-dedup"
    req_id = "req-uuid-12345"

    # First call: succeeds and registers request_id
    res1 = store.save_turn(sid, "咳嗽", "止咳糖浆", "medication", request_id=req_id)
    assert res1 is True
    assert req_id in conn.dedup_keys
    assert len(conn.messages) == 2
    assert conn.commit_count == 1

    # Second call with identical request_id: idempotent skip
    res2 = store.save_turn(sid, "咳嗽", "止咳糖浆", "medication", request_id=req_id)
    assert res2 is True
    # Still 2 messages in MySQL (no double insertion!)
    assert len(conn.messages) == 2
    # Commit count did not increase (rollback or skip without commit)
    assert conn.commit_count == 1


def test_load_context_cache_hit_returns_aligned_window(mock_storage):
    """When hot cache hits, returns up to history_window * 2 (24 messages)."""
    conn, redis = mock_storage
    sid = "sess-cache-hit"

    # Build 30 messages (15 turns)
    raw_msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"} for i in range(30)]
    # Store in Redis (head-inserted, so reverse order in list)
    redis.lists[f"sess:{sid}"] = [json.dumps(m) for m in reversed(raw_msgs)]

    context = store.load_context(sid)
    expected_count = settings.history_window * 2  # 24 messages
    assert len(context) == expected_count
    # Should be the most recent 24 messages
    assert context[0]["content"] == "msg 6"
    assert context[-1]["content"] == "msg 29"


def test_load_context_cache_miss_mysql_fallback_alignment(mock_storage):
    """When hot cache misses, MySQL penetration loads aligned window (24 messages / 12 turns) and backfills Redis."""
    conn, redis = mock_storage
    sid = "sess-cache-miss"

    # Provide 30 messages in MySQL (15 turns)
    all_rows = []
    for i in range(30):
        all_rows.append({
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"mysql_msg_{i}",
            "intent": "general",
            "created_at": None,
        })

    with patch("src.history.store.get_messages", return_value=all_rows):
        context = store.load_context(sid)

    expected_count = settings.history_window * 2  # 24 messages
    assert len(context) == expected_count
    assert context[0]["content"] == "mysql_msg_6"
    assert context[-1]["content"] == "mysql_msg_29"

    # Redis should be backfilled with these 24 messages
    assert f"sess:{sid}" in redis.lists


def test_insert_message_external_conn_reused():
    """insert_message reuses external conn and does not commit or close it."""
    fake_conn = FakeMySQLConnection()
    ok = store.insert_message("test_cid", "user", "测试问题", "intent_a", conn=fake_conn)
    assert ok is True
    assert not fake_conn.committed
    assert not fake_conn.closed
    assert len(fake_conn.messages) == 1


def test_delete_conversation_cleans_request_dedup_and_redis(mock_storage):
    """delete_conversation deletes messages, conversations, request_dedup, and Redis hot cache."""
    conn, redis = mock_storage
    cid = "sess-delete-target"
    redis.lists[f"sess:{cid}"] = ["mock_cache"]

    ok = store.delete_conversation(cid)
    assert ok is True
    assert conn.committed is True
    assert cid in conn.deleted_dedup
    assert f"sess:{cid}" in redis.deleted_keys
