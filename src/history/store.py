"""对话历史存储：冷热分层企业方案。

- 热缓存 Redis：sess:{sid} List，LPUSH 插头 + LTRIM 保留最近 12 轮（24 条消息）+ EXPIRE 24h；
- 冷存储 MySQL：conversations / messages 两表（utf8mb4），全量历史、唯一事实源；
- 双写容错：先写 MySQL 再写 Redis；Redis 失败 catch 降级（缓存可缺不可幻），
  MySQL 失败不落 Redis —— 恒定保持「缓存 ⊆ 数据库」；
- 读取三级命中：Redis → 未命中从 MySQL 回填最近 12 轮 → 原流程；
- session_id == conversation_id：一个 id 串起前端会话、Redis、MySQL 三层。

所有 SQL 均为字面量 + 参数化占位符（%s），无任何字符串拼接用户输入。
"""
import json
import uuid
from datetime import datetime, timedelta

import mysql.connector
import redis as redis_lib

from config.settings import settings
from src.common.logger import setup_logger

logger = setup_logger("history.store", "history.log")


def new_session_id() -> str:
    return str(uuid.uuid4())


# ================= MySQL 层（冷存储，唯一事实源）=================

def _mysql_conn():
    return mysql.connector.connect(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password,
        database=settings.mysql_database,
        charset="utf8mb4",
        connection_timeout=5,
    )


_TABLES_INITIALIZED = False


def _ensure_tables(conn) -> None:
    """首次使用自动建表（utf8mb4 是存中文/emoji 的底线）；每进程只执行一次。"""
    global _TABLES_INITIALIZED
    if _TABLES_INITIALIZED:
        return
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id VARCHAR(64) PRIMARY KEY,
            title VARCHAR(255) NOT NULL DEFAULT '新对话',
            created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL,
            INDEX idx_updated (updated_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            conversation_id VARCHAR(64) NOT NULL,
            role VARCHAR(16) NOT NULL,
            content TEXT NOT NULL,
            intent VARCHAR(32) NOT NULL DEFAULT '',
            created_at DATETIME NOT NULL,
            INDEX idx_conversation (conversation_id),
            CONSTRAINT fk_msg_conv FOREIGN KEY (conversation_id)
                REFERENCES conversations(id) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
    )
    conn.commit()
    cursor.close()
    _TABLES_INITIALIZED = True


def insert_message(cid: str, role: str, content: str, intent: str = "") -> bool:
    """写一行消息；conversation 不存在则自动创建（title=首条 user 内容截断 24 字）。"""
    conn = None
    try:
        conn = _mysql_conn()
        _ensure_tables(conn)
        now = datetime.now()
        cursor = conn.cursor()

        cursor.execute("SELECT id, title FROM conversations WHERE id = %s", (cid,))
        row = cursor.fetchone()
        if row is None:
            title = content.strip().replace("\n", " ")[:24] if role == "user" else "新对话"
            cursor.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) "
                "VALUES (%s, %s, %s, %s)",
                (cid, title, now, now),
            )
        elif role == "user" and row[1] == "新对话":
            cursor.execute(
                "UPDATE conversations SET title = %s WHERE id = %s",
                (content.strip().replace("\n", " ")[:24], cid),
            )

        cursor.execute(
            "INSERT INTO messages (conversation_id, role, content, intent, created_at) "
            "VALUES (%s, %s, %s, %s, %s)",
            (cid, role, content, intent, now),
        )
        cursor.execute("UPDATE conversations SET updated_at = %s WHERE id = %s", (now, cid))
        conn.commit()
        cursor.close()
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("MySQL 写入失败（不落 Redis）: %s", exc)
        return False
    finally:
        if conn is not None:
            conn.close()


def list_conversations(limit: int = 50) -> list[dict]:
    """对话列表（按 updated_at 倒序）。"""
    conn = None
    try:
        conn = _mysql_conn()
        _ensure_tables(conn)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT id, title, created_at, updated_at FROM conversations "
            "ORDER BY updated_at DESC LIMIT %s",
            (limit,),
        )
        return cursor.fetchall()
    except Exception as exc:  # noqa: BLE001
        logger.error("MySQL 会话列表查询失败: %s", exc)
        return []
    finally:
        if conn is not None:
            conn.close()


def get_messages(cid: str) -> list[dict]:
    """某对话全部消息（时间正序）。"""
    conn = None
    try:
        conn = _mysql_conn()
        _ensure_tables(conn)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT role, content, intent, created_at FROM messages "
            "WHERE conversation_id = %s ORDER BY created_at ASC, id ASC",
            (cid,),
        )
        return cursor.fetchall()
    except Exception as exc:  # noqa: BLE001
        logger.error("MySQL 消息查询失败: %s", exc)
        return []
    finally:
        if conn is not None:
            conn.close()


def delete_conversation(cid: str) -> bool:
    """删除对话及消息（MySQL 两表 + Redis key 同步清空）。"""
    conn = None
    try:
        conn = _mysql_conn()
        _ensure_tables(conn)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages WHERE conversation_id = %s", (cid,))
        cursor.execute("DELETE FROM conversations WHERE id = %s", (cid,))
        conn.commit()
        cursor.close()
        delete_hot(cid)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("MySQL 删除失败: %s", exc)
        return False
    finally:
        if conn is not None:
            conn.close()


# ================= Redis 层（热缓存）=================

_redis_pool = None


def _get_redis():
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = redis_lib.ConnectionPool.from_url(settings.redis_url)
    return redis_lib.Redis(connection_pool=_redis_pool)


def hot_key(sid: str) -> str:
    return f"sess:{sid}"


def get_hot(sid: str) -> list[dict] | None:
    """读热缓存（LRANGE 后反序 = 时间正序）。失败/未命中返回 None。"""
    try:
        raw = _get_redis().lrange(hot_key(sid), 0, -1)
        if not raw:
            return None
        return [json.loads(item) for item in reversed(raw)]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Redis 读取失败（降级 MySQL）: %s", exc)
        return None


def append_hot(sid: str, turn: dict) -> None:
    """写一轮（LPUSH 插头 + LTRIM 保留 N 轮 + EXPIRE）。失败 catch 降级不影响主链路。"""
    try:
        client = _get_redis()
        key = hot_key(sid)
        client.lpush(key, json.dumps(turn, ensure_ascii=False))
        client.ltrim(key, 0, settings.history_window * 2 - 1)  # 一轮=2条消息
        client.expire(key, int(timedelta(hours=settings.history_ttl_hours).total_seconds()))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Redis 写入失败（缓存可缺不可幻）: %s", exc)


def delete_hot(sid: str) -> None:
    try:
        _get_redis().delete(hot_key(sid))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Redis 删除失败: %s", exc)


# ================= 通用短缓存（语义缓存/建议缓存等接入层优化用）=================

def cache_get(key: str) -> str | None:
    try:
        raw = _get_redis().get(key)
        return raw.decode("utf-8") if raw else None
    except Exception:  # noqa: BLE001
        return None


def cache_set(key: str, value: str, ttl_hours: int = 24) -> None:
    try:
        _get_redis().setex(key, int(timedelta(hours=ttl_hours).total_seconds()), value)
    except Exception as exc:  # noqa: BLE001
        logger.debug("cache_set 失败（降级）: %s", exc)


# ================= 统一入口 =================

def load_context(sid: str) -> list[dict]:
    """多轮上下文读取：先 Redis，未命中从 MySQL 取最近 N 轮并回填 Redis。"""
    hot = get_hot(sid)
    if hot is not None:
        return hot[-settings.history_window * 2:]  # 最近 12 轮 = 24 条消息

    rows = get_messages(sid)
    if not rows:
        return []

    # 相邻 (user, assistant) 配对成轮次序列
    turns: list[dict] = []
    pending_user: dict | None = None
    for row in rows:
        msg = {"role": row["role"], "content": row["content"]}
        if row["role"] == "user":
            pending_user = msg
        elif pending_user is not None:
            turns.append(pending_user)
            turns.append(msg)
            pending_user = None
    if pending_user is not None:
        turns.append(pending_user)

    context = turns[-settings.history_window:]
    for msg in context:
        append_hot(sid, msg)
    return context


def save_turn(sid: str, user_query: str, assistant_answer: str, intent: str) -> None:
    """双写一轮：先 MySQL（事实源）后 Redis；MySQL 失败则不写 Redis（缓存 ⊆ 数据库）。"""
    ok = insert_message(sid, "user", user_query, intent)
    if not ok:
        return
    insert_message(sid, "assistant", assistant_answer, intent)
    append_hot(sid, {"role": "user", "content": user_query})
    append_hot(sid, {"role": "assistant", "content": assistant_answer})
