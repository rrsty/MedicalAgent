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
        use_pure=True,
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
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS request_dedup (
            request_id VARCHAR(64) PRIMARY KEY,
            conversation_id VARCHAR(64) NOT NULL,
            created_at DATETIME NOT NULL,
            INDEX idx_conv (conversation_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
    )
    conn.commit()
    cursor.close()
    _TABLES_INITIALIZED = True


def insert_message(cid: str, role: str, content: str, intent: str = "", conn=None) -> bool:
    """写一行消息；conversation 不存在则自动创建（title=首条 user 内容截断 24 字）。

    若外部传入 conn，则在该连接/事务上执行且不 commit / close；
    若 conn 为 None，则自行创建连接、提交并关闭（保持向后兼容）。
    """
    should_close = False
    cursor = None
    try:
        if conn is None:
            conn = _mysql_conn()
            should_close = True
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
        if should_close:
            conn.commit()
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("MySQL 写入失败（不落 Redis）: %s", exc)
        if should_close and conn is not None:
            try:
                conn.rollback()
            except Exception:
                pass
        return False
    finally:
        if cursor is not None:
            try:
                cursor.close()
            except Exception:
                pass
        if should_close and conn is not None:
            try:
                conn.close()
            except Exception:
                pass


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
    """删除对话及消息（MySQL 三表 + Redis key 同步清空）。"""
    conn = None
    try:
        conn = _mysql_conn()
        _ensure_tables(conn)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM request_dedup WHERE conversation_id = %s", (cid,))
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

    context = turns[-settings.history_window * 2:]
    for msg in context:
        append_hot(sid, msg)
    return context


def save_turn(
    sid: str,
    user_query: str,
    assistant_answer: str,
    intent: str,
    request_id: str | None = None,
) -> bool:
    """双写一轮：在单一 MySQL 事务中原子落库 user 和 assistant 消息，成功后再写 Redis 缓存。

    - 幂等性：若提供 request_id，在同一事务中通过 request_dedup 表去重，已存在则直接返回 True；
    - 原子性：user 消息、assistant 消息、更新 conversation.updated_at 均在同一事务中，任一失败则全量回滚；
    - 容错补偿：MySQL 失败绝不写 Redis，返回 False；若 Redis 管道写入失败，则调用 delete_hot(sid) 触发失效补偿，
      恒定保证「缓存 ⊆ 数据库」。
    """
    conn = None
    try:
        conn = _mysql_conn()
        conn.autocommit = False
        _ensure_tables(conn)

        now = datetime.now()

        # 1. request_id 幂等去重检查与插入
        if request_id:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT request_id FROM request_dedup WHERE request_id = %s",
                (request_id,),
            )
            row = cursor.fetchone()
            cursor.close()
            if row is not None:
                logger.info("request_id %s 已存在，幂等跳过重复写入", request_id)
                conn.rollback()
                return True

            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO request_dedup (request_id, conversation_id, created_at) "
                "VALUES (%s, %s, %s)",
                (request_id, sid, now),
            )
            cursor.close()

        # 2. 在同一事务中插入 user 消息
        ok_user = insert_message(sid, "user", user_query, intent, conn=conn)
        if not ok_user:
            raise RuntimeError("写入 user 消息失败")

        # 3. 在同一事务中插入 assistant 消息
        ok_assistant = insert_message(sid, "assistant", assistant_answer, intent, conn=conn)
        if not ok_assistant:
            raise RuntimeError("写入 assistant 消息失败")

        # 4. 提交单一事务
        conn.commit()
    except Exception as exc:  # noqa: BLE001
        logger.error("MySQL 单事务保存一轮失败，执行回滚: %s", exc)
        if conn is not None:
            try:
                conn.rollback()
            except Exception as rb_exc:  # noqa: BLE001
                logger.error("MySQL 回滚失败: %s", rb_exc)
        return False
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    # 5. 提交成功后管道更新 Redis 热缓存（Post-Commit Cache Update）
    try:
        client = _get_redis()
        key = hot_key(sid)
        pipe = client.pipeline()
        pipe.lpush(key, json.dumps({"role": "user", "content": user_query}, ensure_ascii=False))
        pipe.lpush(key, json.dumps({"role": "assistant", "content": assistant_answer}, ensure_ascii=False))
        pipe.ltrim(key, 0, settings.history_window * 2 - 1)
        pipe.expire(key, int(timedelta(hours=settings.history_ttl_hours).total_seconds()))
        pipe.execute()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Redis 管道写入失败，触发失效补偿清理热缓存: %s", exc)
        delete_hot(sid)

    return True
