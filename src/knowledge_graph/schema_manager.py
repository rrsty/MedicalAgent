"""Neo4j 连接管理 + Schema 初始化（唯一约束与索引）。

连接统一走 get_driver() 单例；首次建图前运行 ensure_constraints()。
"""
import sys
import threading
from pathlib import Path

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from neo4j import GraphDatabase

from config.settings import settings
from src.common.logger import setup_logger
from src.knowledge_graph.schema import NODE_PROPERTIES, NodeType

logger = setup_logger("kg.schema_manager", "neo4j.log")

_driver = None
_driver_lock = threading.Lock()


def get_driver():
    """Neo4j Driver 进程级单例（连接池复用，线程安全初始化）。"""
    global _driver
    if _driver is None:
        with _driver_lock:
            if _driver is None:
                _driver = GraphDatabase.driver(
                    settings.neo4j_uri,
                    auth=(settings.neo4j_user, settings.neo4j_password),
                )
    return _driver


def close_driver() -> None:
    global _driver
    with _driver_lock:
        if _driver is not None:
            _driver.close()
            _driver = None


def verify_connectivity() -> bool:
    try:
        get_driver().verify_connectivity()
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Neo4j 连接失败: %s", exc)
        return False


def ensure_constraints() -> None:
    """为每种节点的唯一键创建约束（Neo4j 5 唯一约束自动附带 RANGE 索引）。"""
    with get_driver().session() as session:
        for node_type in NODE_PROPERTIES:
            label = node_type.value  # str 枚举 f-string 会渲染成 NodeType.XXX，必须取 value
            key_prop = "content" if node_type == NodeType.PRECAUTION else "name"
            session.run(
                f"""
                CREATE CONSTRAINT {label.lower()}_{key_prop}_unique IF NOT EXISTS
                FOR (n:{label}) REQUIRE n.{key_prop} IS UNIQUE
                """
            )
    logger.info("Schema 约束与索引已就绪（%d 种节点）", len(NODE_PROPERTIES))


def clear_graph() -> None:
    """清空全图（R5：契约变更后重建数据用 clear + build，不写迁移补丁）。"""
    with get_driver().session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    logger.info("知识图谱已清空")


if __name__ == "__main__":
    if verify_connectivity():
        ensure_constraints()
        print("✅ Neo4j Schema 初始化完成")
    else:
        print("❌ 无法连接 Neo4j，请先: docker compose up -d")
        sys.exit(1)
