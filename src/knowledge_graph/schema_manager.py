"""Neo4j 连接管理 + Schema 初始化（唯一约束与索引）。

连接统一走 get_driver() 单例；首次建图前运行 ensure_constraints()。
"""
from neo4j import GraphDatabase

from config.settings import settings
from src.common.logger import setup_logger
from src.knowledge_graph.schema import NODE_PROPERTIES

logger = setup_logger("kg.schema_manager", "neo4j.log")

_driver = None


def get_driver():
    """Neo4j Driver 进程级单例（连接池复用）。"""
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
    return _driver


def close_driver() -> None:
    global _driver
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
    """为每种节点的唯一键创建约束 + name 索引（MERGE 依赖索引才能高效工作）。"""
    with get_driver().session() as session:
        for node_type in NODE_PROPERTIES:
            label = node_type.value  # str 枚举 f-string 会渲染成 NodeType.XXX，必须取 value
            session.run(
                f"""
                CREATE CONSTRAINT {label.lower()}_name_unique IF NOT EXISTS
                FOR (n:{label}) REQUIRE n.name IS UNIQUE
                """
            )
            session.run(
                f"CREATE INDEX {label.lower()}_name_index IF NOT EXISTS "
                f"FOR (n:{label}) ON (n.name)"
            )
    logger.info("Schema 约束与索引已就绪（%d 种节点）", len(NODE_PROPERTIES))


def clear_graph() -> None:
    """清空全图（R5：契约变更后重建数据用 clear + build，不写迁移补丁）。"""
    with get_driver().session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    logger.info("知识图谱已清空")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    if verify_connectivity():
        ensure_constraints()
        print("✅ Neo4j Schema 初始化完成")
    else:
        print("❌ 无法连接 Neo4j，请先: docker compose up -d")
        sys.exit(1)
