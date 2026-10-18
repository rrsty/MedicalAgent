"""知识图谱数据验证：连接 / 数量 / 孤立节点 / 疾病完整性 / 示例查询。

用法：python src/knowledge_graph/graph_validator.py
全部通过进程退出码 0，否则非 0（R7 模块自检）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.logger import setup_logger
from src.knowledge_graph.graph_queries import GraphQueries
from src.knowledge_graph.schema_manager import close_driver, verify_connectivity

logger = setup_logger("kg.validator", "neo4j.log")

EMERGENCY_KEYWORDS = ["胸痛", "呼吸困难", "意识模糊"]


def validate() -> bool:
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, ok, detail))
        print(f"{'✅' if ok else '❌'} {name}" + (f" | {detail}" if detail else ""))

    # 1. 连接测试
    check("Neo4j 连接", verify_connectivity())
    if not results[-1][1]:
        print("💡 请先启动: docker compose up -d")
        return False

    queries = GraphQueries()
    stats = queries.get_stats()

    # 2/3. 数量统计
    check("节点数量 > 0", stats["total_nodes"] > 0, f"共 {stats['total_nodes']}")
    check("关系数量 > 0", stats["total_relations"] > 0, f"共 {stats['total_relations']}")
    print(f"   节点分布: {stats['node_stats']}")

    # 4. 孤立节点检测
    try:
        from src.knowledge_graph.schema_manager import get_driver

        with get_driver().session() as session:
            isolated = session.run(
                "MATCH (n) WHERE NOT (n)--() RETURN count(n) AS c"
            ).single()["c"]
        ratio = isolated / max(stats["total_nodes"], 1)
        check("孤立节点率 < 20%", ratio < 0.2, f"{isolated} 个（{ratio:.1%}）")
    except Exception as exc:  # noqa: BLE001
        check("孤立节点检测", False, str(exc))

    # 5. 疾病完整性（每个疾病至少有症状与科室）
    try:
        from src.knowledge_graph.schema_manager import get_driver

        with get_driver().session() as session:
            record = session.run(
                """
                MATCH (d:Disease)
                OPTIONAL MATCH (d)-[:HAS_SYMPTOM]->(s:Symptom)
                OPTIONAL MATCH (d)-[:BELONGS_TO_DEPARTMENT]->(dep:Department)
                WITH d, count(DISTINCT s) AS sym_count, count(DISTINCT dep) AS dep_count
                RETURN count(d) AS diseases,
                       sum(CASE WHEN sym_count > 0 AND dep_count > 0 THEN 1 ELSE 0 END) AS complete
                """
            ).single()
        total_diseases = record["diseases"]
        complete = record["complete"]
        ok = total_diseases == 0 or complete / total_diseases >= 0.8
        check(
            "疾病完整性 ≥ 80%",
            ok,
            f"{complete}/{total_diseases} 个疾病同时有症状与科室",
        )
    except Exception as exc:  # noqa: BLE001
        check("疾病完整性检查", False, str(exc))

    # 6. 示例查询
    sample = queries.get_disease_info("高血压")
    check("示例查询（高血压）", bool(sample.get("symptoms") or sample.get("departments")),
          f"症状 {len(sample.get('symptoms', []))} 个 / 科室 {sample.get('departments', [])}")

    all_ok = all(ok for _, ok, _ in results)
    print(f"\n📊 验证结果: {sum(ok for _, ok, _ in results)}/{len(results)} 通过")
    return all_ok


if __name__ == "__main__":
    success = validate()
    close_driver()
    sys.exit(0 if success else 1)
