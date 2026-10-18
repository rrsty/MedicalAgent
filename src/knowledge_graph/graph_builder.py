"""知识图谱构建器：entities.json / relations.json → Neo4j（MERGE 幂等 + 批量事务）。

用法：
    python src/knowledge_graph/graph_builder.py build    # 建图（先节点后关系）
    python src/knowledge_graph/graph_builder.py stats    # 统计节点/关系/密度
    python src/knowledge_graph/graph_builder.py clear    # 清空（R5：clear + build 重建）
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tqdm import tqdm

from config.paths import KG_ENTITIES_FILE, KG_RELATIONS_FILE
from src.common.logger import setup_logger
from src.common.utils import read_json
from src.knowledge_graph.schema import (
    NODE_PROPERTIES,
    RELATION_DATA_FIELDS,
    RELATION_ENDPOINTS,
    REVERSE_RELATIONS,
    NodeType,
    RelationType,
)
from src.knowledge_graph.schema_manager import (
    close_driver,
    ensure_constraints,
    get_driver,
    verify_connectivity,
)

logger = setup_logger("kg.builder", "neo4j.log")

BATCH_SIZE = 500  # 每 500 条提交一次事务


class KnowledgeGraphBuilder:
    """图谱构建：MERGE 确保幂等，重复执行不产生重复数据。"""

    def __init__(self):
        self.driver = get_driver()

    # ---------- 节点 ----------
    def build_nodes(self, entities: list[dict]) -> int:
        """批量 MERGE 节点（先建约束/索引）。返回成功数。"""
        ensure_constraints()
        created = 0
        with self.driver.session() as session:
            for i in tqdm(range(0, len(entities), BATCH_SIZE), desc="建节点"):
                batch = entities[i:i + BATCH_SIZE]
                with session.begin_transaction() as tx:
                    for entity in batch:
                        node_type = entity.get("type")
                        if node_type not in NODE_PROPERTIES:
                            logger.warning("未知节点类型，跳过: %s", node_type)
                            continue
                        # Precaution 以 content 为唯一键（Schema 契约：Precaution(content)）
                        if node_type == NodeType.PRECAUTION.value:
                            content = (entity.get("content") or "").strip()
                            if not content:
                                continue
                            tx.run(
                                "MERGE (n:Precaution {content: $content}) RETURN n.content",
                                content=content,
                            )
                            created += 1
                            continue
                        props = {
                            key: entity.get(key)
                            for key in NODE_PROPERTIES[node_type]
                            if entity.get(key) is not None
                        }
                        props["name"] = entity.get("name")
                        if not props["name"]:
                            continue
                        # Medication 增强 pass 补属性也要生效：ON MATCH 用 coalesce（增强 pass 兼容）
                        set_clause = ", ".join(
                            f"n.{key} = coalesce(n.{key}, ${key})" for key in props if key != "name"
                        )
                        tx.run(
                            f"MERGE (n:{node_type} {{name: $name}}) "
                            + (f"ON CREATE SET {set_clause} " if set_clause else "")
                            + (f"ON MATCH SET {set_clause} " if set_clause else "")
                            + "RETURN n.name",
                            **props,
                        )
                        created += 1
        return created

    # ---------- 关系 ----------
    def build_relations(self, relations: list[dict]) -> int:
        """批量 MERGE 关系；为 R6 例外对（TREATED_BY/BELONGS_TO）同时建反向边。"""
        created = 0
        skipped = 0
        with self.driver.session() as session:
            for i in tqdm(range(0, len(relations), BATCH_SIZE), desc="建关系"):
                batch = relations[i:i + BATCH_SIZE]
                with session.begin_transaction() as tx:
                    for rel in batch:
                        rel_type = rel.get("type")
                        data = rel.get("data", {})
                        if rel_type not in RELATION_ENDPOINTS:
                            skipped += 1
                            continue
                        start_field, end_field = RELATION_DATA_FIELDS[rel_type]
                        start_name = str(data.get(start_field, "") or "").strip()
                        end_name = str(data.get(end_field, "") or "").strip()
                        if not start_name or not end_name:
                            skipped += 1
                            continue
                        start_type, end_type = RELATION_ENDPOINTS[rel_type]
                        n = tx.run(
                            f"""
                            MATCH (a:{start_type.value} {{name: $start_name}})
                            MATCH (b:{end_type.value} {{name: $end_name}})
                            MERGE (a)-[:{rel_type}]->(b)
                            RETURN 1
                            """,
                            start_name=start_name, end_name=end_name,
                        ).single()
                        if n is None:
                            skipped += 1
                            continue
                        created += 1
                        # R6 例外：成对反向边（TREATS_DISEASE / HANDLES_DISEASE）
                        reverse_type = REVERSE_RELATIONS.get(rel_type)
                        if reverse_type:
                            tx.run(
                                f"""
                                MATCH (a:{start_type.value} {{name: $start_name}})
                                MATCH (b:{end_type.value} {{name: $end_name}})
                                MERGE (b)-[:{reverse_type.value}]->(a)
                                """,
                                start_name=start_name, end_name=end_name,
                            )
        if skipped:
            logger.warning("跳过 %d 条无法落库的关系（端点缺失或类型未知）", skipped)
        return created

    def build_all(self, entities: list[dict], relations: list[dict]) -> dict:
        t0 = time.perf_counter()
        n_nodes = self.build_nodes(entities)
        n_rels = self.build_relations(relations)
        stats = self.get_stats()
        stats["build_seconds"] = round(time.perf_counter() - t0, 1)
        stats["built_nodes"] = n_nodes
        stats["built_relations"] = n_rels
        return stats

    # ---------- 统计 ----------
    def get_stats(self) -> dict:
        with self.driver.session() as session:
            total_nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            total_rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]

            node_stats = {}
            for record in session.run("MATCH (n) UNWIND labels(n) AS l RETURN l, count(*) AS c"):
                node_stats[record["l"]] = record["c"]

            rel_stats = {
                record["t"]: record["c"]
                for record in session.run(
                    "MATCH ()-[r]->() UNWIND type(r) AS t RETURN t, count(*) AS c"
                )
            }

        density = round(total_rels / total_nodes, 2) if total_nodes else 0.0
        return {
            "total_nodes": total_nodes,
            "total_relations": total_rels,
            "node_stats": node_stats,
            "relation_stats": rel_stats,
            "graph_density": density,
        }

    def clear(self) -> None:
        from src.knowledge_graph.schema_manager import clear_graph

        clear_graph()


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in {"build", "stats", "clear"}:
        print("用法: python src/knowledge_graph/graph_builder.py [build|stats|clear]")
        sys.exit(1)

    if not verify_connectivity():
        print("❌ 无法连接 Neo4j，请先启动: docker compose up -d")
        sys.exit(1)

    builder = KnowledgeGraphBuilder()
    cmd = sys.argv[1]

    if cmd == "clear":
        builder.clear()
        print("✅ 知识图谱已清空")
    elif cmd == "build":
        entities = read_json(KG_ENTITIES_FILE).get("entities", [])
        relations = read_json(KG_RELATIONS_FILE).get("relations", [])
        print(f"读取 {len(entities)} 实体 / {len(relations)} 关系")
        stats = builder.build_all(entities, relations)
        print("✅ 建图完成:")
        print(f"  节点总数: {stats['total_nodes']}（按类型: {stats['node_stats']}）")
        print(f"  关系总数: {stats['total_relations']}（按类型: {stats['relation_stats']}）")
        print(f"  图谱密度: {stats['graph_density']}（合格标准 > 0.5）")
        print(f"  耗时: {stats['build_seconds']}s")
    elif cmd == "stats":
        stats = builder.get_stats()
        import json

        print(json.dumps(stats, ensure_ascii=False, indent=2))

    close_driver()


if __name__ == "__main__":
    main()
