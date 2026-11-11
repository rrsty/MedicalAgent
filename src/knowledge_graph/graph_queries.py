"""图谱查询服务：Agent / 工具层的 Neo4j 只读查询入口。

约定：所有方法查询失败返回空结果并记日志（绝不抛异常打断 Agent 主流程）；
关系反查用 Cypher 反向遍历（R6）。
"""
from config.settings import settings
from src.common.provenance import decode_provenance
from src.common.logger import setup_logger
from src.knowledge_graph.schema_manager import get_driver

logger = setup_logger("kg.queries", "neo4j.log")


class GraphQueries:
    """知识图谱只读查询（供科室推荐 / 用药建议 / ReAct 工具 / Text2Cypher 使用）。"""

    # ---------- 疾病 ----------
    def get_disease_info(self, name: str) -> dict:
        """疾病摘要：症状 / 科室 / 药物 / 检查 / 身体部位。"""
        try:
            with get_driver().session() as session:
                record = session.run(
                    """
                    MATCH (d:Disease {name: $name})
                    OPTIONAL MATCH (d)-[:HAS_SYMPTOM]->(s:Symptom)
                    OPTIONAL MATCH (d)-[:BELONGS_TO_DEPARTMENT]->(dep:Department)
                    OPTIONAL MATCH (d)-[:TREATED_BY_MEDICATION]->(m:Medication)
                    OPTIONAL MATCH (d)-[:NEEDS_EXAMINATION]->(e:Examination)
                    OPTIONAL MATCH (d)-[:AFFECTS_BODY_PART]->(b:BodyPart)
                    RETURN d.name AS name, d.description AS description,
                           collect(DISTINCT s.name) AS symptoms,
                           collect(DISTINCT dep.name) AS departments,
                           collect(DISTINCT m.name) AS medications,
                           collect(DISTINCT e.name) AS examinations,
                           collect(DISTINCT b.name) AS body_parts
                    """,
                    name=name,
                ).single()
                if record is None:
                    return {}
                return {
                    key: (record[key] or [] if isinstance(record[key], list) else record[key])
                    for key in (
                        "name", "description", "symptoms", "departments",
                        "medications", "examinations", "body_parts",
                    )
                }
        except Exception as exc:  # noqa: BLE001
            logger.error("get_disease_info(%s) 失败: %s", name, exc)
            return {}

    def get_symptoms_for_disease(self, name: str) -> list[str]:
        try:
            with get_driver().session() as session:
                return [
                    r["symptom"] for r in session.run(
                        "MATCH (d:Disease {name:$name})-[:HAS_SYMPTOM]->(s:Symptom) "
                        "RETURN s.name AS symptom",
                        name=name,
                    )
                ]
        except Exception as exc:  # noqa: BLE001
            logger.error("get_symptoms_for_disease(%s) 失败: %s", name, exc)
            return []

    def match_disease_in_text(self, text: str) -> str | None:
        """返回文本中包含的图谱疾病名（最长匹配优先）。

        用于「颈椎病应该挂什么科？」这类直接点名疾病的问句——
        query CONTAINS 疾病名即可命中，无需精确整句匹配。
        """
        try:
            with get_driver().session() as session:
                record = session.run(
                    """
                    MATCH (d:Disease)
                    WHERE $text CONTAINS d.name
                    RETURN d.name AS name ORDER BY size(d.name) DESC LIMIT 1
                    """,
                    text=text,
                ).single()
                return record["name"] if record else None
        except Exception as exc:  # noqa: BLE001
            logger.error("match_disease_in_text 失败: %s", exc)
            return None

    def get_diseases_for_symptom(self, symptom: str, limit: int = 10) -> list[str]:
        """反向遍历：症状 → 疾病（R6 单向关系的 Cypher 反查）。"""
        try:
            with get_driver().session() as session:
                return [
                    r["disease"] for r in session.run(
                        "MATCH (s:Symptom {name:$symptom})<-[:HAS_SYMPTOM]-(d:Disease) "
                        "RETURN d.name AS disease LIMIT $limit",
                        symptom=symptom, limit=limit,
                    )
                ]
        except Exception as exc:  # noqa: BLE001
            logger.error("get_diseases_for_symptom(%s) 失败: %s", symptom, exc)
            return []

    def get_related_diseases(self, name: str, limit: int = 5) -> list[str]:
        """共症状相关疾病（多跳：疾病→症状←疾病）。"""
        try:
            with get_driver().session() as session:
                return [
                    r["related"] for r in session.run(
                        """
                        MATCH (d:Disease {name:$name})-[:HAS_SYMPTOM]->(s:Symptom)
                              <-[:HAS_SYMPTOM]-(other:Disease)
                        WHERE other.name <> $name
                        RETURN other.name AS related, count(*) AS shared
                        ORDER BY shared DESC LIMIT $limit
                        """,
                        name=name, limit=limit,
                    )
                ]
        except Exception as exc:  # noqa: BLE001
            logger.error("get_related_diseases(%s) 失败: %s", name, exc)
            return []

    # ---------- 科室 ----------
    def get_departments_for_symptom(self, symptom: str) -> dict:
        """症状 → 疾病 → 科室（按匹配疾病数排序），导诊核心查询。"""
        try:
            with get_driver().session() as session:
                records = session.run(
                    """
                    MATCH (s:Symptom {name: $symptom})
                          <-[:HAS_SYMPTOM]-(d:Disease)
                          -[:BELONGS_TO_DEPARTMENT]->(dep:Department)
                    RETURN dep.name AS department, count(DISTINCT d) AS score
                    ORDER BY score DESC
                    """,
                    symptom=symptom,
                )
                return {r["department"]: r["score"] for r in records}
        except Exception as exc:  # noqa: BLE001
            logger.error("get_departments_for_symptom(%s) 失败: %s", symptom, exc)
            return {}

    def get_diseases_for_department(self, department: str, limit: int = 20) -> list[str]:
        try:
            with get_driver().session() as session:
                return [
                    r["disease"] for r in session.run(
                        "MATCH (dep:Department {name:$dep})<-[:BELONGS_TO_DEPARTMENT]-"
                        "(d:Disease) RETURN d.name AS disease LIMIT $limit",
                        dep=department, limit=limit,
                    )
                ]
        except Exception as exc:  # noqa: BLE001
            logger.error("get_diseases_for_department(%s) 失败: %s", department, exc)
            return []

    # ---------- 药物 ----------
    def get_medications_for_disease(self, name: str) -> list[dict]:
        try:
            with get_driver().session() as session:
                return [
                    {
                        "name": r["name"],
                        "category": r["category"],
                        "side_effects": r["side_effects"],
                    }
                    for r in session.run(
                        """
                        MATCH (d:Disease {name:$name})-[:TREATED_BY_MEDICATION]->(m:Medication)
                        RETURN m.name AS name, m.category AS category,
                               m.side_effects AS side_effects
                        """,
                        name=name,
                    )
                ]
        except Exception as exc:  # noqa: BLE001
            logger.error("get_medications_for_disease(%s) 失败: %s", name, exc)
            return []

    def get_medication_info(self, name: str) -> dict:
        """药物详情：类别 / 副作用 / 禁忌 / 治疗疾病（属性缺失用 .get 容错）。"""
        try:
            with get_driver().session() as session:
                record = session.run(
                    """
                    MATCH (m:Medication {name:$name})
                    OPTIONAL MATCH (m)-[:TREATS_DISEASE]->(d:Disease)
                    RETURN m.name AS name,
                           m.category AS category,
                           m.side_effects AS side_effects,
                           m.contraindications AS contraindications,
                           collect(DISTINCT d.name) AS treats
                    """,
                    name=name,
                ).single()
                if record is None:
                    return {}
                return {
                    "name": record["name"],
                    "category": record.get("category"),
                    "side_effects": record.get("side_effects") or [],
                    "contraindications": record.get("contraindications") or [],
                    "treats": record.get("treats") or [],
                }
        except Exception as exc:  # noqa: BLE001
            logger.error("get_medication_info(%s) 失败: %s", name, exc)
            return {}

    def get_medication_interactions(self, name: str) -> list[str]:
        try:
            with get_driver().session() as session:
                return [
                    r["other"] for r in session.run(
                        """
                        MATCH (m:Medication {name:$name})-[r:INTERACTS_WITH]-(other:Medication)
                        RETURN other.name AS other
                        """,
                        name=name,
                    )
                ]
        except Exception as exc:  # noqa: BLE001
            logger.error("get_medication_interactions(%s) 失败: %s", name, exc)
            return []

    # ---------- 实体查找 ----------
    def find_entity(self, keyword: str) -> dict | None:
        """按名称精确/前缀查找实体（图谱检索的精确匹配路）。"""
        try:
            with get_driver().session() as session:
                record = session.run(
                    """
                    MATCH (n)
                    WHERE n.name = $kw OR n.name STARTS WITH $kw
                    RETURN labels(n)[0] AS type, n.name AS name,
                           coalesce(n.description, '') AS description
                    LIMIT 1
                    """,
                    kw=keyword,
                ).single()
                if record is None:
                    return None
                return {
                    "type": record["type"],
                    "name": record["name"],
                    "description": record["description"],
                }
        except Exception as exc:  # noqa: BLE001
            logger.error("find_entity(%s) 失败: %s", keyword, exc)
            return None

    def expand_entity(self, name: str) -> dict:
        """按实体类型扩展结构化事实（强制检索 Step 1 的图谱路）。

        Disease → 症状/科室/药物；Symptom → 疾病/科室；Medication → 详情。
        """
        try:
            with get_driver().session() as session:
                record = session.run(
                    """
                    MATCH (n {name: $name})
                    WITH n, labels(n)[0] AS t
                    RETURN t AS type,
                    CASE t
                      WHEN 'Disease' THEN {
                        symptoms: [(n)-[:HAS_SYMPTOM]->(s:Symptom) | s.name],
                        departments: [(n)-[:BELONGS_TO_DEPARTMENT]->(dep:Department) | dep.name],
                        medications: [(n)-[:TREATED_BY_MEDICATION]->(m:Medication) | m.name],
                        examinations: [(n)-[:NEEDS_EXAMINATION]->(e:Examination) | e.name],
                        body_parts: [(n)-[:AFFECTS_BODY_PART]->(b:BodyPart) | b.name]
                      }
                      WHEN 'Symptom' THEN {
                        diseases: [(n)<-[:HAS_SYMPTOM]-(d:Disease) | d.name]
                      }
                      WHEN 'Medication' THEN {
                        category: n.category, side_effects: n.side_effects,
                        contraindications: n.contraindications,
                        treats: [(n)-[:TREATS_DISEASE]->(d:Disease) | d.name]
                      }
                      ELSE {}
                    END AS facts
                    """,
                    name=name,
                ).single()
                if record is None:
                    return {}
                return {"type": record["type"], "facts": record["facts"] or {}}
        except Exception as exc:  # noqa: BLE001
            logger.error("expand_entity(%s) 失败: %s", name, exc)
            return {}

    def expand_multihop(
        self, name: str, max_hops: int | None = None,
        limit_per_hop: int | None = None, total_limit: int | None = None,
        entity_type: str | None = None,
    ) -> dict:
        """Bounded breadth-first expansion, preserving actual relationship direction.

        Node identity is elementId, not its display name. Limits apply to the
        entire frontier at each depth; explicit arguments cannot raise config caps.
        """
        def bounded(value, cap):
            if value is not None and (not isinstance(value, int) or value < 1):
                raise ValueError("graph traversal limits must be positive integers")
            return min(value if value is not None else cap, cap)

        hops = bounded(max_hops, min(settings.kg_max_hops, 3))
        per_hop = bounded(limit_per_hop, settings.kg_multihop_limit_per_hop)
        total = bounded(total_limit, settings.kg_multihop_total_limit)
        result = {
            "root_entity": name, "configured_max_hops": settings.kg_max_hops,
            "applied_hops": hops, "executed_hops": 0, "total_triples": 0,
            "hop1_count": 0, "hop2_count": 0, "hop3_count": 0,
            "triples": [], "visited_entities": [name],
            "provenance": {"source_platform": "unknown", "medical_reviewed": False,
                           "evidence_grade": "unverified"},
        }
        visited_ids, edge_ids, visited_names = set(), set(), {name}
        frontier = []
        try:
            from neo4j import READ_ACCESS

            with get_driver().session(default_access_mode=READ_ACCESS) as session:
                for hop in range(1, hops + 1):
                    remaining = total - len(result["triples"])
                    if remaining <= 0:
                        break
                    records = session.run(
                        """
                        MATCH (mid)-[r]-(target)
                        WHERE (($first AND mid.name = $name
                                AND ($entity_type IS NULL OR $entity_type IN labels(mid)))
                            OR (NOT $first AND elementId(mid) IN $frontier))
                          AND NOT elementId(target) IN $visited
                        WITH mid, r, target, startNode(r) AS src, endNode(r) AS tgt
                        RETURN elementId(mid) AS mid_id,
                               elementId(target) AS target_id,
                               coalesce(target.name, target.content) AS target_name,
                               elementId(r) AS edge_id,
                               labels(src)[0] AS src_type,
                               coalesce(src.name, src.content) AS src_name,
                               type(r) AS rel, labels(tgt)[0] AS tgt_type,
                               coalesce(tgt.name, tgt.content) AS tgt_name,
                               coalesce(r.provenance_records, r.provenance_json) AS provenance_json,
                               coalesce(src.provenance_records, src.provenance_json) AS source_provenance_json,
                               coalesce(tgt.provenance_records, tgt.provenance_json) AS target_provenance_json
                        ORDER BY src_type, src_name, rel, tgt_type, tgt_name, edge_id
                        LIMIT $limit
                        """,
                        first=hop == 1, name=name, entity_type=entity_type,
                        frontier=frontier, visited=sorted(visited_ids),
                        limit=min(per_hop, remaining),
                    )
                    next_frontier = set()
                    for record in records:
                        edge_id = record["edge_id"]
                        if edge_id in edge_ids:
                            continue
                        edge_ids.add(edge_id)
                        visited_ids.add(record["mid_id"])
                        visited_names.add(record["target_name"])
                        next_frontier.add(record["target_id"])
                        result["triples"].append({
                            "subject": record["src_name"], "subject_type": record["src_type"],
                            "predicate": record["rel"], "object": record["tgt_name"],
                            "object_type": record["tgt_type"], "hop": hop,
                            # A node's source is not evidence for every incident
                            # edge. Only recorded relationship sources support it.
                            "provenance": decode_provenance(record.get("provenance_json")),
                        })
                        result[f"hop{hop}_count"] += 1
                    result["executed_hops"] = hop
                    visited_ids.update(next_frontier)
                    frontier = sorted(next_frontier)
                    if not frontier:
                        break
            result["total_triples"] = len(result["triples"])
            result["visited_entities"] = sorted(n for n in visited_names if n)
            return result
        except Exception as exc:
            logger.error("expand_multihop(%s) failed: %s", name, exc)
            # Do not expose partial traversal as a complete successful result.
            result.update(triples=[], total_triples=0, hop1_count=0, hop2_count=0,
                          hop3_count=0, error=str(exc))
            return result

    # ---------- 统计 ----------
    def get_stats(self) -> dict:
        try:
            with get_driver().session() as session:
                total_nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
                total_rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
                node_stats = {
                    r["l"]: r["c"] for r in session.run(
                        "MATCH (n) UNWIND labels(n) AS l RETURN l, count(*) AS c"
                    )
                }
            return {
                "total_nodes": total_nodes,
                "total_relations": total_rels,
                "node_stats": node_stats,
                "graph_density": round(total_rels / total_nodes, 2) if total_nodes else 0.0,
            }
        except Exception as exc:  # noqa: BLE001
            logger.error("get_stats 失败: %s", exc)
            return {"total_nodes": 0, "total_relations": 0, "node_stats": {}, "graph_density": 0.0}
