"""知识图谱 Schema —— R1 单一事实来源（全项目唯一允许定义节点/关系类型处）。

8 种节点类型、12 种关系类型；其他模块一律 `from src.knowledge_graph.schema import
NodeType, RelationType` 引用，禁止字符串字面量重复定义（TDD 第 0 章 R1）。

R6 单向关系原则的两个例外：TREATS_DISEASE / HANDLES_DISEASE 与其正向关系
（TREATED_BY_MEDICATION / BELONGS_TO_DEPARTMENT）成对真实建边，简化反向查询；
其余关系均为单向，反查用 Cypher 反向遍历。
"""
from enum import Enum


class NodeType(str, Enum):
    """节点类型（8 种）。"""

    DISEASE = "Disease"
    SYMPTOM = "Symptom"
    MEDICATION = "Medication"
    DEPARTMENT = "Department"
    BODY_PART = "BodyPart"
    EXAMINATION = "Examination"
    FOOD = "Food"
    PRECAUTION = "Precaution"


class RelationType(str, Enum):
    """关系类型（12 种）。"""

    HAS_SYMPTOM = "HAS_SYMPTOM"                          # Disease → Symptom
    MAY_INDICATE = "MAY_INDICATE"                        # Symptom → Disease
    BELONGS_TO_DEPARTMENT = "BELONGS_TO_DEPARTMENT"      # Disease → Department
    HANDLES_DISEASE = "HANDLES_DISEASE"                  # Department → Disease（反向）
    TREATED_BY_MEDICATION = "TREATED_BY_MEDICATION"      # Disease → Medication
    TREATS_DISEASE = "TREATS_DISEASE"                    # Medication → Disease（反向）
    HAS_SIDE_EFFECT = "HAS_SIDE_EFFECT"                  # Medication → Symptom
    INTERACTS_WITH = "INTERACTS_WITH"                    # Medication → Medication
    NEEDS_EXAMINATION = "NEEDS_EXAMINATION"              # Disease → Examination
    AFFECTS_BODY_PART = "AFFECTS_BODY_PART"              # Disease → BodyPart
    RECOMMENDS_FOOD = "RECOMMENDS_FOOD"                  # Disease → Food
    HAS_PRECAUTION = "HAS_PRECAUTION"                    # Disease → Precaution


# 每种节点的属性模板（graph_builder / schema_manager / 检查器共用）
NODE_PROPERTIES: dict[str, list[str]] = {
    NodeType.DISEASE: ["name", "description", "icd_code"],
    NodeType.SYMPTOM: ["name", "severity"],
    NodeType.MEDICATION: ["name", "category", "side_effects", "contraindications"],
    NodeType.DEPARTMENT: ["name", "floor"],
    NodeType.BODY_PART: ["name"],
    NodeType.EXAMINATION: ["name", "type"],
    NodeType.FOOD: ["name", "type"],
    NodeType.PRECAUTION: ["content"],
}

# 每种关系：起点类型 → 终点类型（check_contracts / text2cypher 的 Schema 文本共用）
RELATION_ENDPOINTS: dict[str, tuple[str, str]] = {
    RelationType.HAS_SYMPTOM: (NodeType.DISEASE, NodeType.SYMPTOM),
    RelationType.MAY_INDICATE: (NodeType.SYMPTOM, NodeType.DISEASE),
    RelationType.BELONGS_TO_DEPARTMENT: (NodeType.DISEASE, NodeType.DEPARTMENT),
    RelationType.HANDLES_DISEASE: (NodeType.DEPARTMENT, NodeType.DISEASE),
    RelationType.TREATED_BY_MEDICATION: (NodeType.DISEASE, NodeType.MEDICATION),
    RelationType.TREATS_DISEASE: (NodeType.MEDICATION, NodeType.DISEASE),
    RelationType.HAS_SIDE_EFFECT: (NodeType.MEDICATION, NodeType.SYMPTOM),
    RelationType.INTERACTS_WITH: (NodeType.MEDICATION, NodeType.MEDICATION),
    RelationType.NEEDS_EXAMINATION: (NodeType.DISEASE, NodeType.EXAMINATION),
    RelationType.AFFECTS_BODY_PART: (NodeType.DISEASE, NodeType.BODY_PART),
    RelationType.RECOMMENDS_FOOD: (NodeType.DISEASE, NodeType.FOOD),
    RelationType.HAS_PRECAUTION: (NodeType.DISEASE, NodeType.PRECAUTION),
}

# 与正向关系成对真实建边的反向关系（R6 例外）
REVERSE_RELATIONS: dict[str, str] = {
    RelationType.TREATED_BY_MEDICATION: RelationType.TREATS_DISEASE,
    RelationType.BELONGS_TO_DEPARTMENT: RelationType.HANDLES_DISEASE,
}

# 关系序列化时的端点字段名（relations.json 契约：{"type": ..., "data": {...}}）
RELATION_DATA_FIELDS: dict[str, tuple[str, str]] = {
    RelationType.HAS_SYMPTOM: ("disease_name", "symptom_name"),
    RelationType.MAY_INDICATE: ("symptom_name", "disease_name"),
    RelationType.BELONGS_TO_DEPARTMENT: ("disease_name", "department_name"),
    RelationType.HANDLES_DISEASE: ("department_name", "disease_name"),
    RelationType.TREATED_BY_MEDICATION: ("disease_name", "drug_name"),
    RelationType.TREATS_DISEASE: ("drug_name", "disease_name"),
    RelationType.HAS_SIDE_EFFECT: ("drug_name", "symptom_name"),
    RelationType.INTERACTS_WITH: ("drug_name", "interacting_drug_name"),
    RelationType.NEEDS_EXAMINATION: ("disease_name", "examination_name"),
    RelationType.AFFECTS_BODY_PART: ("disease_name", "body_part_name"),
    RelationType.RECOMMENDS_FOOD: ("disease_name", "food_name"),
    RelationType.HAS_PRECAUTION: ("disease_name", "precaution_content"),
}


def generate_llm_schema_text() -> str:
    """生成给 LLM（Text2Cypher）看的 Schema 文本：节点类型 + 属性 + 关系方向。"""
    lines = ["节点类型与属性："]
    for node_type, props in NODE_PROPERTIES.items():
        lines.append(f"  - :{node_type.value} {{{', '.join(props)}}}")
    lines.append("")
    lines.append("关系类型（起点 → 关系 → 终点）：")
    for rel_type, (start, end) in RELATION_ENDPOINTS.items():
        lines.append(f"  - (:{start.value})-[:{rel_type.value}]->(:{end.value})")
    lines.append("")
    lines.append("约束：所有节点按 name 唯一；属性名统一用英文（name/description/...）。")
    return "\n".join(lines)
