"""ReAct 工具集：把已测能力包成 @tool（docstring 即模型判断"何时调用"的依据）。

铁律：
- 一律返回 str（ToolNode 需要可序列化文本）；
- 内部 try/except 兜底，查不到返回「未找到」、异常返回「查询失败：…」，绝不抛异常；
- 对图谱可能缺失的属性用 .get(...) 容错；
- 相互作用查不到时诚实声明，绝不编造（医疗安全红线）。
"""
from langchain_core.tools import tool

from src.knowledge_graph.graph_queries import GraphQueries
from src.knowledge_graph.text2cypher import natural_language_query
from src.vector_store.search import get_shared_retriever

_INTERACTION_NOTICE = (
    "本系统未内置完整药物相互作用库，联用风险请以药师/权威数据库为准，切勿自行合用。"
)


@tool
def search_knowledge_graph(entity: str) -> str:
    """根据疾病或症状名，查询知识图谱，返回其科室/症状/药物/检查摘要。
    当用户问到某疾病的科室、症状、用药、检查，或某症状对应哪些疾病时使用。"""
    try:
        queries = GraphQueries()
        info = queries.get_disease_info(entity)
        if info:
            parts = [f"{info['name']}"]
            if info.get("description"):
                parts.append(f"描述：{info['description']}")
            if info.get("symptoms"):
                parts.append("常见症状：" + "、".join(info["symptoms"][:10]))
            if info.get("departments"):
                parts.append("就诊科室：" + "、".join(info["departments"]))
            if info.get("medications"):
                parts.append("常用药物：" + "、".join(info["medications"][:10]))
            if info.get("examinations"):
                parts.append("相关检查：" + "、".join(info["examinations"][:8]))
            return "\n".join(parts)

        # 疾病查不到 → 按症状反查疾病
        diseases = queries.get_diseases_for_symptom(entity, limit=8)
        if diseases:
            return f"「{entity}」可能提示的疾病：{'、'.join(diseases)}"
        return f"未找到「{entity}」的相关信息"
    except Exception as exc:  # noqa: BLE001
        return f"查询失败：{exc}"


@tool
def semantic_search(query: str) -> str:
    """语义搜索医学知识库，返回与问题最相关的实体（名称/类型/相似度）。
    当需要模糊匹配、口语表述或不确定实体名称时使用。"""
    try:
        results = get_shared_retriever().search(query, top_k=5)
        if not results:
            return f"未找到与「{query}」语义相关的知识"
        return "\n".join(
            f"- {r['name']}（{r.get('type', '')}, 相似度{r['score']}）" for r in results
        )
    except Exception as exc:  # noqa: BLE001
        return f"查询失败：{exc}"


@tool
def get_medication_info(name: str) -> str:
    """查询指定药物的详细信息：类别、副作用、禁忌、适应疾病、相互作用。
    当用户问到某个具体药物的副作用、禁忌或是否适合自己时使用。"""
    try:
        queries = GraphQueries()
        info = queries.get_medication_info(name)
        if not info:
            return f"未找到药物「{name}」的信息。请注意：请勿自行用药，务必遵医嘱。"

        parts = [f"药物：{info['name']}"]
        if info.get("category"):
            parts.append(f"类别：{info['category']}")
        if info.get("side_effects"):
            effects = info["side_effects"]
            if isinstance(effects, str):
                effects = [effects]
            parts.append("常见副作用：" + "、".join(effects))
        if info.get("contraindications"):
            cons = info["contraindications"]
            if isinstance(cons, str):
                cons = [cons]
            parts.append("禁忌：" + "、".join(cons))
        if info.get("treats"):
            parts.append("用于治疗：" + "、".join(info["treats"]))

        interactions = queries.get_medication_interactions(name)
        if interactions:
            parts.append("已知相互作用药物：" + "、".join(interactions))
        else:
            # 诚实声明：不编造相互作用（医疗安全红线）
            parts.append(f"相互作用：{_INTERACTION_NOTICE}")
        parts.append("提醒：具体用法用量请遵医嘱。")
        return "\n".join(parts)
    except Exception as exc:  # noqa: BLE001
        return f"查询失败：{exc}"


@tool
def natural_language_graph_query(query: str) -> str:
    """把自然语言问题转换为只读 Cypher 查询知识图谱，支持多跳、聚合、条件过滤等灵活查询。
    当前面的工具无法满足、需要精确的多跳关系查询（如"治疗X的药有哪些副作用"）时使用。"""
    try:
        result = natural_language_query(query)
        if not result.get("success"):
            return f"查询失败：{result.get('error', '未知错误')}"
        columns, rows = result["columns"], result["rows"]
        if not rows:
            return "图谱中没有查询到相关数据"
        lines = [f"查询结果（{len(rows)} 行）："]
        for row in rows:
            lines.append(" | ".join(str(v) for v in row))
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        return f"查询失败：{exc}"


def get_react_tools() -> list:
    """ReAct 层工具集（settings.enable_react_layer 开启时使用）。"""
    return [
        search_knowledge_graph,
        semantic_search,
        get_medication_info,
        natural_language_graph_query,
    ]
