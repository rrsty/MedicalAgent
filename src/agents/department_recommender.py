"""科室推荐 Agent：Neo4j 图谱候选 → LLM 取舍 → 默认值 三级回退（结构化决策）。"""
from src.agents.state import AgentState
from src.common.llm import get_llm
from src.common.logger import setup_logger
from src.common.utils import parse_llm_json
from src.knowledge_graph.graph_queries import GraphQueries

logger = setup_logger("agent.department", "agents.log")

DEFAULT_DEPARTMENTS = ["全科医学科", "内科"]

SYSTEM_PROMPT = """你是一个医院导诊专家。根据患者症状从候选科室中推荐最合适的就诊科室。

要求：
1. 只从候选科室中选择，按优先级排序（最推荐的在前）
2. 最多推荐 3 个科室
3. 输出 JSON：{"departments": ["科室1", "科室2"]}
4. 候选科室都不合适时可以给出更合适的标准科室名称"""


class DepartmentRecommender:
    def __init__(self):
        self.queries = GraphQueries()

    def _tier_graph(self, symptoms: list[str]) -> dict[str, int]:
        """一级：Neo4j 图谱查询（症状 → 疾病 → 科室，按匹配疾病数打分）。"""
        scores: dict[str, int] = {}
        for symptom in symptoms:
            for dept, score in self.queries.get_departments_for_symptom(symptom).items():
                scores[dept] = scores.get(dept, 0) + score
        return scores

    def _tier_llm(self, symptoms: list[str], candidates: dict[str, int]) -> list[str]:
        """二级：LLM 取舍排序（图谱候选为空时，LLM 凭医学常识兜底）。"""
        candidate_text = "、".join(candidates.keys()) if candidates else "（图谱无候选）"
        try:
            resp = get_llm(temperature=0.1).invoke([
                ("system", SYSTEM_PROMPT),
                ("human", f"症状：{('、'.join(symptoms)) or '未描述'}\n候选科室：{candidate_text}"),
            ])
            data = parse_llm_json(resp.content) or {}
            departments = [str(d).strip() for d in data.get("departments", []) if str(d).strip()]
            if departments:
                return departments[:3]
        except Exception as exc:  # noqa: BLE001
            logger.warning("科室推荐 LLM 失败: %s", exc)
        return []

    def __call__(self, state: AgentState) -> dict:
        query = state.get("query", "").strip()
        symptoms = state.get("symptoms", [])

        # 疾病直查：query 里点名了图谱疾病（如「颈椎病应该挂什么科」）时，
        # 不需要症状提取，直接推荐该病的科室（零 LLM 成本）
        if not symptoms and query:
            disease = self.queries.match_disease_in_text(query)
            if disease:
                info = self.queries.get_disease_info(disease)
                if info.get("departments"):
                    departments = info["departments"][:3]
                    logger.info("科室推荐(疾病直查): %s ← %s", departments, disease)
                    return {
                        "departments": departments,
                        "department_sources": ["graph"] * len(departments),
                    }

        # 信息不足 → 置追问标记，照常走到答案融合输出追问话术
        if not symptoms:
            return {
                "departments": [],
                "department_sources": [],
                "needs_clarification": True,
                "clarification_question": "请问您有哪些具体症状或不适？（例如：头痛、发热、腹痛等）",
            }

        candidates = self._tier_graph(symptoms)
        source = "graph"
        departments = sorted(candidates, key=candidates.get, reverse=True)

        if not departments:
            departments = self._tier_llm(symptoms, candidates)
            source = "llm"
        if not departments:
            departments = list(DEFAULT_DEPARTMENTS)  # 三级：默认值
            source = "default"

        result = departments[:3]
        logger.info("科室推荐: %s (source=%s, 症状=%s)", result, source, symptoms)
        return {
            "departments": result,
            "department_sources": [source] * len(result),
        }
