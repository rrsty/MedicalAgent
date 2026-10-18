"""用药建议 Agent：Neo4j 候选药 + LLM 安全过滤 + 诚实兜底。

安全红线：相互作用类问题不编造（本系统未内置权威相互作用库，由工具层诚实声明）；
剂量问题一律引导遵医嘱。
"""
from src.agents.state import AgentState
from src.common.llm import get_llm
from src.common.utils import parse_llm_json
from src.common.logger import setup_logger
from src.knowledge_graph.graph_queries import GraphQueries

logger = setup_logger("agent.medication", "agents.log")

MEDICATION_REMINDER = "以上药物信息仅供参考，务必遵医嘱用药，具体用法用量请咨询医生或药师。"


class MedicationAdvisor:
    def __init__(self):
        self.queries = GraphQueries()

    def _kg_meds(self, symptoms: list[str], query: str) -> list[dict]:
        """从图谱找候选药：query 命中疾病 → 疾病常用药；症状 → 疾病 → 药物。"""
        meds: dict[str, dict] = {}
        disease_names = [query.strip()] + self._match_diseases(symptoms)
        for disease in disease_names:
            info = self.queries.get_disease_info(disease)
            for name in info.get("medications", []):
                if name not in meds:
                    meds[name] = {"name": name, "for_disease": disease}
        return list(meds.values())

    def _match_diseases(self, symptoms: list[str]) -> list[str]:
        diseases: list[str] = []
        for symptom in symptoms:
            for d in self.queries.get_diseases_for_symptom(symptom, limit=3):
                if d not in diseases:
                    diseases.append(d)
        return diseases

    def _llm_filter(self, query: str, candidates: list[dict]) -> list[dict]:
        """LLM 过滤与补充说明（用途/副作用），失败时返回精简的图谱候选。"""
        if not candidates:
            return []
        candidate_text = "、".join(m["name"] for m in candidates[:10])
        try:
            resp = get_llm(temperature=0.1).invoke([
                (
                    "system",
                    "你是药学顾问。从候选药物中筛选与患者情况最相关的药物（最多5个），"
                    "并简要说明用途与常见副作用。只基于候选列表，不要添加列表外的药物。"
                    '输出 JSON: {"medications": [{"name":"", "purpose":"", "side_effects":""}]}',
                ),
                ("human", f"患者情况：{query}\n候选药物：{candidate_text}"),
            ])
            data = parse_llm_json(resp.content) or {}
            meds = []
            for item in data.get("medications", [])[:5]:
                name = str(item.get("name", "")).strip()
                if not name:
                    continue
                meds.append({
                    "name": name,
                    "purpose": str(item.get("purpose", "")).strip(),
                    "side_effects": str(item.get("side_effects", "")).strip(),
                    "reminder": MEDICATION_REMINDER,
                })
            if meds:
                return meds
        except Exception as exc:  # noqa: BLE001
            logger.warning("用药建议 LLM 过滤失败，退回图谱候选: %s", exc)

        return [
            {
                "name": m["name"],
                "purpose": f"用于{m.get('for_disease', '相关疾病')}的治疗",
                "side_effects": "",
                "reminder": MEDICATION_REMINDER,
            }
            for m in candidates[:5]
        ]

    def __call__(self, state: AgentState) -> dict:
        query = state.get("query", "")
        symptoms = state.get("symptoms", [])

        candidates = self._kg_meds(symptoms, query)
        medications = self._llm_filter(query, candidates)

        logger.info("用药建议: %d 个候选 → %d 个推荐", len(candidates), len(medications))
        return {"medications": medications}
