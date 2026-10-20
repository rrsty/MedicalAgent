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
        """从图谱找候选药：query 命中疾病 → 疾病常用药；无疾病时由症状反查。"""
        meds: dict[str, dict] = {}

        # 优先提取问句中明确提及的具体疾病（如"我有糖尿病，我应该吃什么药"中的"糖尿病"）
        matched_disease = self.queries.match_disease_in_text(query)
        if matched_disease:
            # 明确提到了疾病：以该疾病为唯一锚点，切断上文无关症状的混淆
            disease_names = [matched_disease]
        else:
            disease_names = self._match_diseases(symptoms)

        for disease in disease_names:
            if not disease:
                continue
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

    def _llm_filter(self, query: str, candidates: list[dict], is_pregnant: bool = False) -> list[dict]:
        """LLM 过滤与补充说明（用途/副作用），失败时返回精简的图谱候选。"""
        if not candidates:
            return []
        candidate_text = "、".join(m["name"] for m in candidates[:10])

        reminder = (
            "🤰 特别警示：患者处于孕期/特殊生理阶段，妊娠期用药极具风险，未经产科及专科医生面诊评估处方，切勿自行购买和服用任何药物！"
            if is_pregnant else MEDICATION_REMINDER
        )

        sys_prompt = (
            "你是资深临床药师。重要提示：患者处于【孕期/特殊妊娠阶段】！\n"
            "规则：\n"
            "1. 绝大多数口服药物（尤其妊娠早期）对胎儿器官发育有致畸或不良影响；\n"
            "2. 严禁随意推荐常规药物，必须严格审查候选药物的妊娠期安全性，说明孕期风险与禁忌；\n"
            "3. 输出最多3个最审慎的方案，首要强调必须遵产科和专科医嘱。\n"
            '输出 JSON: {"medications": [{"name":"", "purpose":"", "side_effects":""}]}'
            if is_pregnant else
            "你是药学顾问。从候选药物中筛选与患者情况最相关的药物（最多5个），"
            "并简要说明用途与常见副作用。只基于候选列表，不要添加列表外的药物。"
            '输出 JSON: {"medications": [{"name":"", "purpose":"", "side_effects":""}]}'
        )

        try:
            resp = get_llm(temperature=0.1).invoke([
                ("system", sys_prompt),
                ("human", f"患者情况：{query}\n候选药物：{candidate_text}"),
            ])
            data = parse_llm_json(resp.content) or {}
            meds = []
            max_count = 3 if is_pregnant else 5
            for item in data.get("medications", [])[:max_count]:
                name = str(item.get("name", "")).strip()
                if not name:
                    continue
                meds.append({
                    "name": name,
                    "purpose": str(item.get("purpose", "")).strip(),
                    "side_effects": str(item.get("side_effects", "")).strip(),
                    "reminder": reminder,
                })
            if meds:
                return meds
        except Exception as exc:  # noqa: BLE001
            logger.warning("用药建议 LLM 过滤失败，退回图谱候选: %s", exc)

        max_count = 3 if is_pregnant else 5
        return [
            {
                "name": m["name"],
                "purpose": f"用于{m.get('for_disease', '相关疾病')}的治疗",
                "side_effects": "",
                "reminder": reminder,
            }
            for m in candidates[:max_count]
        ]

    def __call__(self, state: AgentState) -> dict:
        query = state.get("query", "")
        symptoms = state.get("symptoms", [])
        history = state.get("history", [])

        # 检查多轮对话是否涉及孕妇等特殊人群
        is_pregnant = any(
            kw in query for kw in ("孕妇", "怀孕", "孕期", "孕周", "产后", "哺乳期", "备孕")
        ) or any(
            any(kw in str(t.get("content", "")) for kw in ("孕妇", "怀孕", "孕期", "孕周"))
            for t in history[-4:]
        )

        candidates = self._kg_meds(symptoms, query)
        medications = self._llm_filter(query, candidates, is_pregnant=is_pregnant)

        logger.info("用药建议: %d 个候选 → %d 个推荐 (is_pregnant=%s)", len(candidates), len(medications), is_pregnant)
        return {"medications": medications}
