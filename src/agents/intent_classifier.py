"""意图识别 Agent：LLM 分类 + Few-shot，输出 intent + confidence。"""
from config.settings import settings
from src.agents.safety_checker import EMERGENCY_SYMPTOM_KEYWORDS
from src.agents.state import AgentState, IntentType
from src.common.context_window import build_chat_history
from src.common.llm import get_llm
from src.common.utils import parse_llm_json
from src.common.logger import setup_logger

logger = setup_logger("agent.intent", "agents.log")

SYSTEM_PROMPT = """你是一个专业的医疗问诊意图识别专家。请判断用户的问题属于以下哪种意图：
- appointment：看病挂号。询问应该看什么科、挂什么号、就医科室推荐。
- medication：用药建议。询问具体疾病吃什么药、具体用药方法、用法用量、用药配伍禁忌、药物副作用及不良反应处理等临床用药指导，以及在用药咨询上下文中补充自身特殊生理状态（如"我是孕妇"、"处于哺乳期"、"有药物过敏"）或追问能否吃药。
- knowledge：医学与药物知识。询问疾病概念/定义、病因机制、预防保健、检查化验指标，以及药物/药品科普、定义、分类、药理机制，以及陈述慢性病史、病程阶段（如"怀孕前就有的糖尿病，目前孕5周"、"我有高血压病史"）。
- emergency：急症。胸痛、呼吸困难、意识模糊、昏迷、抽搐、中毒、大出血等需要立即紧急就医的危险症状。
- general：非医疗问题。日常问候、闲聊、天气、诗歌、笑话、或对话历史问答（如"刚才说了什么"）等非医疗专业问答。

只输出 JSON：{"intent": "...", "confidence": 0.0~1.0}

示例：
1. "我最近头痛，应该挂什么科？" → {"intent": "appointment", "confidence": 0.95}
2. "高血压吃什么药好？" → {"intent": "medication", "confidence": 0.95}
3. "什么是糖尿病？" → {"intent": "knowledge", "confidence": 0.95}
4. "司美格鲁肽是什么？" → {"intent": "knowledge", "confidence": 0.95}
5. "司美格鲁肽" → {"intent": "knowledge", "confidence": 0.95}
6. "二甲双胍是管什么的？" → {"intent": "knowledge", "confidence": 0.95}
7. "阿司匹林有什么副作用？" → {"intent": "medication", "confidence": 0.9}
8. "奥美拉唑饭前吃还是饭后吃？" → {"intent": "medication", "confidence": 0.95}
9. "我说了我是孕妇" → {"intent": "medication", "confidence": 0.95}
10. "怀孕前就有的糖尿病，目前孕5周" → {"intent": "knowledge", "confidence": 0.95}
11. "我说的是我有糖尿病" → {"intent": "medication", "confidence": 0.95}
12. "高血压是怎么引起的？" → {"intent": "knowledge", "confidence": 0.92}
13. "我突然胸痛、呼吸困难！" → {"intent": "emergency", "confidence": 0.98}
14. "今天天气怎么样？" → {"intent": "general", "confidence": 0.9}
15. "你好" → {"intent": "general", "confidence": 0.9}"""

_VALID_INTENTS = {i.value for i in IntentType}

_DRUG_SUFFIXES = (
    "格鲁肽", "双胍", "地平", "沙坦", "拉唑", "他汀", "普利", "洛尔",
    "西林", "霉素", "头孢", "列净", "列汀", "匹林", "布洛芬", "对乙酰氨基酚",
    "阿司匹林", "素", "多肽", "单抗", "胶囊", "片", "分散片",
    "泡腾片", "注射液", "软膏", "滴眼液", "疫苗", "栓剂",
)
_CONCEPT_KEYWORDS = (
    "是什么", "是啥", "什么是", "怎么回事", "有什么用", "作用是什么",
    "是干什么", "是管什么", "有什么功效", "有什么危害", "含义", "定义",
)
_SPECIAL_POPULATION_KEYWORDS = (
    "孕妇", "怀孕", "孕期", "孕周", "产后", "哺乳期", "备孕", "婴儿", "幼儿", "老人", "高龄",
)
_CHRONIC_DISEASE_KEYWORDS = (
    "糖尿病", "高血压", "冠心病", "心脏病", "哮喘", "胃炎", "痛风", "贫血", "肾病", "肝病",
    "病史", "确诊", "前就有的", "前就有",
)
_PURE_CHAT_KEYWORDS = (
    "你好", "您好", "在吗", "早安", "晚安", "再见", "拜拜", "谢谢", "感谢",
    "今天天气", "明天天气", "天气怎么样", "天气如何", "下雨", "讲个笑话",
    "讲笑话", "写一首诗", "写首诗", "作诗", "你是谁", "你叫什么", "你能做什么",
    "哈哈", "好的", "收到", "辛苦了", "没事了", "打扰了",
)
_HISTORY_TEST_KEYWORDS = ("刚才", "上一轮", "前面说", "历史", "之前说")

_BODY_PARTS = (
    "头", "脑", "额", "太阳穴", "颈", "脖子", "咽", "喉", "肩", "胸", "心", "肺",
    "乳房", "乳腺", "胃", "肠", "腹", "肚子", "肚脐", "腰", "背", "脊椎", "颈椎",
    "腰椎", "骨", "关节", "膝盖", "腿", "脚", "脚踝", "手臂", "胳膊", "手腕", "手指",
    "眼", "耳", "鼻", "牙", "齿", "龈", "舌", "嘴", "唇", "皮", "肤", "肝", "胆",
    "脾", "胰", "肾", "膀胱", "尿道", "肛门", "血管", "静脉", "淋巴", "甲状腺",
    "子宫", "卵巢", "阴道", "前列腺", "睾丸",
)
_SYMPTOM_SENSATIONS = (
    "痛", "疼", "酸", "胀", "麻", "痒", "晕", "眩", "吐", "恶心", "呕", "咳",
    "痰", "烧", "热", "冷", "寒", "喘", "闷", "憋", "肿", "硬", "虚", "乏",
    "累", "无力", "疲劳", "悸", "慌", "抽", "痉挛", "炎", "斑", "疱", "疹", "痘",
    "血", "脓", "溃烂", "出汗", "盗汗", "发冷", "便秘", "拉肚子", "腹泻", "失眠",
    "做噩梦", "食欲不振", "消瘦", "脱发", "耳鸣", "视力模糊", "口苦", "口臭",
    "透风", "怕风", "畏寒", "发凉", "发热", "抽筋", "僵硬", "麻木", "咔咔", "弹响",
)
_MEDICAL_TESTS_INDICATORS = (
    "ct", "b超", "核磁", "mri", "心电图", "胃镜", "肠镜", "喉镜", "血常规", "尿常规",
    "便常规", "化验", "生化", "体检", "检查", "指标", "转氨酶", "肌酐", "尿酸", "血糖",
    "血压", "血脂", "胆固醇", "甘油三酯", "白细胞", "红细胞", "血小板", "抗体", "抗原",
    "阳性", "阴性", "偏高", "偏低", "超标", "结节", "囊肿", "增生", "息肉", "积液",
    "结石", "钙化", "阴影", "占位", "浸润",
)
_CLINICAL_INQUIRY_PATTERNS = (
    "严重吗", "厉害吗", "会死吗", "会自愈吗", "能自愈吗", "会传染吗", "能根治吗",
    "能治好吗", "怎么调理", "怎么缓解", "吃什么能好", "吃什么好", "后遗症",
    "有必要做吗", "要不要做", "复发", "病因", "诱因", "危害", "禁忌", "早期症状",
    "前兆", "正常吗", "要紧吗",
)


_APPOINTMENT_KEYWORDS = (
    "挂号", "科室", "门诊", "挂哪", "看哪个科", "看什么科", "去哪个科", "看哪科", "就诊", "找哪个医生",
)


def _load_known_entities() -> tuple[set[str], set[str]]:
    try:
        from pathlib import Path
        import json
        path = Path(__file__).resolve().parents[2] / "data" / "knowledge_graph" / "entities.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            diseases = {
                e["name"] for e in data.get("entities", [])
                if e.get("type") == "Disease" and len(e.get("name", "")) >= 2
            }
            medications = {
                e["name"] for e in data.get("entities", [])
                if e.get("type") == "Medication" and len(e.get("name", "")) >= 2
            }
            return diseases, medications
    except Exception:
        pass
    return set(), set()


_KNOWN_DISEASES, _KNOWN_MEDICATIONS = _load_known_entities()


def is_potential_medical_query(query: str) -> bool:
    """终审复检：判断一个问题是否包含明确的身体症状、化验指标、用药或疾病特征（非纯闲聊）。"""
    q = query.strip().lower()
    if not q:
        return False

    # 1. 历史追溯测试短路（如"刚才我说的是什么症状？"）
    if any(hw in q for hw in _HISTORY_TEST_KEYWORDS):
        return False

    # 2. 纯闲聊白名单短路（但排除"你好我头疼"这类以问候开头的实际医疗问题）
    is_pure_greeting = any(q == w or q.startswith(w) for w in _PURE_CHAT_KEYWORDS)
    if is_pure_greeting and not any(ss in q for ss in _SYMPTOM_SENSATIONS) and not any(bp in q for bp in _BODY_PARTS):
        return False

    # 3. 医学特征矩阵
    has_body = any(bp in q for bp in _BODY_PARTS)
    has_symptom = any(ss in q for ss in _SYMPTOM_SENSATIONS)
    has_test = any(ti in q for ti in _MEDICAL_TESTS_INDICATORS)
    has_clinical = any(ci in q for ci in _CLINICAL_INQUIRY_PATTERNS)
    has_drug = any(ds in q for ds in _DRUG_SUFFIXES) or any(km in q for km in _KNOWN_MEDICATIONS)
    has_chronic = any(cd in q for cd in _CHRONIC_DISEASE_KEYWORDS) or any(kd in q for kd in _KNOWN_DISEASES)
    has_concept = any(ck in q for ck in _CONCEPT_KEYWORDS)
    has_special_pop = any(sp in q for sp in _SPECIAL_POPULATION_KEYWORDS)
    has_appointment = any(ak in q for ak in _APPOINTMENT_KEYWORDS)

    if (
        (has_body and (has_symptom or has_concept or has_appointment))
        or has_test
        or has_clinical
        or has_drug
        or has_chronic
        or has_special_pop
        or has_appointment
    ):
        return True
    if has_symptom and len(q) >= 3:
        return True
    return False


def intent_classifier(state: AgentState) -> dict:
    """意图识别节点（LLM 分类 + 医疗复检门兜底，解析失败降级 general）。"""
    query = state.get("query", "")

    # 急症关键词先行短路（R1 词表唯一来源 safety_checker；安全优先：LLM 误判也有兜底）
    if any(kw in query for kw in EMERGENCY_SYMPTOM_KEYWORDS):
        return {"intent": IntentType.EMERGENCY.value, "confidence": 0.99}

    try:
        messages = [("system", SYSTEM_PROMPT)]
        # 统一预算并保留长回答的开头与结论，避免只截前缀丢掉关键结论。
        messages.extend(build_chat_history(
            state.get("history", []),
            max_messages=4,
            per_message_chars=300,
            total_chars=1200,
        ))
        messages.append(("human", query))

        resp = get_llm(temperature=0.1).invoke(messages)
        data = parse_llm_json(resp.content) or {}
        intent = str(data.get("intent", "general")).lower()
        if intent not in _VALID_INTENTS:
            intent = IntentType.GENERAL.value
        confidence = float(data.get("confidence", 0.5))
    except Exception as exc:  # noqa: BLE001
        logger.warning("意图识别 LLM 失败，降级 general: %s", exc)
        intent, confidence = IntentType.GENERAL.value, 0.3

    # 防御性兜底：若被判为 general（或 LLM 异常降级），启动医疗二次复检门禁
    if intent == IntentType.GENERAL.value:
        q_lower = query.strip().lower()
        is_history_test = any(hw in q_lower for hw in _HISTORY_TEST_KEYWORDS)
        is_pure_chat = (
            any(q_lower == w or q_lower.startswith(w) for w in _PURE_CHAT_KEYWORDS)
            and not any(ss in q_lower for ss in _SYMPTOM_SENSATIONS)
            and not any(bp in q_lower for bp in _BODY_PARTS)
        )

        should_recover = False
        if not is_history_test and not is_pure_chat:
            # 条件一：命中全量医学特征网（器官/感觉/化验/药物/慢病/特殊人群）
            # 条件二：或者模型置信度偏低（confidence <= 0.75 且非纯闲聊，模型犹豫不决）
            if is_potential_medical_query(query) or (confidence <= 0.75 and len(query) >= 4):
                should_recover = True

        if should_recover:
            # 智能重定向到最合适的分支（结合多轮上下文消解）
            hist_user_text = ""
            for turn in reversed(state.get("history", [])[-3:]):
                if turn.get("role") == "user":
                    hist_user_text = turn.get("content", "")
                    break

            if any(kw in query for kw in ("挂号", "挂什么科", "看什么科", "看哪个科", "挂哪", "看哪", "门诊", "科室")):
                logger.info("医疗复检门纠偏: general -> appointment | query=%s", query)
                intent = IntentType.APPOINTMENT.value
            elif any(kw in query for kw in ("药", "吃什么", "怎么用", "用法", "用量", "剂量", "副作用", "能吃吗", "禁忌")) or (
                any(kw in query for kw in _SPECIAL_POPULATION_KEYWORDS + _CHRONIC_DISEASE_KEYWORDS)
                and any(mkw in hist_user_text for mkw in ("药", "吃什么", "怎么用", "用法", "用量", "用哪个"))
            ):
                logger.info("医疗复检门纠偏: general -> medication | query=%s", query)
                intent = IntentType.MEDICATION.value
            else:
                logger.info("医疗复检门纠偏: general -> knowledge | query=%s", query)
                intent = IntentType.KNOWLEDGE.value
            confidence = max(confidence, 0.88)

    logger.info("意图识别: %s (confidence=%.2f) | query=%s", intent, confidence, query[:50])
    return {"intent": intent, "confidence": confidence}
