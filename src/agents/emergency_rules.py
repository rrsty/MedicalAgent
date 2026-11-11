"""急症规则库与审计追踪引擎（Emergency Rule Registry & Audit Engine）。

设计原则：
1. 规则来源透明：候选指南名仅作为待核验资料，不代表已完成原文、版本或条款核验。
2. 否定词窗过滤：结合医学文本特征，过滤「无胸痛」「排除了心梗」「未见抽搐」等否定表达与误报场景。
3. 审计留痕：输出判定结果的同时输出规则命中依据与被压制（否定排除）记录。
4. 单一事实来源：系统所有急症关键词统一由注册中心导出。
"""
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EmergencyRule:
    rule_id: str
    name: str
    category: str
    keywords: list[str]
    guideline_source: str
    clinical_rationale: str
    applicable_conditions: str
    negation_patterns: list[str] = field(default_factory=list)
    false_positive_cases: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 演示规则注册库；候选资料尚未完成版本、原文条款与专业复核
# ---------------------------------------------------------------------------
EMERGENCY_RULES: list[EmergencyRule] = [
    EmergencyRule(
        rule_id="EMERG_CARDIO_001",
        name="急性胸痛 / 心血管危急症",
        category="心血管系统",
        keywords=["胸痛", "心前区疼痛", "压榨性胸痛", "心跳骤停", "心绞痛发作", "心梗", "心肌梗死"],
        guideline_source="待核验候选资料：《中国急性胸痛急诊分诊与诊治专家共识》《急性冠脉综合征急诊快速诊疗指南》",
        clinical_rationale="胸痛为急性心肌梗死、主动脉夹层、肺栓塞等致死性危急重症的主要首发表现，预检分诊中列为Ⅰ/Ⅱ级（濒危/危重），需即刻心电图并开放绿色通道。",
        applicable_conditions="突发剧烈胸痛、持续不缓解（>15-20分钟）、向左肩背或下颌放射、伴大汗或伴呼吸困难。",
        negation_patterns=[
            r"无胸痛", r"没有胸痛", r"未见胸痛", r"排除了?心梗", r"排除心绞痛",
            r"不是胸痛", r"胸痛已完全缓解", r"胸痛已愈", r"体检无胸痛",
        ],
        false_positive_cases=[
            "患者心电图正常，无胸痛症状",
            "医生已经帮我排除了心梗可能",
            "之前怀疑心绞痛，后来确诊只是肋软骨炎，无胸痛",
            "我没有胸痛，只是右小腿酸痛",
        ],
    ),
    EmergencyRule(
        rule_id="EMERG_RESP_001",
        name="急性呼吸困难 / 窒息",
        category="呼吸系统",
        keywords=["呼吸困难", "窒息", "气促不能平卧", "严重喘鸣", "喉头水肿"],
        guideline_source="待核验候选资料：《成人急性呼吸窘迫综合征急诊诊断与治疗指南》",
        clinical_rationale="突发严重呼吸困难或气道梗阻可在数分钟内导致严重低氧血症与呼吸衰竭，属于急诊Ⅰ级抢救范畴。",
        applicable_conditions="端坐呼吸、发绀、辅助呼吸肌参与、三凹征、指脉氧饱和度显著下降（<90%）或喉部喘鸣。",
        negation_patterns=[
            r"无呼吸困难", r"没有呼吸困难", r"未见呼吸困难", r"排除了?窒息",
            r"呼吸顺畅", r"呼吸平稳", r"不是窒息",
            r"(?:电视|电影|新闻|小说|科普|网上看到).*(?:窒息|呼吸困难)",
        ],
        false_positive_cases=[
            "体检肺部无异常，无呼吸困难",
            "平稳呼吸，没有呼吸困难的表现",
            "慢跑后稍微气喘，不是窒息",
            "电视里演主角窒息抢救",
        ],
    ),
    EmergencyRule(
        rule_id="EMERG_NEURO_001",
        name="急性神经危急症（意识障碍 / 抽搐 / 昏迷）",
        category="神经系统",
        keywords=["意识模糊", "昏迷", "抽搐", "高热惊厥", "晕厥", "意识丧失", "突发偏瘫"],
        guideline_source="待核验候选资料：《中国急性缺血性脑卒中诊治指南》《中国成人癫痫持续状态急诊诊治专家共识》《急诊昏迷患者诊断与鉴别诊断专家共识》",
        clinical_rationale="急性意识障碍或持续抽搐提示严重颅脑病变（脑卒中、颅内出血、癫痫持续状态）或重度脑缺氧，存在不可逆脑损伤和呼吸心跳抑制风险。",
        applicable_conditions="突发神志不清、言语不能、持续四肢强直抽搐（>5分钟）、呼之不应、一侧肢体突然无力。",
        negation_patterns=[
            r"神志清楚", r"神清", r"无意识模糊", r"未昏迷", r"没有抽搐",
            r"未见抽搐", r"排除了?脑出血", r"排除抽搐", r"意识清醒",
            r"(?:曾经|以前|年前|小时候|既往).*(?:高热惊厥|抽搐|昏迷).*(?:正常|好转|痊愈|完全正常|没有)",
            r"无高热惊厥",
        ],
        false_positive_cases=[
            "神志清楚，无意识模糊与抽搐",
            "5年前有过一次高热惊厥，现在完全正常",
            "体检报告说无意识障碍",
        ],
    ),
    EmergencyRule(
        rule_id="EMERG_HEMORRHAGE_001",
        name="急性活动性大出血",
        category="循环与创伤",
        keywords=["大出血", "大量出血", "血流不止", "呕血", "咯血不止", "喷射状出血"],
        guideline_source="待核验候选资料：《创伤失血性休克中国急诊专家共识》《急性上消化道出血急诊诊治流程专家共识》",
        clinical_rationale="急性活动性出血可急剧引发低血容量性休克与循环衰竭，属于急诊分诊Ⅰ/Ⅱ级危急症。",
        applicable_conditions="喷射状或涌出性出血、持续加压包扎无效、单次呕血>200ml或咯鲜血、伴面色苍白和冷汗。",
        negation_patterns=[
            r"无大出血", r"没有大出血", r"未见大出血", r"血已止住", r"已完全止血",
            r"不是大出血", r"排除了?出血", r"无大量出血",
        ],
        false_positive_cases=[
            "手指擦破皮出了一点血，已完全止血，无大出血",
            "医生检查凝血功能良好，无大出血风险",
            "轻微牙龈渗血，不是大出血",
        ],
    ),
    EmergencyRule(
        rule_id="EMERG_TOXIC_001",
        name="急性中毒 / 毒物接触",
        category="急诊毒理",
        keywords=["中毒", "误服农药", "百草枯", "有机磷中毒", "一氧化碳中毒", "误服洗涤剂"],
        guideline_source="待核验候选资料：《急性中毒急诊诊断与治疗专家共识》《中国急性百草枯中毒诊治共识》",
        clinical_rationale="有毒物质吸收迅速，多脏器衰竭及不可逆纤维化发生快，需紧急清除未吸收毒物并使用特效解毒剂。",
        applicable_conditions="明确有毒物口服/吸入/皮肤接触史，或群体性出现头晕、恶心、多汗、瞳孔改变等症状。",
        negation_patterns=[
            r"无中毒", r"未见中毒", r"没有中毒", r"排除了?中毒", r"不是中毒",
            r"非中毒", r"化验显示未中毒",
            r"(?:读了|看了|一篇|关于|科普|文章|防范).*中毒",
        ],
        false_positive_cases=[
            "吃了变质面包肚子胀，检查排除了食物中毒",
            "读了一篇关于一氧化碳中毒防范的科普文章",
            "化验各项指标正常，不是中毒",
        ],
    ),
    EmergencyRule(
        rule_id="EMERG_HEADACHE_001",
        name="雷击样 / 爆发性剧烈头痛",
        category="神经系统",
        keywords=["剧烈头痛", "爆裂样头痛", "雷击样头痛", "炸裂般头痛"],
        guideline_source="待核验候选资料：《急诊头痛诊断与处理专家共识》《蛛网膜下腔出血急诊诊治指南》",
        clinical_rationale="瞬间达峰的极剧烈头痛（雷击样头痛）是颅内动脉瘤破裂导致蛛网膜下腔出血（SAH）的典型特征，极具致死与致残风险。",
        applicable_conditions="既往无类似剧痛发作、突发数秒至数分钟内达顶峰、常伴颈项强直、呕吐或意识改变。",
        negation_patterns=[
            r"无剧烈头痛", r"没有剧烈头痛", r"不是剧烈头痛", r"轻度头痛",
            r"偶尔隐痛", r"非剧烈头痛",
        ],
        false_positive_cases=[
            "平时偶尔有一点轻微头胀，无剧烈头痛",
            "工作疲劳引起隐痛，不是剧烈头痛",
        ],
    ),
]

# 单一事实来源：导出所有关键词的扁平唯一列表（供历史依赖代码无缝引用）
EMERGENCY_SYMPTOM_KEYWORDS: list[str] = sorted(
    list({kw for rule in EMERGENCY_RULES for kw in rule.keywords})
)

# 统一前缀否定词表（当关键词前面若干字符内存在显式否定修饰时压制命中）
PREFIX_NEGATION_WORDS = ["无", "未", "没", "没有", "未见", "排除", "排除了", "绝无", "并非", "不是", "并未"]


@dataclass
class EmergencyEvaluationResult:
    is_emergency: bool
    matched_rules: list[dict[str, Any]]
    suppressed_rules: list[dict[str, Any]]
    guideline_sources: list[str]
    matched_keywords: list[str]

    @property
    def audit_summary(self) -> str:
        """生成供日志与审计跟踪的单行摘要。"""
        if not self.is_emergency:
            if self.suppressed_rules:
                supp = ",".join(r["rule_id"] for r in self.suppressed_rules)
                return f"急症安全判定: 正常（存在 {len(self.suppressed_rules)} 条规则被否定表达排除: {supp}）"
            return "急症安全判定: 正常（未触碰危急症关键词）"
        matched = ",".join(r["rule_id"] for r in self.matched_rules)
        return f"🚨 急症安全判定: 触发（命中规则: {matched}，涉及关键词: {self.matched_keywords}）"


class EmergencyRuleEngine:
    """急症规则引擎：基于演示规则的判定 + 否定表达排除。"""

    def __init__(self, rules: list[EmergencyRule] | None = None):
        self.rules = rules or EMERGENCY_RULES

    _CLAUSE_BOUNDARY = re.compile(
        r"[，,。；;！？!?\n]|但是|不过|然而|但|今天|现在|目前|随后|后来|刚刚"
    )
    _NEGATIVE_PREFIX = re.compile(
        r"(?:没有|未见|并未|并非|不是|排除了?|绝无|无|未|没)"
        r"(?:出现|发生|明显|任何)?\s*$"
    )
    _NEGATIVE_SUFFIX = re.compile(r"^\s*(?:已完全止血|已止|已缓解|已愈|消失|已被排除)")

    def _occurrence_negated(
        self, clause: str, start: int, end: int, rule: EmergencyRule
    ) -> tuple[bool, str]:
        # A rule-specific pattern only applies to the occurrence it actually spans.
        # "无胸痛" must not suppress "心跳骤停" in the same rule.
        for pattern in rule.negation_patterns:
            for match in re.finditer(pattern, clause):
                if match.start() <= start and match.end() >= end:
                    return True, f"命中局部否定模式: {pattern}"
        prefix = clause[:start]
        if self._NEGATIVE_PREFIX.search(prefix):
            return True, "命中紧邻前缀否定"
        if re.search(r"(?:排除了?|无|未见|没有)(?:食物|一氧化碳|有机磷)$", prefix):
            return True, "命中中毒类型的紧邻否定"
        # Negation may be coordinated: "未见抽搐与昏迷". Do not extend
        # through arbitrary words such as "无发热伴胸痛".
        coordinated = re.search(
            r"(?:没有|未见|无)([\u4e00-\u9fff、与及和]+[、与及和])$", prefix
        )
        if coordinated:
            terms = re.split(r"[、与及和]", coordinated.group(1))
            if all(not term or term in EMERGENCY_SYMPTOM_KEYWORDS for term in terms):
                return True, "命中并列症状否定"
        if self._NEGATIVE_SUFFIX.search(clause[end:]):
            return True, "命中紧邻后缀否定"
        return False, ""

    def is_negated(self, text: str, kw: str, rule: EmergencyRule) -> tuple[bool, str]:
        """仅在所有出现位置均被局部否定时，才排除该关键词。"""
        reasons = []
        for clause in self._CLAUSE_BOUNDARY.split(text):
            for match in re.finditer(re.escape(kw), clause):
                # A resolved historical occurrence may be ignored; a later
                # affirmative occurrence is still evaluated independently.
                if (re.search(r"曾经|以前|\d+年前|小时候|既往", clause[:match.start()])
                        and re.search(r"(?:现在|目前)(?:已经|已)?(?:完全正常|痊愈|恢复正常)", text)):
                    reasons.append("历史症状且当前明确恢复正常")
                    continue
                negated, reason = self._occurrence_negated(
                    clause, match.start(), match.end(), rule
                )
                if not negated:
                    return False, ""
                reasons.append(reason)
        return bool(reasons), "；".join(dict.fromkeys(reasons))

    def evaluate(self, text: str, symptoms: list[str] | None = None) -> EmergencyEvaluationResult:
        """评估文本与症状列表，返回完整的急症判定与审计追踪。"""
        # Do not append a normalized copy of a symptom already present in the
        # query: that would resurrect "无胸痛" as an affirmative "胸痛".
        query_text = text or ""
        extra_symptoms = [s for s in (symptoms or []) if s and s not in query_text]
        full_text = "；".join([query_text, *extra_symptoms]).strip("； ")
        matched_rules: list[dict[str, Any]] = []
        suppressed_rules: list[dict[str, Any]] = []
        guidelines_set: set[str] = set()
        matched_kws: list[str] = []

        if not full_text:
            return EmergencyEvaluationResult(
                is_emergency=False,
                matched_rules=[],
                suppressed_rules=[],
                guideline_sources=[],
                matched_keywords=[],
            )

        for rule in self.rules:
            hit_kws = [kw for kw in rule.keywords if kw in full_text]
            if not hit_kws:
                continue

            # 对命中的关键词进行否定分析
            rule_negated = True
            negation_reason = ""
            unnegated_kws = []

            for kw in hit_kws:
                negated, reason = self.is_negated(full_text, kw, rule)
                if negated:
                    negation_reason = reason
                else:
                    rule_negated = False
                    unnegated_kws.append(kw)

            if rule_negated:
                suppressed_rules.append({
                    "rule_id": rule.rule_id,
                    "rule_name": rule.name,
                    "category": rule.category,
                    "suppression_reason": negation_reason,
                    "guideline_source": rule.guideline_source,
                    "guideline_verified": False,
                })
            else:
                matched_rules.append({
                    "rule_id": rule.rule_id,
                    "rule_name": rule.name,
                    "category": rule.category,
                    "matched_keywords": unnegated_kws,
                    "guideline_source": rule.guideline_source,
                    "guideline_verified": False,
                    "clinical_rationale": rule.clinical_rationale,
                    "applicable_conditions": rule.applicable_conditions,
                })
                guidelines_set.add(rule.guideline_source)
                matched_kws.extend(unnegated_kws)

        is_emergency = len(matched_rules) > 0

        return EmergencyEvaluationResult(
            is_emergency=is_emergency,
            matched_rules=matched_rules,
            suppressed_rules=suppressed_rules,
            guideline_sources=sorted(list(guidelines_set)),
            matched_keywords=sorted(list(set(matched_kws))),
        )


# 全局单例引擎
emergency_engine = EmergencyRuleEngine()
