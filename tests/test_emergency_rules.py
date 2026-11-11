"""急症规则依据、否定表达与误报过滤的单元测试。

验证内容：
1. 权威临床指南出处、临床风险依据与适用条件的元数据完整性；
2. 典型急症正例的可靠召回；
3. 否定表达（否定词窗、显式排除）的精准过滤与误报压制；
4. 每条急症规则预置误报反例（False Positive Cases）的回归验证；
5. SafetyCheckerAgent 及其审计留痕机制（emergency_audit）。
"""
import pytest
from src.agents.emergency_rules import (
    EMERGENCY_RULES,
    EMERGENCY_SYMPTOM_KEYWORDS,
    EmergencyRuleEngine,
    emergency_engine,
)
from src.agents.safety_checker import SafetyCheckerAgent
from src.agents.state import IntentType


def test_emergency_rules_metadata_integrity():
    """验证所有急症规则均具有明确的临床指南出处、适用条件和临床风险解释。"""
    assert len(EMERGENCY_RULES) >= 5, "急症规则库应覆盖常见危急重症大类"
    for rule in EMERGENCY_RULES:
        assert rule.rule_id.startswith("EMERG_"), f"规则ID必须规范: {rule.rule_id}"
        assert len(rule.keywords) > 0, f"规则必须定义关键词: {rule.rule_id}"
        assert "指南" in rule.guideline_source or "标准" in rule.guideline_source or "共识" in rule.guideline_source, (
            f"规则 {rule.rule_id} 必须标注权威指南/行业标准出处: {rule.guideline_source}"
        )
        assert len(rule.clinical_rationale.strip()) > 10, f"规则 {rule.rule_id} 必须包含充分的临床风险依据"
        assert len(rule.applicable_conditions.strip()) > 10, f"规则 {rule.rule_id} 必须包含前置/适用条件说明"
        assert len(rule.false_positive_cases) > 0, f"规则 {rule.rule_id} 必须预置误报反例用例"


def test_emergency_keywords_single_source():
    """验证导出的关键词列表无重复且覆盖所有规则的关键词。"""
    assert len(EMERGENCY_SYMPTOM_KEYWORDS) == len(set(EMERGENCY_SYMPTOM_KEYWORDS))
    all_rule_kws = {kw for r in EMERGENCY_RULES for kw in r.keywords}
    assert set(EMERGENCY_SYMPTOM_KEYWORDS) == all_rule_kws


@pytest.mark.parametrize(
    "query, expected_rule_id",
    [
        ("突发剧烈胸痛，喘不上气", "EMERG_CARDIO_001"),
        ("老人突然呼吸困难，脸色发紫", "EMERG_RESP_001"),
        ("孩子高热惊厥抽搐，怎么办？", "EMERG_NEURO_001"),
        ("车祸外伤大出血，血流不止", "EMERG_HEMORRHAGE_001"),
        ("误服了农药百草枯中毒", "EMERG_TOXIC_001"),
        ("突发雷击样剧烈头痛，恶心呕吐", "EMERG_HEADACHE_001"),
    ],
)
def test_emergency_positive_cases(query, expected_rule_id):
    """验证标准急症正例能准确命中相应规则并提供指南来源。"""
    res = emergency_engine.evaluate(query)
    assert res.is_emergency is True
    matched_ids = [r["rule_id"] for r in res.matched_rules]
    assert expected_rule_id in matched_ids
    assert len(res.guideline_sources) > 0


@pytest.mark.parametrize(
    "query, negated_kw",
    [
        ("患者目前神志清楚，无胸痛，无呼吸困难", "胸痛"),
        ("医生做过冠脉造影，排除了心梗", "心梗"),
        ("昨天体检心电图一切正常，并没有胸痛", "胸痛"),
        ("之前怀疑哮喘，检查后排除了窒息和呼吸衰竭，平稳呼吸无呼吸困难", "呼吸困难"),
        ("神志清醒，未见抽搐与昏迷", "昏迷"),
        ("手不小心被白纸划破，出了一滴血，已完全止血，无大出血", "大出血"),
        ("吃了路边摊有点恶心，化验显示未中毒", "中毒"),
        ("最近压力大有点神经性偏头痛，不是剧烈头痛", "剧烈头痛"),
    ],
)
def test_emergency_negation_and_suppression(query, negated_kw):
    """验证包含否定修饰词（无、没有、未见、排除、不是、已止）时能有效压制误报。"""
    res = emergency_engine.evaluate(query)
    # 不应被误判为急症
    assert res.is_emergency is False, f"查询应被否定词窗过滤，但不应触发急症: {query}"
    # 压制列表中必须包含被排除记录
    assert len(res.suppressed_rules) > 0
    supp_reasons = [r["suppression_reason"] for r in res.suppressed_rules]
    assert any("否定" in r for r in supp_reasons)


def test_rule_false_positive_cases_all_pass():
    """全量回归每条规则自带的 false_positive_cases，确保全部通过（不误报）。"""
    for rule in EMERGENCY_RULES:
        for fp_case in rule.false_positive_cases:
            res = emergency_engine.evaluate(fp_case)
            assert res.is_emergency is False, (
                f"规则 {rule.rule_id} 的预置反例触发了误报！\n"
                f"用例: '{fp_case}'\n"
                f"命中了: {[r['rule_id'] for r in res.matched_rules]}"
            )


def test_safety_checker_audit_output():
    """验证 SafetyCheckerAgent 执行输出中的 warnings 与 emergency_audit 审计信息。"""
    checker = SafetyCheckerAgent()

    # 1. 急症案例
    state_emerg = {
        "intent": IntentType.GENERAL.value,  # 即使模型意图误判为 general
        "query": "突发压榨性胸痛持续半小时",
        "symptoms": ["胸痛"],
        "medications": [],
    }
    result_emerg = checker(state_emerg)
    assert len(result_emerg["warnings"]) == 1
    assert "急诊" in result_emerg["warnings"][0]
    audit_emerg = result_emerg.get("emergency_audit", {})
    assert audit_emerg["is_emergency"] is True
    assert len(audit_emerg["matched_rules"]) >= 1
    assert len(audit_emerg["guidelines"]) >= 1

    # 2. 否定词过滤案例（不触发报警）
    state_neg = {
        "intent": IntentType.APPOINTMENT.value,
        "query": "患者平稳，体检无胸痛，想挂心内科咨询",
        "symptoms": [],
        "medications": [],
    }
    result_neg = checker(state_neg)
    assert len(result_neg["warnings"]) == 0
    audit_neg = result_neg.get("emergency_audit", {})
    assert audit_neg["is_emergency"] is False
    assert len(audit_neg["suppressed_rules"]) >= 1


@pytest.mark.parametrize("query", [
    "没有抽搐，但现在已经昏迷",
    "昨天没有胸痛，今天胸痛持续加重",
    "无发热，但胸痛持续半小时",
    "无发热伴胸痛",
    "没有胸痛，但心跳骤停",
    "无呼吸困难，但现在呼吸困难越来越重",
    "未见昏迷与抽搐，但是现在抽搐",
])
def test_negation_does_not_suppress_other_occurrences(query):
    assert emergency_engine.evaluate(query).is_emergency


def test_normalized_symptoms_do_not_resurrect_negated_query():
    assert not emergency_engine.evaluate("没有胸痛", ["胸痛"]).is_emergency


def test_symptom_stage_and_route_use_the_same_negation(monkeypatch):
    from src.agents.symptom_detector import SymptomDetection, symptom_detector
    from src.agents.graph import route_after_symptom, route_by_intent
    from unittest.mock import MagicMock

    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.return_value = SymptomDetection(
        symptoms=["腿酸"], severity="low"
    )
    monkeypatch.setattr("src.agents.symptom_detector.get_llm", lambda **kwargs: llm)
    state = {"intent": "appointment", "query": "我没有胸痛，只是腿酸，想挂骨科"}
    assert route_by_intent(state) == "symptom_detector"
    state.update(symptom_detector(state))
    assert state["severity"] == "low"
    assert route_after_symptom(state) == "department_recommender"
    assert not SafetyCheckerAgent()(state)["warnings"]
