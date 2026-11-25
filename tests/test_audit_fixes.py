"""Regression tests for code audit Batch 1 & Batch 2 fixes."""
import asyncio
import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import faiss
import numpy as np
import pytest
from fastapi.testclient import TestClient

from config.settings import settings
from crawler.medical_crawler import MedicalCrawler
from evals.graphrag_benchmark import triage_department_ranker
from evals.judge import llm_judge
from evals.runner import _build_error_sample, aggregate, gates
from scripts.build_index import entity_label, entity_text
from src.agents.answer_fusion import AnswerFusion
from src.agents.emergency_rules import emergency_engine
from src.agents.graph import route_after_department
from src.agents.intent_classifier import is_potential_medical_query
from src.agents.medical_knowledge import SAFE_DEFERRAL, MedicalKnowledgeAgent
from src.agents.medication_advisor import MedicationAdvisor
from src.agents.safety_checker import SafetyCheckerAgent
from src.agents.state import IntentType, Severity
from src.agents.symptom_detector import SymptomDetection, symptom_detector
from src.agents.tools import get_react_tools
from src.api.main import app as fastapi_app
from src.api.models import ConsultationResponse
from src.api.routes import _is_cacheable_response, init_router
from src.common.llm import _thinking_extra_body
from src.common.utils import coerce_str_list, parse_llm_json
from src.extraction.disease_extractor import _OUTPUT_SCHEMA_TEXT
from src.extraction.schemas import DiseaseInfo, SimpleLLMOutput
from src.knowledge_graph.graph_builder import KnowledgeGraphBuilder
from src.knowledge_graph.graph_queries import GraphQueries
from src.knowledge_graph.schema import NodeType, RelationType
from src.knowledge_graph.text2cypher import (
    _ensure_limit,
    _validate_hop_budget,
    _validate_readonly,
)
from src.vector_store.search import VectorRetriever


# ---------- 1. Emergency Rules & SafetyChecker & AnswerFusion ----------


@pytest.mark.parametrize(
    "query",
    [
        "我胸口闷，是不是心梗？",
        "吃了发芽土豆，我是不是中毒了？",
        "孩子翻白眼，是不是抽搐？",
        "医生，我这样有没有呼吸困难？",
        "我有没有中毒？",
        "突然胸口发紧，不知道是不是心肌梗死",
    ],
)
def test_emergency_interrogatives_are_never_suppressed(query):
    res = emergency_engine.evaluate(query)
    assert res.is_emergency is True, f"急症疑问句不应被否定词误压制: {query}"


@pytest.mark.parametrize(
    "query",
    [
        "无恶心、呕吐、胸痛",
        "排除了药物中毒",
        "小时候抽搐过，现在好转了",
    ],
)
def test_emergency_expanded_negation_cases(query):
    res = emergency_engine.evaluate(query)
    assert res.is_emergency is False, f"明确否定/既往已好转表达不应误报急症: {query}"


def test_safety_checker_respects_llm_emergency_severity():
    checker = SafetyCheckerAgent()
    out = checker({
        "intent": IntentType.APPOINTMENT.value,
        "query": "我刚才吞了一整瓶安眠药想睡觉",
        "symptoms": ["药物过量"],
        "severity": Severity.EMERGENCY.value,
        "history": [],
    })
    assert out["warnings"], "severity=emergency 必须触发急症警告"
    assert out["emergency_audit"]["is_emergency"] is True
    assert out["intent"] == IntentType.EMERGENCY.value


def test_safety_checker_ignores_assistant_mentions_and_supports_multiple_populations():
    checker = SafetyCheckerAgent()
    # Assistant previously mentioned 孕妇及儿童慎用; current non-pregnant user should NOT trigger
    out_clean = checker({
        "intent": IntentType.MEDICATION.value,
        "query": "感冒流鼻涕吃什么药",
        "symptoms": ["流涕"],
        "history": [
            {"role": "user", "content": "阿司匹林副作用"},
            {"role": "assistant", "content": "可能引起胃部不适，孕妇及儿童慎用。"},
        ],
    })
    assert not any("孕妇" in d or "儿童" in d for d in out_clean["disclaimers"])

    # Multiple populations in user query should BOTH emit reminders
    out_multi = checker({
        "intent": IntentType.MEDICATION.value,
        "query": "我在哺乳期，家里老人也有感冒",
        "symptoms": ["感冒"],
        "history": [],
    })
    assert any("产科" in d for d in out_multi["disclaimers"])
    assert any("老年人" in d for d in out_multi["disclaimers"])


def test_medication_advisor_ignores_assistant_pregnancy_and_filters_unlisted_drugs():
    advisor = MedicationAdvisor()
    mock_queries = MagicMock()
    mock_queries.match_disease_in_text.return_value = "感冒"
    mock_queries.get_disease_info.return_value = {
        "name": "感冒",
        "medications": ["对乙酰氨基酚"],
    }
    advisor.queries = mock_queries

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content=json.dumps({
            "medications": [
                {"name": "编造神药", "purpose": "万能", "side_effects": ""},
                {"name": "对乙酰氨基酚", "purpose": "解热镇痛", "side_effects": "肝损伤风险"},
            ]
        }, ensure_ascii=False)
    )
    with patch("src.agents.medication_advisor.get_llm", return_value=mock_llm):
        out = advisor({
            "query": "感冒吃什么药",
            "symptoms": ["发热"],
            "history": [
                {"role": "user", "content": "布洛芬有什么副作用"},
                {"role": "assistant", "content": "孕妇及哺乳期禁用。"},
            ],
        })
    meds = out["medications"]
    assert [m["name"] for m in meds] == ["对乙酰氨基酚"]
    assert "孕期" not in meds[0]["reminder"]


def test_answer_fusion_short_circuits_upgraded_emergency_and_pure_knowledge():
    fusion = AnswerFusion()
    mock_llm = MagicMock()
    with patch("src.agents.answer_fusion.get_llm", return_value=mock_llm):
        # Upgraded emergency (intent not yet emergency, but warnings present)
        r_emerg = fusion({
            "intent": IntentType.APPOINTMENT.value,
            "warnings": ["🚨 急症警告"],
            "disclaimers": ["🔴 重要声明"],
        })
        assert not mock_llm.invoke.called
        assert r_emerg["final_answer"].startswith("🚨 急症警告")

        # Pure knowledge intent bypasses redundant LLM rewrite
        r_know = fusion({
            "intent": IntentType.KNOWLEDGE.value,
            "knowledge_answer": "糖尿病是以高血糖为特征的代谢性疾病。",
            "disclaimers": ["🔴 重要声明"],
        })
        assert not mock_llm.invoke.called
        assert "糖尿病是以高血糖为特征的代谢性疾病。" in r_know["final_answer"]
        assert "🔴 重要声明" in r_know["final_answer"]


def test_answer_fusion_appends_every_missing_disclaimer():
    fusion = AnswerFusion()
    mock_llm = MagicMock()
    # LLM only kept the first disclaimer and dropped the second
    mock_llm.invoke.return_value = MagicMock(
        content="推荐挂神经内科。\n\n🔴 重要声明：本建议仅供参考。"
    )
    with patch("src.agents.answer_fusion.get_llm", return_value=mock_llm):
        out = fusion({
            "intent": IntentType.APPOINTMENT.value,
            "symptoms": ["头痛"],
            "departments": ["神经内科"],
            "disclaimers": [
                "🔴 重要声明：本建议仅供参考，不能替代专业医疗诊断。",
                "🤰 您提到的情况如涉及孕妇/孕哺期，用药需特别谨慎。",
            ],
        })
    assert "孕妇/孕哺期" in out["final_answer"]


# ---------- 2. MedicalKnowledge Deferral, Intent Gate, Routing & Latency ----------


def test_medical_knowledge_high_risk_requires_direct_evidence(monkeypatch):
    agent = MedicalKnowledgeAgent()
    retriever = MagicMock()
    retriever.link_entities.return_value = []
    monkeypatch.setattr("src.agents.medical_knowledge.get_shared_retriever", lambda: retriever)
    monkeypatch.setattr(settings, "enable_react_layer", False)

    # Weak single-drug context without INTERACTS_WITH must defer
    monkeypatch.setattr(
        agent, "_retrieve",
        lambda *a: ([{"name": "布洛芬", "type": "Medication"}], ["布洛芬（药物）：用于解热镇痛  [数据源: unknown]"]),
    )
    res_defer = agent({"query": "布洛芬和头孢能一起吃吗", "intent": "medication"})
    assert res_defer["knowledge_answer"] == SAFE_DEFERRAL

    # Dosage query always defers
    res_dose = agent({"query": "二甲双胍一天吃多少剂量", "intent": "medication"})
    assert res_dose["knowledge_answer"] == SAFE_DEFERRAL

    # Direct INTERACTS_WITH triple permits grounded answer
    monkeypatch.setattr(
        agent, "_retrieve",
        lambda *a: (
            [{"name": "阿司匹林", "type": "Medication"}],
            ["(阿司匹林)-[:INTERACTS_WITH]->(华法林)  [数据源: demo]"],
        ),
    )
    monkeypatch.setattr(agent, "_grounded_static_answer", lambda *a: "阿司匹林与华法林存在相互作用，请遵医嘱。")
    res_grounded = agent({"query": "阿司匹林和华法林能一起吃吗", "intent": "general"})
    assert "相互作用" in res_grounded["knowledge_answer"]
    assert res_grounded["intent"] == IntentType.KNOWLEDGE.value


@pytest.mark.parametrize(
    "chat",
    [
        "今天好热啊",
        "今天好冷啊",
        "帮我拍张照片",
        "推荐一部科幻片",
        "这段代码有什么因素",
        "你好，今天好冷",
        "谢谢你的关心",
        "苹果好酸啊",
        "今天上班好累",
    ],
)
def test_everyday_chat_not_intercepted_as_medical(chat):
    assert is_potential_medical_query(chat) is False


def test_route_after_department_skips_pre_visit_on_clarification():
    assert route_after_department({"needs_clarification": True}) == "safety_checker"
    assert route_after_department({"needs_clarification": False}) == "pre_visit_advisor"


def test_symptom_detector_skips_retry_for_disease_named_query(monkeypatch):
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.return_value = SymptomDetection(
        symptoms=[], severity="low"
    )
    monkeypatch.setattr("src.agents.symptom_detector.get_llm", lambda **kwargs: llm)
    out = symptom_detector({"query": "高血压吃什么药好？", "history": []})
    assert out["symptoms"] == []
    assert llm.with_structured_output.return_value.invoke.call_count == 1


def test_get_react_tools_respects_text2cypher_flag(monkeypatch):
    monkeypatch.setattr(settings, "text2cypher_enabled", False)
    names_off = [t.name for t in get_react_tools()]
    assert "natural_language_graph_query" not in names_off

    monkeypatch.setattr(settings, "text2cypher_enabled", True)
    names_on = [t.name for t in get_react_tools()]
    assert "natural_language_graph_query" in names_on


# ---------- 3. KG Schema, Builder, Seed Data, Queries & VectorRetriever ----------


def test_seed_demo_data_separates_avoid_foods_and_includes_warfarin(tmp_path, monkeypatch):
    import scripts.seed_demo_data as sd

    written: dict[str, dict] = {}
    monkeypatch.setattr(sd, "write_json", lambda path, data: written.__setitem__(Path(path).name, data))
    sd.build_seed()

    entities = written["entities.json"]["entities"]
    relations = written["relations.json"]["relations"]

    # Avoid foods use AVOIDS_FOOD, not RECOMMENDS_FOOD
    recommends = [r["data"] for r in relations if r["type"] == RelationType.RECOMMENDS_FOOD.value]
    avoids = [r["data"] for r in relations if r["type"] == RelationType.AVOIDS_FOOD.value]
    assert any(d["disease_name"] == "糖尿病" and d["food_name"] == "高糖食品" for d in avoids)
    assert not any(d["disease_name"] == "糖尿病" and d["food_name"] == "高糖食品" for d in recommends)

    # Warfarin interaction is present
    interactions = [
        (r["data"]["drug_name"], r["data"]["interacting_drug_name"])
        for r in relations
        if r["type"] == RelationType.INTERACTS_WITH.value
    ]
    assert ("阿司匹林", "华法林") in interactions

    # Provenance attached to all entities & relations
    assert all(e.get("provenance") for e in entities)
    assert all(r.get("provenance") for r in relations)

    # Precaution uses content (and build_index label extracts content)
    precautions = [e for e in entities if e["type"] == NodeType.PRECAUTION.value]
    assert precautions
    assert all(entity_label(p) == p["content"] for p in precautions)
    assert all("（Precaution）" != entity_text(p) for p in precautions)


def test_graph_builder_matches_precaution_by_content_and_redirects_legacy_avoid_food():
    builder = KnowledgeGraphBuilder.__new__(KnowledgeGraphBuilder)
    mock_driver = MagicMock()
    mock_tx = mock_driver.session.return_value.__enter__.return_value.begin_transaction.return_value.__enter__.return_value
    mock_tx.run.return_value.single.return_value = {"1": 1}
    builder.driver = mock_driver

    created = builder.build_relations([
        {
            "type": RelationType.HAS_PRECAUTION.value,
            "data": {"disease_name": "高血压", "precaution_content": "低盐饮食"},
        },
        {
            "type": RelationType.RECOMMENDS_FOOD.value,
            "data": {"disease_name": "糖尿病", "food_name": "高糖食品", "recommendation_type": "禁忌"},
        },
    ])
    assert created == 2
    cyphers = [call.args[0] for call in mock_tx.run.call_args_list]
    assert "{content: $end_name}" in cyphers[0]
    assert "AVOIDS_FOOD" in cyphers[1]


def test_graph_queries_get_disease_info_uses_pattern_comprehension():
    with patch("src.knowledge_graph.graph_queries.get_driver") as mock_get_driver:
        session = mock_get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value.single.return_value = {
            "name": "高血压",
            "description": "desc",
            "symptoms": ["头痛", "头痛"],
            "departments": ["心血管内科"],
            "medications": ["氨氯地平"],
            "examinations": ["血压测量"],
            "body_parts": ["心脏"],
        }
        info = GraphQueries().get_disease_info("高血压")
        cypher = session.run.call_args.args[0]
        assert "OPTIONAL MATCH" not in cypher
        assert info["symptoms"] == ["头痛"]


def test_vector_retriever_handles_index_flat_ip_and_prioritizes_summaries(tmp_path, monkeypatch):
    # IndexFlatIP load should not raise AttributeError on nprobe
    idx = faiss.IndexFlatIP(4)
    idx.add(np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype="float32"))
    idx_file = tmp_path / "faiss.index"
    map_file = tmp_path / "entities.json"
    idx_file.write_bytes(faiss.serialize_index(idx).tobytes())
    map_file.write_text(
        json.dumps({"entities": [{"name": "A", "type": "Disease"}, {"name": "B", "type": "Disease"}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr("src.vector_store.search.INDEX_FILE", idx_file)
    monkeypatch.setattr("src.vector_store.search.INDEX_ENTITY_MAPPING_FILE", map_file)
    monkeypatch.setattr("src.vector_store.search.read_json", lambda p: json.loads(Path(p).read_text(encoding="utf-8")))
    monkeypatch.setattr(settings, "embedding_dim", 4)
    retriever = VectorRetriever()
    assert retriever.is_ready is True

    # hybrid_search places summaries for all entities ahead of raw multihop triples
    retriever.search = lambda *a: [
        {"name": "糖尿病", "type": "Disease", "description": "高血糖代谢病"},
        {"name": "高血压", "type": "Disease", "description": "动脉血压增高"},
    ]
    queries = MagicMock()
    queries.expand_entity.side_effect = lambda name, entity_type=None: {
        "type": "Disease",
        "facts": {"symptoms": [f"{name}症状"], "avoid_foods": ["高糖食品"] if name == "糖尿病" else []},
    }
    queries.expand_multihop.return_value = {
        "triples": [
            {"subject": "糖尿病", "predicate": "HAS_SYMPTOM", "object": f"症状{i}", "provenance": []}
            for i in range(10)
        ]
    }
    monkeypatch.setattr("src.vector_store.search.GraphQueries", lambda: queries)
    out = retriever.hybrid_search("糖尿病和高血压")
    # First 4 items must include both entities' descriptions and structured summaries
    top4 = "\n".join(out["contexts"][:4])
    assert "糖尿病（Disease）：高血糖代谢病" in top4
    assert "忌口/禁忌食物: 高糖食品" in top4
    assert "高血压（Disease）：动脉血压增高" in top4


# ---------- 4. API, Crawler, Text2Cypher, LLM, Utils & Contracts ----------


def test_api_passes_request_id_skips_degraded_cache_and_returns_404_for_unknown_api():
    assert _is_cacheable_response(ConsultationResponse(answer="正常回答")) is True
    assert _is_cacheable_response(ConsultationResponse(answer="抱歉，暂时无法处理您的问题，请尝试换一种提问方式。")) is False

    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "final_answer": "正常医疗建议",
        "intent": "knowledge",
        "disclaimers": ["免责"],
    }
    init_router({"graph_app": mock_graph, "node_labels": {}}, asyncio.Semaphore(4))

    with patch("src.history.store.load_context", return_value=[]), \
         patch("src.history.store.cache_get", return_value=None), \
         patch("src.history.store.cache_set") as mock_cache_set, \
         patch("src.history.store.save_turn", return_value=True) as mock_save:
        client = TestClient(fastapi_app)
        resp = client.post(
            "/api/consult",
            json={"query": "什么是糖尿病", "session_id": "sess-audit-01", "request_id": "req-dedup-001"},
        )
        assert resp.status_code == 200
        mock_save.assert_called_once()
        assert mock_save.call_args.kwargs.get("request_id") == "req-dedup-001"
        assert mock_cache_set.called

        # Nonexistent /api/* path must return 404, not SPA 200
        r404 = client.get("/api/definitely-does-not-exist")
        assert r404.status_code == 404


def test_crawler_robots_txt_properly_extracts_path():
    crawler = MedicalCrawler.__new__(MedicalCrawler)
    crawler._robots_rules = [(False, "/citiao/")]
    assert crawler.can_fetch("https://www.haodf.com/citiao/jibing-gaoxueya.html") is False
    assert crawler.can_fetch("https://www.haodf.com/jibing/index.html") is True


def test_text2cypher_hardening():
    # String literal 'set' is not rejected as write keyword
    assert _validate_readonly("MATCH (d:Disease) WHERE d.description CONTAINS 'set' RETURN d.name") is None
    assert _validate_readonly("MATCH (d:Disease) SET d.name = 'x' RETURN d.name") is not None
    # Excessive LIMIT is capped
    assert _ensure_limit("MATCH (d:Disease) RETURN d.name LIMIT 99999", max_rows=20).endswith("LIMIT 20")
    assert _ensure_limit("MATCH (d:Disease) RETURN d.name LIMIT 5", max_rows=20).endswith("LIMIT 5")
    # Cypher 5 COUNT subquery rejected
    assert _validate_hop_budget("MATCH (d:Disease) WHERE COUNT { (d)--() } > 1 RETURN d.name") is not None


def test_llm_thinking_extra_body_adapts_by_provider():
    assert _thinking_extra_body("https://dashscope.aliyuncs.com/compatible-mode/v1", True) == {
        "enable_thinking": False
    }
    assert _thinking_extra_body("https://api.deepseek.com/v1", True) == {
        "thinking": {"type": "disabled"}
    }
    assert _thinking_extra_body("https://api.openai.com/v1", True) is None
    assert _thinking_extra_body("https://dashscope.aliyuncs.com/compatible-mode/v1", False) is None


def test_utils_and_extraction_fixes():
    assert coerce_str_list({"name": "头痛"}) == ["头痛"]
    assert coerce_str_list(123) == ["123"]
    assert parse_llm_json('{"intent": "general"} (note: format is {intent})') == {"intent": "general"}
    assert "包含以下字段：，包含以下字段：" not in _OUTPUT_SCHEMA_TEXT

    cleaned = SimpleLLMOutput(
        disease=DiseaseInfo(name="疾病名称", description="疾病描述"),
        symptoms=["头痛", "头痛", "症状1"],
    ).clean()
    assert cleaned.disease.name == ""
    assert cleaned.disease.description is None
    assert cleaned.symptoms == ["头痛"]


def test_check_contracts_fstring_enum_regex_actually_works():
    src = Path("scripts/check_contracts.py").read_text(encoding="utf-8")
    assert "\x08" not in src and "\x01" not in src
    m = re.search(r'fstring_enum = re\.compile\(\s*r"""(.*?)"""\s*\)', src, re.S)
    assert m is not None
    regex = re.compile(m.group(1))
    assert regex.search('q = f"MATCH (n:{NodeType.DISEASE}) RETURN n"') is not None
    assert regex.search('q = f"MATCH (n:{NodeType.DISEASE.value}) RETURN n"') is None


# ---------- 5. Evals Runner, Judge & Benchmark Fixes ----------


def test_evals_aggregate_includes_failed_samples_and_zero_emergency_cases_does_not_pass():
    results = [
        {
            "id": "a1", "error": None, "latency_ms": 100,
            "intent_ok": True, "disclaimer_ok": True,
            "gold_emergency": False, "emergency_pred": False,
            "symptom_f1": 1.0, "dept_hit3": 1.0, "dept_recall": 1.0,
        },
        _build_error_sample(
            {"id": "a2", "query": "q", "gold": {"symptoms": ["头痛"], "departments_acceptable": ["神经内科"]}},
            TimeoutError("timeout"),
            5000,
        ),
    ]
    agg = aggregate(results)
    # Failed case's 0.0 must be included in averages (not dropped to 1.0)
    assert agg["symptom_f1"] == 0.5
    assert agg["dept_hit3"] == 0.5
    # Zero emergency cases must NOT falsely report 1.0 recall
    assert agg["emergency_cases"] == 0
    assert agg["emergency_recall"] == 0.0
    gate_map = {name: ok for name, ok, _, _ in gates(agg)}
    assert gate_map["急症召回率 == 1.0"] is False


def test_llm_judge_returns_none_on_malformed_json():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="not valid json at all")
    with patch("src.common.llm.get_llm", return_value=mock_llm):
        assert llm_judge("什么是糖尿病", "高血糖代谢病") is None


def test_triage_department_ranker_does_not_double_count_substrings():
    deps = ["内科", "呼吸内科", "心血管内科"]
    ctx = "建议就诊呼吸内科，呼吸内科可做肺功能检查。"
    ranked = triage_department_ranker(ctx, deps)
    assert ranked == ["呼吸内科"]
