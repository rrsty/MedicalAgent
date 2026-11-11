"""Real FAISS tests use injected embeddings/graph/model: no network or weights."""
from unittest.mock import MagicMock
import pytest
import numpy as np
from evals.graphrag_benchmark import (
    EvidenceBudgetManager, GraphRAGBenchmarkSuite, HomologousKnowledgeBase,
    PureVectorRetriever, GraphRAGRetriever, compute_metrics, text_hash, score_saved_report,
)


class Encoder:
    def encode(self, texts, **kwargs):
        return np.array([[1., 0.] if t.startswith(("偏头痛", "头痛")) else [0., 1.]
                         for t in texts], dtype="float32")


def corpus():
    return HomologousKnowledgeBase(nodes=[
        {"id": "1", "type": "Disease", "props": {"name": "偏头痛", "description": "头痛"}},
        {"id": "2", "type": "Department", "props": {"name": "神经内科"}},
    ], edges=[{"source": "1", "target": "2", "relation": "BELONGS_TO_DEPARTMENT"}])


def queries():
    q = MagicMock()
    q.expand_multihop.return_value = {"triples": [
        {"subject": "偏头痛", "predicate": "BELONGS_TO_DEPARTMENT", "object": "神经内科"},
    ]}
    return q


def test_same_corpus_contains_full_relationships_and_real_faiss():
    kb = corpus()
    assert all("BELONGS_TO_DEPARTMENT" in d["content"] for d in kb.rich_documents)
    retriever = PureVectorRetriever(kb, encoder=Encoder())
    assert retriever.index.__class__.__name__ == "IndexFlatIP"
    assert retriever.index.ntotal == 2
    docs, latency = retriever.search("头痛")
    assert docs[0]["name"] == "偏头痛"
    assert isinstance(latency, float)


def test_graph_uses_same_anchor_count_and_database_queries():
    kb = corpus()
    vector = PureVectorRetriever(kb, encoder=Encoder())
    q = queries()
    graph = GraphRAGRetriever(kb, vector, queries=q)
    contexts, latency, triples = graph.search("头痛", top_k=1)
    q.expand_multihop.assert_called_once_with("偏头痛", entity_type="Disease")
    assert triples[0]["subject"] == "偏头痛"
    assert latency >= 0


def test_budget_is_an_exact_upper_bound():
    for cap in [1, 5, 100]:
        assert len(EvidenceBudgetManager(cap).apply("长证据" * 100)) <= cap


def test_contradiction_is_not_automatically_scored_as_supported():
    answer = "阿莫西林能治疗病毒感染。"
    context = "阿莫西林不能治疗病毒感染。"
    assert compute_metrics(answer, context, answer, [], [])[:2] == (None, None)
    review = {"reviewer": "test-reviewer", "answer_hash": text_hash(answer),
              "context_hash": text_hash(context), "factual_correct": False,
              "correctness_basis": "否定关系与原文相反", "claims_complete": True,
              "claims": [{"claim": answer, "verdict": "contradicted"}]}
    assert compute_metrics(answer, context, answer, [], [], review)[:2] == (0., 0.)
    with pytest.raises(ValueError, match="hashes"):
        compute_metrics("不同答案", context, answer, [], [], review)


def test_model_failure_remains_an_error_and_triage_failure_counts_in_denominator():
    def unavailable(*args):
        raise RuntimeError("model unavailable")
    suite = GraphRAGBenchmarkSuite(kb=corpus(), encoder=Encoder(), queries=queries(), generator=unavailable)
    result = suite.run_benchmark(cases=[{
        "id": "a1", "query": "头痛挂什么科", "tags": ["appointment"],
        "gold": {"departments_acceptable": ["神经内科"]},
    }])
    for variant in ["vector", "graph"]:
        row = result["samples"][0][variant]
        assert row["status"] == "error"
        assert "model unavailable" in row["error"]
        assert "answer" not in row
    assert result["vector_baseline"]["triage_hit3"] == 0
    assert result["vector_baseline"]["triage_samples"] == 1
    assert result["vector_baseline"]["faithfulness"] is None


def test_benchmark_retains_raw_evidence_and_does_not_invent_medical_scores():
    suite = GraphRAGBenchmarkSuite(kb=corpus(), encoder=Encoder(), queries=queries(),
                                  generator=lambda q, c: ("请咨询神经内科。", 0.1))
    result = suite.run_benchmark(cases=[{
        "id": "a1", "query": "头痛", "tags": ["appointment"],
        "gold": {"departments_acceptable": ["神经内科"]},
    }])
    assert result["total_samples"] == 1
    assert result["vector_baseline"]["triage_hit3"] == 1
    assert result["graphrag"]["faithfulness_reviewed"] == 0
    assert result["samples"][0]["graph"]["context_hash"]
    assert result["samples"][0]["vector"]["factual_accuracy"] is None


def test_saved_answers_can_be_scored_without_regeneration():
    generator = MagicMock(return_value=("请咨询神经内科。", 0.1))
    suite = GraphRAGBenchmarkSuite(kb=corpus(), encoder=Encoder(), queries=queries(),
                                  generator=generator)
    result = suite.run_benchmark(cases=[{
        "id": "a1", "query": "头痛", "tags": ["appointment"],
        "gold": {"departments_acceptable": ["神经内科"]},
    }])
    arm = result["samples"][0]["vector"]
    review = {
        "reviewer": "test-reviewer", "answer_hash": arm["answer_hash"],
        "context_hash": arm["context_hash"], "factual_correct": True,
        "correctness_basis": "test fixture", "claims_complete": True,
        "claims": [{"claim": "请咨询神经内科", "verdict": "supported",
                    "support_quote": "神经内科"}],
    }
    calls_before = generator.call_count
    result = score_saved_report(result, {"a1:vector": review})
    assert generator.call_count == calls_before
    assert result["vector_baseline"]["factual_accuracy"] == 1
    assert result["graphrag"]["factual_accuracy"] is None
    assert result["vector_baseline"]["accuracy_reviewed"] == 1
