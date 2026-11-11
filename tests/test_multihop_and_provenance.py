"""Regression tests for directed bounded expansion and acquisition provenance."""
from unittest.mock import MagicMock, patch
import json
import pytest
from config.settings import settings
from src.common.provenance import (
    SourceProvenance, format_context_with_provenance, decode_provenance,
)
from src.knowledge_graph.graph_queries import GraphQueries


def edge(mid="symptom-1", target="disease-1", edge_id="edge-1", **changes):
    record = dict(mid_id=mid, target_id=target, edge_id=edge_id,
                  target_name="偏头痛", src_name="偏头痛", src_type="Disease",
                  rel="HAS_SYMPTOM", tgt_name="头痛", tgt_type="Symptom")
    record.update(changes)
    return record


def driver_session():
    context = patch("src.knowledge_graph.graph_queries.get_driver")
    driver = context.start()
    session = driver.return_value.session.return_value.__enter__.return_value
    return context, session


def test_expand_multihop_preserves_reverse_traversal_direction():
    context, session = driver_session()
    try:
        session.run.side_effect = [[edge()], []]
        result = GraphQueries().expand_multihop("头痛", entity_type="Symptom")
        assert result["triples"][0]["subject"] == "偏头痛"
        assert result["triples"][0]["predicate"] == "HAS_SYMPTOM"
        assert result["triples"][0]["object"] == "头痛"
        assert "startNode(r)" in session.run.call_args_list[0].args[0]
        assert session.run.call_args_list[0].kwargs["entity_type"] == "Symptom"
        assert set(session.run.call_args_list[1].kwargs["visited"]) == {"symptom-1", "disease-1"}
    finally:
        context.stop()


def test_expand_multihop_caps_and_provenance(monkeypatch):
    monkeypatch.setattr(settings, "kg_max_hops", 2)
    monkeypatch.setattr(settings, "kg_multihop_limit_per_hop", 1)
    monkeypatch.setattr(settings, "kg_multihop_total_limit", 1)
    context, session = driver_session()
    try:
        source = {"source_url": "https://example.test/fact", "medical_reviewed": False}
        session.run.return_value = [edge(provenance_json=[json.dumps(source)])]
        result = GraphQueries().expand_multihop("头痛", max_hops=9, limit_per_hop=99, total_limit=99)
        assert result["applied_hops"] == 2
        assert result["total_triples"] == 1
        assert session.run.call_count == 1
        assert session.run.call_args.kwargs["limit"] == 1
        assert result["triples"][0]["provenance"] == [source]
    finally:
        context.stop()


def test_expand_multihop_does_not_merge_nodes_with_same_display_name(monkeypatch):
    monkeypatch.setattr(settings, "kg_max_hops", 3)
    context, session = driver_session()
    try:
        session.run.side_effect = [
            [edge(mid="root", target="symptom", target_name="同名")],
            [edge(mid="symptom", target="disease", target_name="同名", edge_id="edge-2")], [],
        ]
        result = GraphQueries().expand_multihop("root")
        assert result["total_triples"] == 2
        assert session.run.call_args_list[2].kwargs["frontier"] == ["disease"]
    finally:
        context.stop()


def test_expand_multihop_failure_is_explicit():
    context, session = driver_session()
    try:
        session.run.side_effect = RuntimeError("database unavailable")
        result = GraphQueries().expand_multihop("头痛")
        assert result["total_triples"] == 0
        assert "database unavailable" in result["error"]
    finally:
        context.stop()


def test_relationship_does_not_inherit_unrelated_node_sources():
    context, session = driver_session()
    try:
        session.run.side_effect = [[edge(source_provenance_json=[json.dumps(
            {"source_url": "https://example.test/node-only"}
        )])], []]
        result = GraphQueries().expand_multihop("头痛")
        assert result["triples"][0]["provenance"] == []
    finally:
        context.stop()


@pytest.mark.parametrize("value", [0, -1])
def test_invalid_traversal_limit_is_rejected(value):
    with pytest.raises(ValueError):
        GraphQueries().expand_multihop("头痛", max_hops=value)


def test_cleaning_does_not_invent_acquisition_checks():
    from crawler.data_processor import clean_record
    result = clean_record({"name": "偏头痛", "url": "https://jib.xywy.com/example"})
    provenance = result["provenance"]
    assert provenance["source_platform"] == "unknown"
    assert provenance["robots_txt_status"] == "unknown"
    assert provenance["rate_limit_applied"] is None
    assert provenance["crawl_timestamp"] == "unknown"


def test_recorded_source_survives_cleaning_and_extraction():
    from crawler.data_processor import clean_record
    from scripts.run_extraction import no_llm_convert, serialize
    source = SourceProvenance.create_for_crawler(
        "https://jib.xywy.com/example", platform="xywy", robots_allowed=True,
        rate_limit_applied=True,
    ).to_dict()
    cleaned = clean_record({"name": "偏头痛", "symptoms": ["头痛"], "provenance": source})
    entities, relations = serialize(no_llm_convert(cleaned), cleaned["provenance"])
    assert all(e["provenance"][0]["source_platform"] == "xywy" for e in entities)
    assert all(r["provenance"][0]["source_url"] == source["source_url"] for r in relations)
    text = format_context_with_provenance("fact", entities[0]["provenance"])
    assert source["source_url"] in text
    assert source["crawl_timestamp"] in text


def test_provenance_default_is_unknown_and_json_list_roundtrips():
    source = SourceProvenance.create_for_crawler("https://example.test")
    assert source.robots_txt_status == "unknown"
    assert source.rate_limit_applied is None
    assert "unknown" in format_context_with_provenance("fact")
    assert decode_provenance([json.dumps(source.to_dict())]) == [source.to_dict()]


def test_hybrid_online_search_calls_bounded_expansion(monkeypatch):
    from src.vector_store.search import VectorRetriever
    retriever = VectorRetriever.__new__(VectorRetriever)
    retriever.search = lambda *args: [{"name": "头痛", "type": "Symptom", "description": ""}]
    queries = MagicMock()
    queries.expand_entity.return_value = {}
    queries.expand_multihop.return_value = {"triples": [
        {"subject": "偏头痛", "predicate": "HAS_SYMPTOM", "object": "头痛",
         "provenance": [{"source_url": "https://example.test/fact"}]},
    ]}
    monkeypatch.setattr("src.vector_store.search.GraphQueries", lambda: queries)
    output = retriever.hybrid_search("头疼")
    queries.expand_multihop.assert_called_once_with("头痛", entity_type="Symptom")
    assert "https://example.test/fact" in output["contexts"][0]
    assert "(偏头痛)-[:HAS_SYMPTOM]->(头痛)" in output["contexts"][0]
