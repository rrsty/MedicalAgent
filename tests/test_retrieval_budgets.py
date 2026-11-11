"""Limits must apply to executable paths, not just configuration metadata."""
from unittest.mock import MagicMock
import pytest
from langchain_core.messages import AIMessage
from config.settings import settings
from src.knowledge_graph.text2cypher import _validate_hop_budget, run_readonly_cypher
from src.agents.medical_knowledge import MedicalKnowledgeAgent


@pytest.mark.parametrize("cypher", [
    "MATCH (a)-[:R*]->(b) RETURN b.name",
    "MATCH (a)-[:R*1..]->(b) RETURN b.name",
    "MATCH (a)-[:R*1..3]->(b) RETURN b.name",
    "MATCH (a)-[:R]->(b)-[:R]->(c)-[:R]->(d) RETURN d.name",
    "MATCH p=((a)-[:R]->(b)){,5} RETURN p",
    "CALL { MATCH (a) RETURN a } RETURN a",
])
def test_generated_cypher_rejected_before_database(cypher, monkeypatch):
    monkeypatch.setattr(settings, "kg_max_hops", 2)
    driver = MagicMock()
    monkeypatch.setattr("src.knowledge_graph.text2cypher.get_driver", driver)
    assert not run_readonly_cypher(cypher)["success"]
    driver.assert_not_called()


@pytest.mark.parametrize("cypher", [
    "MATCH (a)-[:R]->(b) RETURN b.name",
    "MATCH (a)<-[:R]-(b)-[:S]->(c) RETURN c.name",
    "MATCH (a)-[:R*1..2]->(b) RETURN b.name",
    "MATCH (a {name: '包含*或CALL字面量'}) RETURN a.name",
])
def test_bounded_cypher_is_accepted(cypher, monkeypatch):
    monkeypatch.setattr(settings, "kg_max_hops", 2)
    assert _validate_hop_budget(cypher) is None


def test_react_limits_total_tools_even_in_parallel(monkeypatch):
    monkeypatch.setattr(settings, "react_max_tool_calls", 2)
    tool = MagicMock()
    tool.name = "query"
    monkeypatch.setattr("src.agents.tools.get_react_tools", lambda: [tool])
    llm = MagicMock()
    llm.bind_tools.return_value.invoke.return_value = AIMessage(content="", tool_calls=[
        {"id": str(i), "name": "query", "args": {}} for i in range(3)
    ])
    monkeypatch.setattr("src.agents.medical_knowledge.get_llm", lambda *a, **k: llm)
    with pytest.raises(RuntimeError, match="budget"):
        MedicalKnowledgeAgent()._react_answer("q", ["evidence"])
    tool.invoke.assert_not_called()


def test_react_stops_repeated_calls_and_uses_grounded_fallback(monkeypatch):
    monkeypatch.setattr(settings, "react_max_tool_calls", 2)
    monkeypatch.setattr(settings, "enable_react_layer", True)
    tool = MagicMock()
    tool.name = "query"
    tool.invoke.return_value = "fact"
    monkeypatch.setattr("src.agents.tools.get_react_tools", lambda: [tool])
    llm = MagicMock()
    llm.bind_tools.return_value.invoke.return_value = AIMessage(
        content="", tool_calls=[{"id": "1", "name": "query", "args": {}}]
    )
    monkeypatch.setattr("src.agents.medical_knowledge.get_llm", lambda *a, **k: llm)
    retriever = MagicMock()
    retriever.link_entities.return_value = []
    monkeypatch.setattr("src.agents.medical_knowledge.get_shared_retriever", lambda: retriever)
    agent = MedicalKnowledgeAgent()
    monkeypatch.setattr(agent, "_retrieve", lambda *a: ([], ["verified original fact"]))
    static = MagicMock(return_value="grounded answer")
    monkeypatch.setattr(agent, "_grounded_static_answer", static)
    result = agent({"query": "q"})
    assert tool.invoke.call_count == 2
    assert result["knowledge_answer"] == "grounded answer"
    static.assert_called_once_with("q", ["verified original fact"], [])
    assert any(t["step"] == "react_fallback" for t in result["trace"])


def test_crawler_does_not_claim_robots_permission_on_fetch_failure(monkeypatch):
    from crawler.medical_crawler import MedicalCrawler
    crawler = MedicalCrawler.__new__(MedicalCrawler)
    crawler.site = {"robots": "https://example.test/robots.txt"}
    crawler._robots_rules = None
    crawler._get = lambda *a, **k: None
    assert crawler.can_fetch("https://example.test/page") is False


def test_emergency_audit_survives_langgraph_state():
    from langgraph.graph import StateGraph, START, END
    from src.agents.state import AgentState
    from src.agents.safety_checker import SafetyCheckerAgent
    graph = StateGraph(AgentState)
    graph.add_node("safety", SafetyCheckerAgent())
    graph.add_edge(START, "safety")
    graph.add_edge("safety", END)
    result = graph.compile().invoke({"query": "胸痛", "intent": "emergency", "symptoms": []})
    assert result["emergency_audit"]["is_emergency"] is True
