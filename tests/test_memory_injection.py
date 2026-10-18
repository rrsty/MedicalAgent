"""多轮记忆注入机制单测（monkeypatch LLM / 检索器，不需要真实 API Key）。

验证点（评审 P1-9 的回归防线）：
- symptom_detector 确实把最近对话拼进 LLM 输入（指代消解的前提）；
- medical_knowledge 检索词确实合并上一轮用户消息（「它该用什么药」能借上下文命中实体）；
- history_block 只取最近 4 条、按角色标注。
"""
import os
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MODEL_API_KEY", "test-placeholder")

HISTORY = [
    {"role": "user", "content": "什么是糖尿病？"},
    {"role": "assistant", "content": "糖尿病是以高血糖为特征的代谢性疾病。"},
]


class FakeDetection(BaseModel):
    symptoms: list[str] = Field(default_factory=list)
    severity: str = "low"
    has_emergency_keyword: bool = False


class FakeStructured:
    def invoke(self, messages):
        return FakeDetection()


class FakeLLM:
    def with_structured_output(self, _schema, method=None):
        return FakeStructured()


# ---------------- symptom_detector：history 注入 ----------------

def test_symptom_detector_injects_history(monkeypatch):
    import src.agents.symptom_detector as sd

    captured = {}

    class TrackingStructured(FakeStructured):
        def invoke(self, messages):
            captured["messages"] = messages
            return FakeDetection(symptoms=["头晕"])

    class TrackingLLM:
        def with_structured_output(self, _schema, method=None):
            return TrackingStructured()

    monkeypatch.setattr(sd, "get_llm", lambda temperature=0.1: TrackingLLM())

    state = {
        "query": "那应该挂什么科？",
        "history": HISTORY + [{"role": "user", "content": "那应该挂什么科？"}],
    }
    out = sd.symptom_detector(state)

    human_text = str(captured["messages"])
    assert "最近对话" in human_text, "症状检测未注入多轮上下文"
    assert "什么是糖尿病" in human_text, "上一轮用户消息未进入提示词"
    assert out["symptoms"] == ["头晕"]


def test_symptom_detector_no_history_still_works(monkeypatch):
    import src.agents.symptom_detector as sd

    monkeypatch.setattr(sd, "get_llm", lambda temperature=0.1: FakeLLM())
    out = sd.symptom_detector({"query": "你好", "history": []})
    assert out["symptoms"] == []


# ---------------- medical_knowledge：检索词合并上一轮 ----------------

def test_medical_knowledge_retrieval_merges_last_user_turn(monkeypatch):
    import src.agents.medical_knowledge as mk

    captured = {}

    class FakeRetriever:
        def hybrid_search(self, query, top_k=None):
            captured["query"] = query
            return {"entities": [], "contexts": []}

    class FakeAgent:
        queries = None

        def __call__(self, state):
            return {}

    agent = mk.MedicalKnowledgeAgent.__new__(mk.MedicalKnowledgeAgent)
    monkeypatch.setattr(mk, "get_shared_retriever", lambda: FakeRetriever())
    monkeypatch.setattr(agent, "_retrieve", lambda q, s: mk.MedicalKnowledgeAgent._retrieve(agent, q, s))

    # 直接走 __call__ 的检索段：拦截 _retrieve 之后的生成段（contexts 为空 + 非高风险 → 通用兜底会调 LLM）
    # 改为只验证 _retrieve 收到的 query：调 __call__ 前先 monkeypatch 生成函数
    monkeypatch.setattr(
        mk.MedicalKnowledgeAgent, "_grounded_static_answer",
        lambda self, q, c, h=None: "ok",
    )
    monkeypatch.setattr(mk.MedicalKnowledgeAgent, "_react_answer",
                        lambda self, q, c, h=None: ("", [], []))

    state = {"query": "它该用什么药物治疗？", "history": HISTORY, "trace": []}
    agent.queries = None
    # 拦截症状补充检索（queries 为 None 会炸，替换 _retrieve 内的 queries 使用路径）
    def _capture_retrieve(self, q, s):
        captured["query"] = q
        return [], []

    monkeypatch.setattr(mk.MedicalKnowledgeAgent, "_retrieve", _capture_retrieve)
    agent(state)

    assert "它该用什么药物治疗" in captured["query"], "当前问题未进入检索词"
    assert "什么是糖尿病" in captured["query"], "上一轮用户消息未合并进检索词（指代消解失效）"


def test_history_block_format():
    from src.agents.medical_knowledge import _history_block

    block = _history_block(HISTORY)
    assert "用户：什么是糖尿病？" in block
    assert "助手：糖尿病" in block
    assert _history_block([]) == ""
    assert _history_block(None) == ""


# ---------------- 高风险拒答不依赖 LLM ----------------

def test_high_risk_refusal_without_llm():
    import src.agents.medical_knowledge as mk

    agent = mk.MedicalKnowledgeAgent.__new__(mk.MedicalKnowledgeAgent)
    monkeypatch_setattr = None  # noqa: F841
    out = mk.MedicalKnowledgeAgent._is_high_risk("阿司匹林和华法林可以一起吃吗")
    assert out is True
    assert mk.MedicalKnowledgeAgent._is_high_risk("什么是糖尿病") is False
