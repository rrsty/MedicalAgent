"""Unit tests for AnswerFusion deterministic fallback and incremental assembly (Milestone M3, Feature F14)."""
from unittest.mock import MagicMock, patch
import pytest

from src.agents.answer_fusion import AnswerFusion


def test_fallback_medication_with_knowledge_answer():
    """Medication intent with empty medications but valid knowledge answer.
    
    Must retain knowledge_answer in final answer and NOT fallback to generic failure message.
    """
    state = {
        "intent": "medication",
        "medications": [],
        "knowledge_answer": "阿莫西林属于广谱青霉素类抗生素，用于敏感菌所致的呼吸道感染等。",
        "disclaimers": ["本建议仅供参考，具体用药请遵医嘱。"],
    }
    fallback = AnswerFusion._deterministic_fallback(state)

    assert "阿莫西林属于广谱青霉素类抗生素" in fallback
    assert "抱歉，暂时无法处理您的问题" not in fallback
    assert "---\n本建议仅供参考，具体用药请遵医嘱。" in fallback


def test_fallback_medication_with_drugs_and_knowledge():
    """Medication intent with both medications list and knowledge answer.
    
    Both medication guidance and knowledge answer must be present.
    """
    state = {
        "intent": "medication",
        "medications": [
            {
                "name": "布洛芬缓释胶囊",
                "purpose": "缓解轻至中度疼痛或退热",
                "side_effects": "恶心、呕吐、胃部不适",
            }
        ],
        "knowledge_answer": "服药期间请注意多饮水，并避免饮酒。",
    }
    fallback = AnswerFusion._deterministic_fallback(state)

    assert "**用药参考**：" in fallback
    assert "布洛芬缓释胶囊" in fallback
    assert "缓解轻至中度疼痛或退热" in fallback
    assert "常见副作用：恶心、呕吐、胃部不适" in fallback
    assert "服药期间请注意多饮水，并避免饮酒。" in fallback
    assert "抱歉，暂时无法处理您的问题" not in fallback


def test_fallback_appointment_structure():
    """Appointment intent with symptoms, recommended departments, and advice."""
    state = {
        "intent": "appointment",
        "symptoms": ["头晕", "偏头痛"],
        "departments": ["神经内科", "急诊科"],
        "advice": ["避免剧烈运动", "记录发作时间与频率"],
        "disclaimers": ["建议仅供参考。"],
    }
    fallback = AnswerFusion._deterministic_fallback(state)

    assert "**检测到的症状**：头晕、偏头痛" in fallback
    assert "**推荐就诊科室**：神经内科（第1优先） → 急诊科（第2优先）" in fallback
    assert "**就医前建议**：\n- 避免剧烈运动\n- 记录发作时间与频率" in fallback
    assert "---\n建议仅供参考。" in fallback
    assert "抱歉，暂时无法处理您的问题" not in fallback


def test_fallback_emergency_with_warnings():
    """Emergency state with warnings prepended at top and disclaimers at bottom."""
    state = {
        "intent": "emergency",
        "warnings": ["⚠️ 疑似急性心肌梗死高危风险，请立即拨打 120 急救电话！"],
        "disclaimers": ["紧急情况请第一时间送医，本系统不承担急救责任。"],
    }
    fallback = AnswerFusion._deterministic_fallback(state)

    assert fallback.startswith("⚠️ 疑似急性心肌梗死高危风险")
    assert "---\n紧急情况请第一时间送医" in fallback
    assert "抱歉，暂时无法处理您的问题" not in fallback


def test_fallback_all_empty_gives_graceful_message():
    """Completely empty state falls back to generic helpful message."""
    state = {}
    fallback = AnswerFusion._deterministic_fallback(state)

    assert fallback == "抱歉，暂时无法处理您的问题，请尝试换一种提问方式。"


def test_fallback_empty_with_disclaimer_only():
    """State with only disclaimers includes fallback message and disclaimer."""
    state = {
        "disclaimers": ["本内容由 AI 生成。"],
    }
    fallback = AnswerFusion._deterministic_fallback(state)

    assert "抱歉，暂时无法处理您的问题，请尝试换一种提问方式。" in fallback
    assert "---\n本内容由 AI 生成。" in fallback


def test_fusion_call_llm_exception_triggers_fallback():
    """AnswerFusion.__call__ gracefully catches LLM exception and invokes deterministic fallback."""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = RuntimeError("OpenAI API rate limit / connection timeout")

    fusion = AnswerFusion()
    state = {
        "intent": "medication",
        "medications": [],
        "knowledge_answer": "阿莫西林为常见青霉素类抗生素。",
        "disclaimers": ["用药前请咨询医师。"],
    }

    with patch("src.agents.answer_fusion.get_llm", return_value=mock_llm):
        result = fusion(state)

    assert "final_answer" in result
    answer = result["final_answer"]
    assert "阿莫西林为常见青霉素类抗生素。" in answer
    assert "用药前请咨询医师。" in answer
    assert "抱歉，暂时无法处理您的问题" not in answer


def test_fusion_call_llm_empty_triggers_fallback():
    """AnswerFusion.__call__ handles empty content from LLM and falls back."""
    mock_resp = MagicMock()
    mock_resp.content = "   "
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_resp

    fusion = AnswerFusion()
    state = {
        "intent": "appointment",
        "symptoms": ["咳嗽"],
        "departments": ["呼吸内科"],
        "advice": ["清淡饮食"],
    }

    with patch("src.agents.answer_fusion.get_llm", return_value=mock_llm):
        result = fusion(state)

    assert "final_answer" in result
    answer = result["final_answer"]
    assert "**检测到的症状**：咳嗽" in answer
    assert "呼吸内科" in answer


def test_fusion_call_emergency_bypasses_llm():
    """Emergency intent directly runs deterministic fallback without invoking LLM."""
    mock_llm = MagicMock()
    fusion = AnswerFusion()
    state = {
        "intent": "emergency",
        "warnings": ["⚠️ 严重过敏性休克警告，请立即就医！"],
    }

    with patch("src.agents.answer_fusion.get_llm", return_value=mock_llm):
        result = fusion(state)

    assert not mock_llm.invoke.called
    assert "⚠️ 严重过敏性休克警告" in result["final_answer"]
