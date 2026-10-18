"""通用工具单测：文本清理 / LLM JSON 解析 / 截断。"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.utils import (  # noqa: E402
    clean_text,
    extract_json_block,
    parse_llm_json,
    truncate,
)


def test_clean_text():
    # 全角空格/零宽字符直接删除，连续半角空白折叠为一个空格
    assert clean_text("  头痛　头晕 \n") == "头痛头晕"
    assert clean_text("a\u200bb") == "ab"
    assert clean_text(None) == ""


def test_extract_json_block_with_fence():
    text = '以下是结果：\n```json\n{"a": 1}\n```\n希望有帮助'
    assert extract_json_block(text) == '{"a": 1}'


def test_extract_json_block_uppercase_fence():
    text = '```JSON\n{"a": 1}\n```'
    assert extract_json_block(text) == '{"a": 1}'


def test_parse_llm_json_with_noise():
    text = '好的，这是您要的JSON： {"intent": "appointment", "confidence": 0.9} 以上。'
    assert parse_llm_json(text) == {"intent": "appointment", "confidence": 0.9}


def test_parse_llm_json_invalid_returns_none():
    assert parse_llm_json("完全不是JSON") is None
    assert parse_llm_json("") is None


def test_truncate():
    assert truncate("abcdefgh", 5) == "abcde..."
    assert truncate("abc", 5) == "abc"
    assert truncate(None, 5) == ""
