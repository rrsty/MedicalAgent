"""Shared context compaction helpers for Agent prompts.

Conversation storage keeps the full text. Prompt construction uses bounded views
so one unusually long response cannot crowd out the current question or recent
user facts. Long messages retain both their opening and conclusion.
"""

from collections.abc import Iterable

_COMPACTION_MARKER = "\n…[中间内容已压缩]…\n"


def compact_text(text: object, max_chars: int, tail_ratio: float = 0.45) -> str:
    """Bound text while preserving both the beginning and the conclusion."""
    value = str(text or "").strip()
    if len(value) <= max_chars:
        return value
    if max_chars <= len(_COMPACTION_MARKER) + 8:
        return value[:max_chars]

    available = max_chars - len(_COMPACTION_MARKER)
    tail_len = max(1, int(available * tail_ratio))
    head_len = available - tail_len
    return value[:head_len] + _COMPACTION_MARKER + value[-tail_len:]


def build_chat_history(
    history: list[dict] | None,
    *,
    max_messages: int,
    per_message_chars: int,
    total_chars: int,
) -> list[tuple[str, str]]:
    """Build a newest-first-budgeted chat history in chronological order."""
    if not history or max_messages <= 0 or total_chars <= 0:
        return []

    selected = history[-max_messages:]
    remaining = total_chars
    packed_reversed: list[tuple[str, str]] = []

    for turn in reversed(selected):
        if remaining < 24:
            break
        role = "user" if turn.get("role") == "user" else "assistant"
        tail_ratio = 0.35 if role == "user" else 0.55
        content = compact_text(
            turn.get("content", ""),
            min(per_message_chars, remaining),
            tail_ratio=tail_ratio,
        )
        if not content:
            continue
        packed_reversed.append((role, content))
        remaining -= len(content)

    return list(reversed(packed_reversed))


def format_history_block(
    history: list[dict] | None,
    *,
    max_messages: int,
    per_message_chars: int,
    total_chars: int,
    title: str,
) -> str:
    """Format a bounded Chinese conversation block for a single prompt."""
    messages = build_chat_history(
        history,
        max_messages=max_messages,
        per_message_chars=per_message_chars,
        total_chars=total_chars,
    )
    if not messages:
        return ""
    lines = [
        f"{'用户' if role == 'user' else '助手'}：{content}"
        for role, content in messages
    ]
    return title + "\n" + "\n".join(lines) + "\n\n"


def format_context_items(
    contexts: Iterable[object],
    *,
    max_items: int = 10,
    per_item_chars: int = 600,
    total_chars: int = 4800,
) -> str:
    """Pack ranked RAG contexts into a bounded bullet list."""
    lines: list[str] = []
    remaining = total_chars
    for raw in list(contexts)[:max_items]:
        if remaining < 24:
            break
        item = compact_text(raw, min(per_item_chars, remaining), tail_ratio=0.35)
        if not item:
            continue
        lines.append(f"- {item}")
        remaining -= len(item)
    return "\n".join(lines)
