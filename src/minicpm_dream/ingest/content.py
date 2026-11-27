"""Flatten the message content shapes different harnesses write to disk."""

from __future__ import annotations

from typing import Any

from ..util.text import normalize_text


def flatten_block(block: Any) -> tuple[str, str]:
    """Return (text, kind) for one content block.

    kind is "text" for prose, "tool_result" for tool output, "tool_use" for a
    tool call, or "" when the block carries nothing useful.
    """
    if isinstance(block, str):
        return normalize_text(block), "text"
    if not isinstance(block, dict):
        return "", ""
    kind = str(block.get("type") or "")
    if kind == "text":
        return normalize_text(str(block.get("text") or "")), "text"
    if kind in {"tool_result", "function_result"}:
        inner = block.get("content")
        if isinstance(inner, list):
            parts = [flatten_block(part)[0] for part in inner]
            return normalize_text(" ".join(part for part in parts if part)), "tool_result"
        return normalize_text(str(inner or "")), "tool_result"
    if kind in {"tool_use", "function_call"}:
        # A tool call is intent, not evidence; the result is what carries content.
        return "", "tool_use"
    if kind == "thinking":
        return "", ""
    # Unknown block shape: keep any obvious text field rather than dropping it.
    for key in ("text", "content", "output"):
        value = block.get(key)
        if isinstance(value, str) and value.strip():
            return normalize_text(value), kind or "text"
    return "", ""


def flatten_content(content: Any) -> tuple[str, str]:
    """Flatten a message's content into (text, dominant_kind)."""
    if content is None:
        return "", ""
    if isinstance(content, str):
        return normalize_text(content), "text"
    if isinstance(content, dict):
        return flatten_block(content)
    if not isinstance(content, list):
        return normalize_text(str(content)), "text"
    texts: list[str] = []
    kinds: list[str] = []
    for block in content:
        text, kind = flatten_block(block)
        if text:
            texts.append(text)
            kinds.append(kind)
    if not texts:
        return "", ""
    dominant = "tool_result" if kinds and all(kind == "tool_result" for kind in kinds) else "text"
    return normalize_text(" ".join(texts)), dominant
