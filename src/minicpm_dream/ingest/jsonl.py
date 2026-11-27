"""Generic JSONL transcript adapter.

Each line may look like:
  {"session_id": "...", "role": "user", "timestamp": "2026-11-05T14:03:00+10:00",
   "content": "...", "source": "custom", "tz": "Australia/Brisbane"}

OpenAI/Anthropic-style records with a nested ``message`` and a content-block
list are accepted too, so an existing export usually works unmodified.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..util.text import normalize_text
from .base import Episode, make_episode
from .content import flatten_content


def _record_text(record: dict[str, Any]) -> tuple[Any, str, str]:
    """Return (text, role, kind) from either flat or nested message records."""
    message = record.get("message")
    role = str(record.get("role") or "")
    timestamp = record.get("timestamp") or record.get("ts") or record.get("time") or record.get("created_at")
    raw: Any = record.get("text")
    if raw is None:
        raw = record.get("content")
    if raw is None and isinstance(message, dict):
        raw = message.get("content")
        role = role or str(message.get("role") or "")
        timestamp = timestamp or message.get("timestamp")
    if isinstance(raw, str):
        return timestamp, normalize_text(raw), role
    text, kind = flatten_content(raw)
    return timestamp, text, role


def read_jsonl(
    path: str | Path,
    *,
    tz: str,
    default_source: str = "jsonl",
    trust_tier: str = "trusted",
) -> Iterator[Episode]:
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                continue
            timestamp, text, role = _record_text(record)
            if not text:
                continue
            if timestamp is None:
                raise ValueError(f"{path}:{line_no}: missing timestamp")
            yield make_episode(
                source=str(record.get("source") or default_source),
                session_id=str(
                    record.get("session_id") or record.get("conversation_id") or record.get("sessionId") or "jsonl"
                ),
                role=role or "user",
                timestamp=timestamp,
                text=text,
                tz=str(record.get("tz") or tz),
                message_id=record.get("id") or record.get("message_id") or line_no,
                trust_tier=str(record.get("trust_tier") or trust_tier),
                meta={k: record[k] for k in ("chat_type", "chat_id", "title") if k in record},
            )
