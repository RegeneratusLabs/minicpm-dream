"""Generic JSONL transcript adapter.

Each line may look like:
  {"session_id": "...", "role": "user", "timestamp": "2026-11-05T14:03:00+10:00",
   "content": "...", "source": "custom", "tz": "Australia/Brisbane"}
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from ..util.text import normalize_text
from .base import Episode, make_episode


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
            text = record.get("text") or record.get("content") or ""
            if isinstance(text, (dict, list)):
                text = json.dumps(text, ensure_ascii=False)
            text = normalize_text(str(text))
            if not text:
                continue
            timestamp = (
                record.get("timestamp")
                or record.get("ts")
                or record.get("time")
                or record.get("created_at")
            )
            if timestamp is None:
                raise ValueError(f"{path}:{line_no}: missing timestamp")
            yield make_episode(
                source=str(record.get("source") or default_source),
                session_id=str(record.get("session_id") or record.get("conversation_id") or "jsonl"),
                role=str(record.get("role") or "user"),
                timestamp=timestamp,
                text=text,
                tz=str(record.get("tz") or tz),
                message_id=record.get("id") or record.get("message_id") or line_no,
                trust_tier=str(record.get("trust_tier") or trust_tier),
                meta={k: record[k] for k in ("chat_type", "chat_id", "title") if k in record},
            )
