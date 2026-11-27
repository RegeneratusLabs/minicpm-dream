"""Claude Code transcript adapter.

Claude Code writes one JSONL file per session under ``~/.claude/projects``. Each
line is an event: user/assistant messages, tool results, summaries and sidechain
(subagent) turns. The prose is in ``message.content`` as a string or a list of
blocks, and the useful part of an agent's day is often a tool result rather than
a chat turn, so tool output can be included with ``include_tool_results``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from ..util.logging import get_logger
from ..util.timeutil import parse_timestamp
from .base import Episode, make_episode
from .content import flatten_content

log = get_logger("ingest.claude_code")

_SKIP_PREFIXES = (
    "<local-command-stdout>",
    "<local-command-stderr>",
    "<command-name>",
    "<command-message>",
    "<command-args>",
    "<bash-stdout>",
    "<bash-stderr>",
    "[Request interrupted by user",
    "Caveat: The messages below were generated",
)
_SUMMARY_TYPES = {"summary", "system", "queue-operation", "file-history-snapshot"}


def session_files(root: str | Path) -> list[Path]:
    base = Path(root).expanduser()
    if not base.exists():
        return []
    if base.is_file():
        return [base]
    return sorted(base.glob("**/*.jsonl"))


def _skip_text(text: str) -> bool:
    head = text.lstrip()[:80]
    return any(head.startswith(prefix) for prefix in _SKIP_PREFIXES)


def _message_text(record: dict[str, Any], *, include_tool_results: bool) -> tuple[str, str]:
    message = record.get("message")
    if isinstance(message, dict):
        text, kind = flatten_content(message.get("content"))
        role = str(message.get("role") or record.get("type") or "user")
    else:
        text, kind = flatten_content(record.get("content") or record.get("text"))
        role = str(record.get("type") or "user")
    if kind == "tool_result":
        if not include_tool_results:
            return "", ""
        role = "tool"
    if kind == "tool_use":
        # The call itself is intent, not evidence; keep it out of extraction.
        return "", ""
    if _skip_text(text):
        return "", ""
    return text, role


def iter_claude_code(
    root: str | Path,
    *,
    tz: str,
    include_tool_results: bool = True,
    since: datetime | None = None,
    until: datetime | None = None,
) -> Iterator[Episode]:
    for path in session_files(root):
        try:
            handle = path.open(encoding="utf-8")
        except OSError as exc:
            log.warning("cannot read %s: %s", path, exc)
            continue
        with handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                event_type = str(record.get("type") or "")
                if event_type in _SUMMARY_TYPES:
                    continue
                text, role = _message_text(record, include_tool_results=include_tool_results)
                if not text:
                    continue
                timestamp = (
                    record.get("timestamp")
                    or (record.get("message") or {}).get("timestamp")
                )
                if timestamp is None:
                    continue
                try:
                    spoken_at = parse_timestamp(timestamp, default_tz=tz)
                except (TypeError, ValueError):
                    continue
                if since is not None and spoken_at < since:
                    continue
                if until is not None and spoken_at >= until:
                    continue
                session_id = str(
                    record.get("sessionId")
                    or record.get("session_id")
                    or path.stem
                )
                sidechain = bool(record.get("isSidechain") or record.get("agentId"))
                yield make_episode(
                    source="subagent" if sidechain else "claude-code",
                    session_id=session_id,
                    role=role if role in {"user", "assistant", "tool"} else "user",
                    timestamp=spoken_at,
                    text=text,
                    tz=str(record.get("tz") or tz),
                    message_id=record.get("uuid") or f"{path.name}:{line_no}",
                    trust_tier="trusted",
                    meta={"cwd": record.get("cwd"), "path": str(path)},
                )
