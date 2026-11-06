"""Apply reviewed actions to Hermes MEMORY.md / USER.md safely."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import Config
from ..util.atomic import ensure_dir, write_text_atomic
from ..util.locking import FileLock
from ..util.logging import get_logger
from .exports import read_entries, render_entries

log = get_logger("apply.hermes")


@dataclass
class ApplyReport:
    target: str
    path: str
    added: int = 0
    replaced: int = 0
    removed: int = 0
    chars_before: int = 0
    chars_after: int = 0
    backup: str = ""
    error: str = ""
    ok: bool = True
    fields_applied: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "path": self.path,
            "added": self.added,
            "replaced": self.replaced,
            "removed": self.removed,
            "chars_before": self.chars_before,
            "chars_after": self.chars_after,
            "backup": self.backup,
            "error": self.error,
            "ok": self.ok,
        }


def _match_indices(entries: list[str], old_text: str) -> list[int]:
    return [i for i, entry in enumerate(entries) if old_text and old_text in entry]


def apply_actions(
    path: str | Path,
    actions: list[dict[str, Any]],
    *,
    target: str,
    limit: int,
    backup_dir: str | Path,
) -> ApplyReport:
    p = Path(path).expanduser()
    report = ApplyReport(target=target, path=str(p))
    if not p.is_file():
        report.ok = False
        report.error = f"{p} does not exist"
        return report
    lock_path = p.with_name(p.name + ".lock")
    with FileLock(lock_path, timeout=10):
        before_mtime = p.stat().st_mtime_ns
        before_text = p.read_text(encoding="utf-8")
        entries = read_entries(p)
        report.chars_before = len(before_text)
        try:
            for action in actions:
                kind = str(action.get("action", "")).lower()
                content = str(action.get("content", "")).strip()
                old_text = str(action.get("old_text", ""))
                if kind == "add":
                    if content and content not in entries:
                        entries.append(content)
                        report.added += 1
                elif kind == "replace":
                    matches = _match_indices(entries, old_text)
                    if len(matches) != 1:
                        raise ValueError(
                            f"replace anchor matched {len(matches)} entries (needs exactly 1): {old_text!r}"
                        )
                    entries[matches[0]] = content
                    report.replaced += 1
                elif kind == "remove":
                    matches = _match_indices(entries, old_text)
                    if len(matches) != 1:
                        raise ValueError(
                            f"remove anchor matched {len(matches)} entries (needs exactly 1): {old_text!r}"
                        )
                    entries.pop(matches[0])
                    report.removed += 1
                else:
                    raise ValueError(f"unknown action {kind!r}")
            rendered = render_entries(entries)
            if len(rendered) > limit:
                raise ValueError(
                    f"result is {len(rendered)} chars, over the {limit}-char Hermes limit"
                )
            if p.stat().st_mtime_ns != before_mtime:
                raise ValueError("file changed while reviewing; aborting to avoid clobbering a concurrent write")
            backup = ensure_dir(backup_dir) / f"{p.name}.{datetime.now(UTC):%Y%m%dT%H%M%SZ}.bak"
            shutil.copy2(p, backup)
            report.backup = str(backup)
            write_text_atomic(p, rendered, mode=0o600)
            report.chars_after = len(rendered)
        except Exception as exc:  # noqa: BLE001
            report.ok = False
            report.error = str(exc)
            log.warning("Hermes memory write refused: %s", exc)
    return report


def apply_review_actions(
    cfg: Config,
    actions: list[dict[str, Any]],
    *,
    memory_limit: int | None = None,
    user_limit: int | None = None,
) -> list[ApplyReport]:
    grouped: dict[str, list[dict[str, Any]]] = {"memory": [], "user": []}
    for action in actions:
        target = str(action.get("target", "memory"))
        if target not in grouped:
            target = "memory"
        grouped[target].append(action)
    paths = {
        "memory": (Path(cfg.hermes.home) / cfg.hermes.memory_file, memory_limit or cfg.hermes.memory_char_limit),
        "user": (Path(cfg.hermes.home) / cfg.hermes.user_file, user_limit or cfg.hermes.user_char_limit),
    }
    reports: list[ApplyReport] = []
    for target, target_actions in grouped.items():
        if not target_actions:
            continue
        path, limit = paths[target]
        reports.append(
            apply_actions(
                path,
                target_actions,
                target=target,
                limit=limit,
                backup_dir=cfg.paths.backups,
            )
        )
    return reports


def as_memory_tool_calls(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ready-to-run Hermes `memory` tool calls for an agent review turn."""
    calls = []
    for action in actions:
        call = {
            "tool": "memory",
            "action": action.get("action"),
            "target": action.get("target", "memory"),
        }
        if action.get("action") == "add":
            call["content"] = action.get("content")
        elif action.get("action") == "replace":
            call["old_text"] = action.get("old_text")
            call["content"] = action.get("content")
        elif action.get("action") == "remove":
            call["old_text"] = action.get("old_text")
        calls.append(call)
    return calls
