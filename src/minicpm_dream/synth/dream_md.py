"""Human-readable dream diary (DREAMS.md)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..util.atomic import write_text_atomic
from ..util.text import truncate
from ..util.timeutil import iso_utc, utc_now

HEADER = """# Dreams

Nightly memory consolidation for an AI agent. Each section is one run:
what was promoted, what changed, what still needs a human, and what the
safety gates did. Machine state lives in the SQLite ledger; this file is
for reading.
"""


def _evidence(evidence: list[dict[str, Any]]) -> str:
    if not evidence:
        return ""
    ev = evidence[0]
    return f"  \n  ↳ “{truncate(str(ev.get('quote', '')), 160)}” (`{ev.get('episode_id', '')}`)"


def render_section(
    *,
    started_at: str,
    window: str,
    run_id: str,
    model: str,
    counts: dict[str, Any],
    promoted: list[dict[str, Any]],
    changed: list[dict[str, Any]],
    needs_review: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    safety: dict[str, Any],
) -> str:
    lines = [f"## {started_at} — {window}", ""]
    lines.append(
        f"Run `{run_id}` · model `{model}` · {counts.get('episodes_kept', 0)} episodes → "
        f"{counts.get('candidates', 0)} candidates → {counts.get('verified', 0)} verified · "
        f"{counts.get('applied', 0)} changes · {counts.get('duration_s', 0):.1f}s"
    )
    lines.append("")

    lines.append("### Promoted")
    if promoted:
        for item in promoted:
            lines.append(
                f"- `{item.get('memory_id', '')}` **{item.get('section', '')}** — "
                f"{item.get('text', '')}{_evidence(item.get('evidence', []))}"
            )
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("### Changed")
    if changed:
        for item in changed:
            targets = ", ".join(f"`{t}`" for t in item.get("targets", [])) or "—"
            lines.append(f"- **{item.get('op', '')}** {targets} — {item.get('text', '')}")
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("### Needs review")
    if needs_review:
        for item in needs_review:
            reason = item.get("reason") or item.get("rationale") or "uncertain"
            lines.append(f"- `{item.get('proposal_id', '')}` — {item.get('text', '')} ({reason})")
    else:
        lines.append("- (none)")
    lines.append("")

    if skipped:
        lines.append("### Skipped")
        for item in skipped:
            lines.append(f"- {item.get('target', '')}: {truncate(str(item.get('text', '')), 200)} "
                         f"({item.get('reason', '')})")
        lines.append("")

    lines.append("### Safety")
    loss = safety.get("loss") or {}
    if loss:
        lines.append(
            f"- loss guard: {'ok' if loss.get('ok') else 'TRIPPED'} "
            f"({loss.get('before_entries')}→{loss.get('after_entries')} entries, "
            f"{loss.get('before_chars')}→{loss.get('after_chars')} chars)"
        )
    filters = safety.get("filters") or {}
    if filters:
        lines.append(
            f"- ingest: kept {filters.get('kept', 0)}/{filters.get('seen', 0)}, "
            f"redactions {filters.get('redactions', 0)}, "
            f"quarantined {filters.get('dropped_injection', 0)}, "
            f"duplicates {filters.get('dropped_duplicate', 0)}"
        )
    if safety.get("notes"):
        for note in safety["notes"]:
            lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def append_section(path: str | Path, section: str, *, keep_sections: int = 90) -> Path:
    p = Path(path).expanduser()
    existing = p.read_text(encoding="utf-8") if p.is_file() else HEADER
    if not existing.startswith("# Dreams"):
        existing = HEADER + "\n" + existing
    parts = existing.split("\n## ")
    head = parts[0].rstrip() + "\n\n"
    sections = parts[1:]
    sections.append(section.lstrip("\n"))
    if keep_sections > 0:
        sections = sections[-keep_sections:]
    text = head + "\n## ".join(sections)
    if not text.endswith("\n"):
        text += "\n"
    p.parent.mkdir(parents=True, exist_ok=True)
    return write_text_atomic(p, text)


def quiet_section(*, started_at: str, window: str, run_id: str, counters: dict[str, Any]) -> str:
    return (
        f"## {started_at} — {window}\n\n"
        f"Run `{run_id}` · quiet pass: no durable memories promoted. "
        f"({counters.get('episodes_kept', 0)} episodes, {counters.get('candidates', 0)} candidates)\n"
    )


def generated_at() -> str:
    return iso_utc(utc_now())
