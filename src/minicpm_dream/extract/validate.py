"""Candidate model and deterministic post-validation of extractor output."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..ingest.base import Episode
from ..ingest.redact import has_secret
from ..prepare.chunk import Chunk
from ..prepare.timeanchor import has_relative_time
from ..util.ids import stable_hash, stable_id
from ..util.text import (
    is_probably_meta,
    is_retraction,
    is_status_echo,
    normalize_text,
    quote_in_source,
)

KINDS = {"fact", "event", "preference", "task", "decision", "self", "question"}
_TASK_CUES = re.compile(
    r"(?i)\b(remind me|appointment|deadline|todo|to-do|follow[- ]?up|due date|"
    r"schedule[sd]?|meeting|call at|at \d{1,2}(?::\d{2})?\s?(?:am|pm))\b"
)

_SECTION_FOR_KIND = {
    "task": "open_loops",
    "event": "open_loops",
    "question": "open_loops",
}

SECTIONS = {
    "identity",
    "preferences",
    "projects",
    "people",
    "environment",
    "procedures",
    "lessons",
    "open_loops",
}

_TRAILING_REL = re.compile(
    r"\s*\(?(?:on\s+)?(?:today|yesterday|tomorrow|last night|last week|last month|last year|"
    r"\d+\s+(?:day|week|month|year)s?\s+ago)\)?\.?$",
    re.IGNORECASE,
)


@dataclass
class Evidence:
    episode_id: str
    quote: str
    char_start: int | None = None
    char_end: int | None = None
    verified: bool = False


@dataclass
class Candidate:
    kind: str
    section: str
    text: str
    subject: str
    entities: list[str]
    date_ref: str
    valid_to_ref: str
    confidence: float
    evidence: list[Evidence]
    chunk_index: int
    prompt_version: str = "extract@1"
    valid_from: str | None = None
    valid_to: str | None = None
    origin: str = "slm"
    rejected_reasons: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return stable_id("cand", self.kind, self.text, self.evidence[0].episode_id if self.evidence else "")

    @property
    def content_hash(self) -> str:
        return stable_hash(self.kind, self.text, self.valid_from or "")


def normalize_kind(kind: str, text: str) -> str:
    """Strong reminder/appointment cues win over a model's 'fact'/'preference'."""
    if kind in {"fact", "preference", "event"} and _TASK_CUES.search(text):
        return "task"
    return kind


def normalize_section(kind: str, section: str, text: str = "") -> str:
    """Keep the model's section only where it makes sense for the kind."""
    forced = _SECTION_FOR_KIND.get(kind)
    if forced:
        return forced
    if kind == "self":
        return section if section in {"procedures", "lessons"} else "lessons"
    if kind == "decision":
        return section if section in {"projects", "procedures", "lessons", "open_loops"} else "projects"
    low = text.casefold()
    if kind in {"fact", "preference"}:
        if re.search(r"\b(cron|script|command|config|must live|have to live|workflow)\b", low):
            return "procedures"
        if re.search(r"\b(unreliable|verify|pitfall|lesson|subagent claims)\b", low):
            return "lessons"
        if re.search(r"\b(favou?rite|prefers?|likes?|dislikes?|hates?)\b", low):
            return "preferences"
        if "~/." in text or re.search(r"\b(ubuntu|optiplex|homelab|ram|server|host)\b", low):
            return "environment"
        if re.search(r"\b(project|ledger|database|postgres|sqlite|minicpm|python)\b", low):
            return "projects"
    if kind == "preference":
        return "identity" if section == "identity" else "preferences"
    if section in {"preferences", "projects", "environment", "procedures"}:
        if re.search(r"\b(cron|script|command|config|cli|workflow|must live|have to live)\b", low):
            return "procedures"
        if "~/." in text or re.search(r"\b(ubuntu|optiplex|homelab|ram|host|server)\b", low):
            return "environment"
        if re.search(r"\b(project|ledger|database|postgres|sqlite|miniCPM|python)\b", low):
            return "projects"
    return section if section in SECTIONS else "projects"


def _str(value: Any, *, limit: int = 400) -> str:
    if value is None:
        return ""
    if not isinstance(value, (str, int, float)):
        return ""
    return normalize_text(str(value))[:limit]


def _find_span(quote: str, source: str) -> tuple[int | None, int | None]:
    q = normalize_text(quote).casefold()
    s = normalize_text(source)
    idx = s.casefold().find(q)
    if idx < 0:
        return None, None
    return idx, idx + len(q)


def resolve_date_refs(candidate: Candidate, chunk: Chunk) -> tuple[str | None, str | None, list[str]]:
    """Return (valid_from, valid_to, errors) using only code-resolved refs."""
    refs = {ref.id: ref for ref in chunk.date_refs}
    valid_from = refs[candidate.date_ref].iso if candidate.date_ref in refs else None
    valid_to = refs[candidate.valid_to_ref].iso if candidate.valid_to_ref in refs else None
    errors: list[str] = []
    if candidate.date_ref not in {"none", *refs}:
        errors.append(f"unknown date_ref {candidate.date_ref}")
    if candidate.valid_to_ref not in {"none", *refs}:
        errors.append(f"unknown valid_to_ref {candidate.valid_to_ref}")
    if valid_to and valid_from and valid_to < valid_from:
        errors.append("valid_to precedes valid_from")
    return valid_from, valid_to, errors


def apply_date_text(candidate: Candidate, chunk: Chunk) -> str:
    """Insert the code-resolved date and strip relative-time phrases from text."""
    refs = {ref.id: ref for ref in chunk.date_refs}
    text = normalize_text(candidate.text)
    if candidate.date_ref in refs:
        ref = refs[candidate.date_ref]
        pattern = re.compile(re.escape(ref.phrase), re.IGNORECASE)
        if ref.phrase == ref.iso and ref.iso in text:
            pass  # already an absolute date; do not prefix a second "on"
        elif pattern.search(text):
            text = pattern.sub(f"on {ref.iso}", text)
        elif any(ev.quote and pattern.search(ev.quote) for ev in candidate.evidence):
            if ref.iso not in text:
                match = re.match(r"^(.*?)([.!?])$", text)
                if match:
                    text = f"{match.group(1)} (on {ref.iso}){match.group(2)}"
                else:
                    text = f"{text} (on {ref.iso})"
    text = _TRAILING_REL.sub("", text).strip()
    return text


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            out.append(value)
    return out


def validate_item(item: dict[str, Any], chunk: Chunk) -> tuple[Candidate | None, list[str]]:
    errors: list[str] = []
    if not isinstance(item, dict):
        return None, ["item is not an object"]
    kind = _str(item.get("kind"), limit=20).lower()
    section = _str(item.get("section"), limit=20).lower()
    if kind not in KINDS:
        errors.append(f"invalid kind {kind!r}")
    if section not in SECTIONS:
        errors.append(f"invalid section {section!r}")
    kind = normalize_kind(kind, _str(item.get("text"), limit=240))
    section = normalize_section(kind, section, _str(item.get("text"), limit=240))

    raw_text = item.get("text")
    if not isinstance(raw_text, str) or len(normalize_text(raw_text)) > 240:
        errors.append("text missing or over 240 chars")
    text = _str(raw_text, limit=240)
    if len(text) < 3:
        errors.append("text too short")
    if is_probably_meta(text):
        errors.append("conversational meta-state")
    if is_retraction(text):
        errors.append("retraction is not a new fact")
    if is_status_echo(text):
        errors.append("status echo is not a durable fact")
    if has_secret(text):
        errors.append("secret-shaped text")

    subject = _str(item.get("subject"), limit=60)
    entities_raw = item.get("entities") or []
    entities: list[str] = []
    if isinstance(entities_raw, list):
        chunk_lower = chunk.text.casefold()
        for ent in entities_raw[:5]:
            ent_text = _str(ent, limit=40)
            if len(ent_text) >= 2 and ent_text.casefold() in chunk_lower:
                entities.append(ent_text)
    entities = _dedupe(entities)

    date_ref = _str(item.get("date_ref"), limit=8) or "none"
    try:
        confidence = float(item.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(1.0, max(0.0, confidence))

    raw_evidence = item.get("evidence") or []
    if isinstance(raw_evidence, str):
        raw_evidence = [raw_evidence]
    item_episode = _str(item.get("episode_id"), limit=40)
    normalized_evidence: list[dict[str, Any]] = []
    if isinstance(raw_evidence, list):
        for ev in raw_evidence:
            if isinstance(ev, str):
                normalized_evidence.append({"episode_id": item_episode, "quote": ev})
            elif isinstance(ev, dict):
                entry = dict(ev)
                entry.setdefault("episode_id", item_episode)
                if "quote" not in entry and "text" in entry:
                    entry["quote"] = entry["text"]
                normalized_evidence.append(entry)
    raw_evidence = normalized_evidence
    evidence: list[Evidence] = []
    episodes: dict[str, Episode] = {ep.id: ep for ep in chunk.episodes}
    if not isinstance(raw_evidence, list) or not raw_evidence:
        errors.append("missing evidence")
    else:
        for ev in raw_evidence[:2]:
            if not isinstance(ev, dict):
                errors.append("evidence is not an object")
                continue
            ep_id = _str(ev.get("episode_id"), limit=40)
            quote = _str(ev.get("quote"), limit=200)
            episode = episodes.get(ep_id)
            if episode is None and quote:
                matches = [
                    candidate_ep
                    for candidate_ep in chunk.episodes
                    if quote_in_source(quote, candidate_ep.text)
                ]
                if len(matches) == 1:
                    episode = matches[0]
                    ep_id = episode.id
            if episode is None:
                errors.append(f"evidence episode {ep_id!r} not in chunk and quote is not verbatim")
                continue
            if len(quote) < 8:
                errors.append("evidence quote too short")
                continue
            if not quote_in_source(quote, episode.text):
                errors.append("evidence quote is not verbatim")
                continue
            start, end = _find_span(quote, episode.text)
            evidence.append(Evidence(ep_id, quote, start, end, verified=True))

    if errors:
        return None, errors

    haystack = f"{text} {evidence[0].quote if evidence else ''}".casefold()
    if date_ref == "none":
        date_matches = [ref for ref in chunk.date_refs if ref.phrase.casefold() in haystack]
        if len(date_matches) == 1:
            date_ref = date_matches[0].id
    else:
        ref = next((r for r in chunk.date_refs if r.id == date_ref), None)
        if ref is None or ref.phrase.casefold() not in haystack:
            date_ref = "none"
    if has_relative_time(text) and date_ref == "none":
        errors.append("relative time without a resolvable date_ref")
    if errors:
        return None, errors

    candidate = Candidate(
        kind=kind,
        section=section,
        text=text,
        subject=subject,
        entities=entities,
        date_ref=date_ref,
        valid_to_ref="none",
        confidence=confidence,
        evidence=evidence,
        chunk_index=chunk.index,
    )
    valid_from, _, date_errors = resolve_date_refs(candidate, chunk)
    if date_errors:
        return None, date_errors
    if valid_from is None and kind in {"event", "task"} and chunk.episodes:
        valid_from = chunk.episodes[0].spoken_at_local[:10]
    candidate.text = apply_date_text(candidate, chunk)
    explicit_dates = re.findall(r"\b20\d{2}-\d{2}-\d{2}\b", candidate.text)
    if explicit_dates:
        valid_from = explicit_dates[0]
    candidate.valid_from = valid_from
    candidate.valid_to = None
    if len(candidate.text) > 300:
        return None, ["resolved text too long"]
    return candidate, []


def dedupe_candidates(candidates: list[Candidate]) -> list[Candidate]:
    from ..util.text import containment, jaccard

    kept: list[Candidate] = []
    for cand in sorted(candidates, key=lambda c: (len(c.text), c.confidence), reverse=True):
        duplicate = False
        for index, other in enumerate(kept):
            same = cand.text.casefold() == other.text.casefold()
            similar = jaccard(cand.text, other.text) >= 0.92
            contained = (
                containment(cand.text, other.text) >= 0.9
                or containment(other.text, cand.text) >= 0.9
            )
            if same or similar or contained:
                # Deterministic facts win over model phrasing; otherwise keep the
                # richer item. Evidence is merged so provenance is not lost.
                prefer_candidate = cand.origin == "derived" and other.origin != "derived"
                if not prefer_candidate and other.origin == "derived" and cand.origin != "derived":
                    prefer_candidate = False
                elif not prefer_candidate:
                    prefer_candidate = len(cand.text) > len(other.text)
                if prefer_candidate:
                    known = {(e.episode_id, e.quote) for e in cand.evidence}
                    for ev in other.evidence:
                        if (ev.episode_id, ev.quote) not in known:
                            cand.evidence.append(ev)
                    kept[index] = cand
                elif cand.evidence:
                    known = {(e.episode_id, e.quote) for e in other.evidence}
                    for ev in cand.evidence:
                        if (ev.episode_id, ev.quote) not in known:
                            other.evidence.append(ev)
                duplicate = True
                break
        if not duplicate:
            kept.append(cand)
    return kept
