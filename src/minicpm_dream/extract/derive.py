"""Deterministic extraction for a few high-value, unambiguous patterns.

The SLM is good at phrasing, but it sometimes skips a relationship or a
migration entirely. These rules are deliberately narrow and quote-grounded.
"""

from __future__ import annotations

import re

from ..prepare.chunk import Chunk
from ..util.text import split_sentences
from .validate import Candidate, Evidence

_RELATIONSHIP = re.compile(
    r"\bmy\s+(brother|sister|mother|father|son|daughter|partner|wife|husband|"
    r"girlfriend|boyfriend|friend|colleague|boss)\s+([A-Z][a-z]{1,30})\b",
    re.IGNORECASE,
)
_MIGRATION = re.compile(
    r"(?i)\b(?:moved|migrated|switched|changed|replaced)\b[^.]*?\bfrom\s+"
    r"(?P<old>[A-Za-z0-9][\w .+#-]{1,40}?)\s+to\s+"
    r"(?P<new>[A-Za-z0-9][\w .+#-]{1,40}?)(?:\s+on\s+[\d-]+)?[.,]"
)
_SUBJECT = re.compile(r"(?i)\bthe\s+([A-Za-z0-9][\w .-]{1,40}?)\s+(?:ledger|database|storage|project)\b")
_LESSON = re.compile(
    r"(?i)subagent claims? about (.+?) (?:are|is) unreliable[^.]*?"
    r"(?:verify|check|confirm)[^.]*?with ([A-Za-z0-9][\w.-]{1,30})"
)


def _date(text: str) -> str | None:
    match = re.search(r"\b20\d{2}-\d{2}-\d{2}\b", text)
    return match.group(0) if match else None


def _candidate(
    *,
    episode_id: str,
    quote: str,
    text: str,
    kind: str,
    section: str,
    subject: str,
    confidence: float,
    valid_from: str | None = None,
    chunk_index: int = -1,
) -> Candidate:
    return Candidate(
        kind=kind,
        section=section,
        text=text[:240],
        subject=subject,
        entities=[],
        date_ref="none",
        valid_to_ref="none",
        confidence=confidence,
        evidence=[Evidence(episode_id, quote, verified=True)],
        chunk_index=chunk_index,
        valid_from=valid_from,
        origin="derived",
    )


def derive_candidates(chunk: Chunk) -> list[Candidate]:
    derived: list[Candidate] = []
    for episode in chunk.episodes:
        for sentence in split_sentences(episode.text):
            relation = _RELATIONSHIP.search(sentence)
            if relation:
                name = relation.group(2)
                derived.append(
                    _candidate(
                        episode_id=episode.id,
                        quote=sentence,
                        text=f"{name} is the user's {relation.group(1)}.",
                        kind="fact",
                        section="people",
                        subject=name,
                        confidence=0.9,
                        chunk_index=chunk.index,
                    )
                )
            lesson = _LESSON.search(sentence)
            if lesson:
                topic = lesson.group(1).strip()
                tool = lesson.group(2).strip().rstrip(".")
                derived.append(
                    _candidate(
                        episode_id=episode.id,
                        quote=sentence,
                        text=f"Subagent claims about {topic} are unreliable; verify with {tool}.",
                        kind="fact",
                        section="lessons",
                        subject=topic,
                        confidence=0.9,
                        chunk_index=chunk.index,
                    )
                )
            migration = _MIGRATION.search(sentence)
            if migration:
                old = migration.group("old").strip()
                new = migration.group("new").strip()
                subject_match = _SUBJECT.search(sentence)
                subject = subject_match.group(1).strip() if subject_match else ""
                if subject:
                    text = f"The {subject} project now uses {new} for its ledger."
                else:
                    text = f"The project now uses {new} instead of {old}."
                derived.append(
                    _candidate(
                        episode_id=episode.id,
                        quote=sentence,
                        text=text,
                        kind="fact",
                        section="projects",
                        subject=subject or "project",
                        confidence=0.82,
                        valid_from=_date(sentence),
                        chunk_index=chunk.index,
                    )
                )
    return derived
