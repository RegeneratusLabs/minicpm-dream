"""Token-budgeted chunking that never splits a turn."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from ..config import ExtractCfg, IngestCfg
from ..ingest.base import Episode
from ..util.text import split_sentences, truncate
from .timeanchor import DateRef, render_refs, resolve_dates


@dataclass
class Chunk:
    index: int
    episodes: list[Episode]
    text: str
    tokens: int
    date_refs: list[DateRef] = field(default_factory=list)

    @property
    def episode_ids(self) -> list[str]:
        return [ep.id for ep in self.episodes]


def estimate_tokens(text: str) -> int:
    """Conservative estimate used only for budgeting (real usage is reported by the backend)."""
    return max(1, math.ceil(len(text) / 3.5))


def render_episode(ep: Episode, *, max_chars: int = 8000) -> str:
    text = truncate(ep.text, max_chars)
    return f"[ep:{ep.id}] {ep.spoken_at_local} {ep.role}: {text}"


def split_episode(ep: Episode, *, budget_tokens: int) -> list[Episode]:
    """Split a long turn into overlapping sentence windows.

    A single 7K-character assistant answer otherwise exceeds the model's useful
    prompt size and is truncated; splitting preserves the tail.
    """
    target_chars = max(600, int(budget_tokens * 3.5 * 0.85))
    if len(ep.text) <= target_chars:
        return [ep]
    sentences = split_sentences(ep.text)
    if not sentences:
        sentences = [ep.text]
    segments: list[str] = []
    current: list[str] = []
    size = 0
    for sentence in sentences:
        addition = len(sentence) + 1
        if current and size + addition > target_chars:
            segments.append(" ".join(current))
            current = [current[-1]] if len(current) > 1 else []
            size = sum(len(item) + 1 for item in current)
        current.append(sentence)
        size += addition
    if current:
        segments.append(" ".join(current))
    out: list[Episode] = []
    for index, segment in enumerate(segments):
        if not segment.strip():
            continue
        out.append(
            replace(
                ep,
                id=f"{ep.id}-p{index}",
                text=segment,
                meta={**ep.meta, "parent_episode_id": ep.id, "segment": index},
            )
        )
    return out or [ep]


def build_chunks(
    episodes: list[Episode],
    *,
    extract: ExtractCfg,
    ingest: IngestCfg,
    tz: str,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    current: list[Episode] = []
    current_tokens = 0
    budget = max(512, extract.max_chunk_tokens)
    overlap_budget = int(budget * max(0.0, min(0.5, extract.overlap_ratio)))

    def flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        rendered = "\n\n".join(render_episode(ep, max_chars=ingest.max_episode_chars) for ep in current)
        refs: list[DateRef] = []
        seen: set[str] = set()
        for ep in current:
            for ref in resolve_dates(ep.text, ep.spoken_at, ep.tz):
                key = f"{ref.iso}|{ref.phrase.lower()}"
                if key in seen:
                    continue
                seen.add(key)
                refs.append(DateRef(id=f"d{len(refs)}", phrase=ref.phrase, iso=ref.iso))
                if len(refs) >= 5:
                    break
            if len(refs) >= 5:
                break
        chunks.append(
            Chunk(
                index=len(chunks),
                episodes=list(current),
                text=rendered,
                tokens=estimate_tokens(rendered),
                date_refs=refs,
            )
        )

    if extract.split_long_messages:
        prepared: list[Episode] = []
        for ep in episodes:
            if estimate_tokens(ep.text) > budget:
                prepared.extend(split_episode(ep, budget_tokens=budget))
            else:
                prepared.append(ep)
        episodes = prepared

    for ep in episodes:
        if current and extract.chunk_by_session and ep.session_id != current[-1].session_id:
            flush()
            current = []
            current_tokens = 0
        rendered = render_episode(ep, max_chars=ingest.max_episode_chars)
        ep_tokens = estimate_tokens(rendered)
        if current and current_tokens + ep_tokens > budget:
            flush()
            # carry a small overlap from the tail of the previous chunk
            carried: list[Episode] = []
            carried_tokens = 0
            for prev in reversed(current):
                prev_tokens = estimate_tokens(render_episode(prev, max_chars=ingest.max_episode_chars))
                if carried_tokens + prev_tokens > overlap_budget:
                    break
                carried.insert(0, prev)
                carried_tokens += prev_tokens
            current = carried
            current_tokens = carried_tokens
        current.append(ep)
        current_tokens += ep_tokens
    flush()
    return chunks


def chunk_context(chunk: Chunk) -> str:
    refs = render_refs(chunk.date_refs)
    return f"DATE REFS:\n{refs}\n\nEPISODES:\n{chunk.text}"
