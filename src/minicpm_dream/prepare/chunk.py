"""Token-budgeted chunking that never splits a turn."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..config import ExtractCfg, IngestCfg
from ..ingest.base import Episode
from ..util.text import truncate
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

    for ep in episodes:
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
