"""Cheap salience gate: skip chunks that cannot contain durable memory.

A nightly pass that spends a model call on every chunk of tool logs is the
reason the pass costs minutes for a day that yields nothing. This gate is
deterministic and conservative: it only skips chunks with no first-person or
durable cues at all, so it cuts the noise tail rather than the signal.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from ..prepare.chunk import Chunk
from ..util.text import tokens

# Cues that a turn can carry something worth remembering. Weights are additive.
_CUES: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"(?i)\b(?:i|i'm|i've|i'd|my|mine|me|we|our|us)\b"), 0.08),
    (re.compile(r"(?i)\b(?:prefer|prefers|like|likes|love|hate|dislike|favourite|favorite|"
                r"always|never|rather)\b"), 0.30),
    (re.compile(r"(?i)\b(?:decided|decision|plan|planning|going to|will|let's|we're moving|"
                r"moved|migrated|switched|switching|chose|choose)\b"), 0.25),
    (re.compile(r"(?i)\b(?:remind|reminder|appointment|deadline|due|tomorrow|tonight|"
                r"next week|next month|follow[- ]?up)\b"), 0.30),
    (re.compile(r"(?i)\b(?:my name|i am|i'm a|based in|live in|lives in|work at|work as|"
                r"i work|my job)\b"), 0.40),
    (re.compile(r"(?i)\b(?:no longer|not anymore|stopped|actually|instead|that's wrong|"
                r"forget that)\b"), 0.30),
    (re.compile(r"(?i)\b(?:brother|sister|mother|father|partner|wife|husband|son|daughter|"
                r"friend|colleague|manager)\b"), 0.25),
    (re.compile(r"(?i)\b(?:uses|use|runs|running|installed|setup|set up|config|version)\b"), 0.12),
    (re.compile(r"(?i)\b(?:remember|note that|keep in mind|important)\b"), 0.20),
]

_NOISE = re.compile(
    r"(?:https?://\S+|www\.\S+|/[\w.-]+/[\w./-]*|[\w.-]+@[\w.-]+\.\w+|\b[0-9a-f]{16,}\b|"
    r"\b[A-Za-z0-9+/]{32,}={0,2}\b)"
)
_CODEISH = re.compile(r"[{};=<>]|^\s*(?:\$|>>>|import |def |class |function )", re.MULTILINE)


@dataclass
class GateStats:
    kept: int = 0
    skipped: int = 0
    skipped_tokens: int = 0
    reasons: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kept": self.kept,
            "skipped": self.skipped,
            "skipped_tokens": self.skipped_tokens,
        }


def salience_score(text: str) -> float:
    """0..1 estimate of how likely this text carries durable memory."""
    if not text or not text.strip():
        return 0.0
    total = 0.0
    for pattern, weight in _CUES:
        if pattern.search(text):
            total += weight
    if total == 0.0:
        return 0.0
    words = max(1, len(tokens(text)))
    density = min(1.0, total / math.sqrt(words) * 3.0)
    noise = len(_NOISE.findall(text)) / max(1.0, words / 20.0)
    code_lines = len(_CODEISH.findall(text))
    penalty = min(0.85, 0.25 * noise + 0.35 * min(1.0, code_lines / max(1, words / 30.0)))
    return round(max(0.0, min(1.0, density * (1.0 - penalty))), 4)


def chunk_score(chunk: Chunk) -> float:
    """A chunk is as salient as its most salient turn, damped by size."""
    if not chunk.episodes:
        return 0.0
    scores = [salience_score(ep.text) for ep in chunk.episodes]
    best = max(scores) if scores else 0.0
    hits = sum(1 for score in scores if score > 0.0)
    return round(best * (0.7 + 0.3 * min(1.0, hits / 3.0)), 4)


def gate_chunks(
    chunks: list[Chunk], *, threshold: float = 0.12, keep_min: int = 1
) -> tuple[list[Chunk], GateStats]:
    """Drop chunks below the threshold, always keeping at least `keep_min`."""
    stats = GateStats()
    scored = [(chunk_score(chunk), chunk) for chunk in chunks]
    kept_indexes = [
        index for index, (score, _chunk) in enumerate(scored) if score >= threshold
    ]
    if len(kept_indexes) < keep_min and chunks:
        order = sorted(range(len(scored)), key=lambda i: scored[i][0], reverse=True)
        kept_indexes = sorted(set(kept_indexes) | set(order[:keep_min]))
    kept_set = set(kept_indexes)
    kept: list[Chunk] = []
    for index, (score, chunk) in enumerate(scored):
        if index in kept_set:
            stats.kept += 1
            kept.append(chunk)
            continue
        stats.skipped += 1
        stats.skipped_tokens += chunk.tokens
        stats.reasons.append(
            {"chunk": chunk.index, "score": score, "tokens": chunk.tokens,
             "reason": "no durable cues"}
        )
    return kept, stats
