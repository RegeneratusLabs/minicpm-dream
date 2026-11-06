"""Transcript ingestion adapters and pre-model filters."""

from .base import Episode, iter_jsonl, make_episode
from .filters import EpisodeFilter, FilterStats
from .hermes import HermesAdapter, HermesInfo
from .jsonl import read_jsonl
from .redact import (
    has_secret,
    injection_flags,
    quarantine_reason,
    redact,
    scrubbed_for_prompt,
)

__all__ = [
    "Episode",
    "EpisodeFilter",
    "FilterStats",
    "HermesAdapter",
    "HermesInfo",
    "has_secret",
    "injection_flags",
    "iter_jsonl",
    "make_episode",
    "quarantine_reason",
    "read_jsonl",
    "redact",
    "scrubbed_for_prompt",
]
