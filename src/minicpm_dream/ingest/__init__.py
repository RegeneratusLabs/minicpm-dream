"""Transcript ingestion adapters and pre-model filters."""

from .base import Episode, iter_jsonl, make_episode
from .claude_code import iter_claude_code, session_files
from .content import flatten_content
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
    "flatten_content",
    "iter_claude_code",
    "iter_jsonl",
    "make_episode",
    "quarantine_reason",
    "read_jsonl",
    "session_files",
    "redact",
    "scrubbed_for_prompt",
]
