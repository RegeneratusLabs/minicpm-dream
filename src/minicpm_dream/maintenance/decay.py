"""Maintenance: decay, archive and explicit forget."""

from __future__ import annotations

from ..store.db import Store


def decay(store: Store, *, older_than_days: int = 180, min_confidence: float = 0.7) -> int:
    return store.archive_stale(older_than_days=older_than_days, min_confidence=min_confidence)


def forget(store: Store, memory_id: str, *, reason: str = "user_request") -> bool:
    return store.forget(memory_id, reason=reason)
