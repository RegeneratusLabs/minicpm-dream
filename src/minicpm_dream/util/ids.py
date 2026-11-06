"""Small, dependency-free utilities shared by the package."""

from __future__ import annotations

import hashlib
import re
import secrets
import time

_SAFE = re.compile(r"[^a-z0-9_-]+")


def new_id(prefix: str = "mem") -> str:
    """Return a time-sortable, collision-resistant identifier."""
    ms = int(time.time() * 1000)
    return f"{prefix}_{ms:013d}{secrets.token_hex(8)}"


def stable_id(prefix: str, *parts: object) -> str:
    """Deterministic identifier derived from content (idempotent re-ingest)."""
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8", "replace"))
        h.update(b"\x1f")
    return f"{prefix}_{h.hexdigest()[:24]}"


def stable_hash(*parts: object) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8", "replace"))
        h.update(b"\x1f")
    return h.hexdigest()


def slug(value: str, *, limit: int = 40) -> str:
    value = _SAFE.sub("-", value.strip().lower()).strip("-")
    return value[:limit] or "item"
