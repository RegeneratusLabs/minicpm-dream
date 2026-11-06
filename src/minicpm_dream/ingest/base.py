"""Canonical episode model shared by all ingest adapters."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..util.ids import stable_hash, stable_id
from ..util.text import normalize_text
from ..util.timeutil import iso_local, iso_utc, parse_timestamp


@dataclass
class Episode:
    id: str
    source: str
    session_id: str
    role: str
    spoken_at: datetime
    tz: str
    text: str
    trust_tier: str = "trusted"
    flags: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def text_hash(self) -> str:
        return stable_hash(self.session_id, self.role, iso_utc(self.spoken_at), self.text)

    @property
    def spoken_at_utc(self) -> str:
        return iso_utc(self.spoken_at)

    @property
    def spoken_at_local(self) -> str:
        return iso_local(self.spoken_at, self.tz)

    def to_record(self, *, store_text: bool = True) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "session_id": self.session_id,
            "role": self.role,
            "spoken_at_utc": self.spoken_at_utc,
            "spoken_at_local": self.spoken_at_local,
            "tz": self.tz,
            "text_hash": self.text_hash,
            "text_redacted": self.text if store_text else None,
            "trust_tier": self.trust_tier,
            "flags": self.flags,
            "meta": self.meta,
        }

    def to_json(self) -> str:
        return json.dumps(
            {
                "id": self.id,
                "source": self.source,
                "session_id": self.session_id,
                "role": self.role,
                "timestamp": iso_utc(self.spoken_at),
                "tz": self.tz,
                "text": self.text,
                "trust_tier": self.trust_tier,
                "flags": self.flags,
                "meta": self.meta,
            },
            ensure_ascii=False,
        )


def make_episode(
    *,
    source: str,
    session_id: str,
    role: str,
    timestamp: object,
    text: str,
    tz: str,
    message_id: object = None,
    trust_tier: str = "trusted",
    flags: list[str] | None = None,
    meta: dict[str, Any] | None = None,
) -> Episode:
    spoken_at = parse_timestamp(timestamp, default_tz=tz)
    clean = normalize_text(text)
    eid = stable_id(
        "ep",
        source,
        session_id,
        role,
        iso_utc(spoken_at),
        message_id if message_id is not None else stable_hash(clean)[:16],
    )
    return Episode(
        id=eid,
        source=source,
        session_id=session_id,
        role=role,
        spoken_at=spoken_at.astimezone(UTC),
        tz=tz,
        text=clean,
        trust_tier=trust_tier,
        flags=flags or [],
        meta=meta or {},
    )


def iter_jsonl(path: str) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
