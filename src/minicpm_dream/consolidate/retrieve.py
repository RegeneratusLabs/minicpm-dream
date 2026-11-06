"""Retrieve relevant existing memories for a candidate."""

from __future__ import annotations

from typing import Any

from ..extract.validate import Candidate
from ..store.db import Store
from ..util.timeutil import human_age, parse_timestamp, utc_now


class Retriever:
    def __init__(self, store: Store, *, top_k: int = 6):
        self.store = store
        self.top_k = top_k

    def query(self, candidate: Candidate) -> str:
        parts = [candidate.text, candidate.subject, " ".join(candidate.entities)]
        return " ".join(part for part in parts if part)

    def retrieve(self, candidate: Candidate) -> list[dict[str, Any]]:
        rows = self.store.search(self.query(candidate), limit=self.top_k)
        now = utc_now()
        for row in rows:
            try:
                row["age"] = human_age(parse_timestamp(row["recorded_at"]), now)
            except (ValueError, TypeError):
                row["age"] = "unknown"
        return rows
