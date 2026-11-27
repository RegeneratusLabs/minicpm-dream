"""Turn-time recall: the read path an agent actually uses.

The nightly pass decides what to keep. This module is how an agent asks for it
back while the conversation is still happening, so it is deliberately cheap:
FTS5 plus cached vectors, no reranker, no model call, one connection.

Every hit carries its receipts (the verbatim quotes and the episodes they came
from) and the flags an agent needs to avoid stating a stale or contested fact as
current truth. When nothing relevant is stored the service says so instead of
returning the least-bad row.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..config import Config
from ..store.db import Store
from ..util.text import containment, normalize_text, quote_in_source, token_set
from ..util.timeutil import human_age, iso_utc, parse_timestamp, utc_now

# Sections where a fact does not go stale with age.
DURABLE_SECTIONS = {"identity", "preferences"}
# Query-coverage below this is "I do not have this", not "here is a weak guess".
DEFAULT_MIN_COVERAGE = 0.2


def _stem_match(query: str, text: str) -> bool:
    """Tolerate inflection without a stemmer: run/runs, editor/editors, prefer/prefers."""
    if query == text:
        return True
    short, long_ = (query, text) if len(query) <= len(text) else (text, query)
    if len(short) >= 4:
        return long_.startswith(short)
    # Three-letter stems only match their immediate inflections (run -> runs),
    # not arbitrary longer words (car -> cardigan).
    return len(short) >= 3 and long_.startswith(short) and len(long_) - len(short) <= 2


def _coverage(query_tokens: set[str], text_tokens: set[str]) -> float:
    """Fraction of query terms present, tolerant of inflection (prefer/prefers)."""
    if not query_tokens:
        return 0.0
    hits = sum(1 for token in query_tokens if any(_stem_match(token, other) for other in text_tokens))
    return hits / len(query_tokens)
# Age beyond which a non-durable memory is reported as stale rather than current.
DEFAULT_STALE_DAYS = 180


@dataclass
class RecallHit:
    memory: dict[str, Any]
    receipts: list[dict[str, Any]] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    caveat: str = ""
    superseded_by: str | None = None
    relevance: float = 0.0
    age: str = ""

    def as_dict(self) -> dict[str, Any]:
        m = self.memory
        return {
            "memory_id": str(m.get("id", "")),
            "text": str(m.get("text", "")),
            "kind": str(m.get("kind", "")),
            "section": str(m.get("section", "")),
            "status": str(m.get("status", "")),
            "valid_from": m.get("valid_from"),
            "valid_to": m.get("valid_to"),
            "recorded_at": m.get("recorded_at"),
            "age": self.age,
            "confidence": float(m.get("confidence") or 0.0),
            "salience": float(m.get("salience") or 0.0),
            "relevance": round(self.relevance, 3),
            "receipts": self.receipts,
            "flags": self.flags,
            "caveat": self.caveat,
            "superseded_by": self.superseded_by,
        }


@dataclass
class RecallResult:
    query: str
    hits: list[RecallHit]
    abstained: bool = False
    reason: str = ""
    elapsed_ms: int = 0
    detail: str = "fast"

    @property
    def ids(self) -> list[str]:
        return [str(hit.memory.get("id", "")) for hit in self.hits]

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "abstained": self.abstained,
            "reason": self.reason,
            "elapsed_ms": self.elapsed_ms,
            "detail": self.detail,
            "hits": [hit.as_dict() for hit in self.hits],
        }

    def render(self, *, limit: int | None = None) -> str:
        """Compact block for a model's context, receipts included."""
        lines: list[str] = []
        if self.abstained:
            lines.append(f"No reliable memory for {self.query!r} ({self.reason}).")
        for hit in self.hits[: limit or len(self.hits)]:
            m = hit.memory
            when = f" since {m['valid_from']}" if m.get("valid_from") else ""
            age = f", {hit.age}" if hit.age else ""
            lines.append(
                f"- [{m.get('id')}] {m.get('section')}/{m.get('kind')}{when}{age}: {m.get('text')}"
            )
            for receipt in hit.receipts[:2]:
                said = receipt.get("spoken_at") or ""
                lines.append(f"    ↳ {said} \"{receipt.get('quote', '')}\"")
            if hit.flags:
                lines.append(f"    ! {'; '.join(hit.flags)}")
            if hit.caveat:
                lines.append(f"    ! {hit.caveat}")
        if not self.hits and not self.abstained:
            lines.append("No memories matched.")
        return "\n".join(lines)


class RecallService:
    """Read path over the ledger. Cheap by construction: no model, no reranker."""

    def __init__(
        self,
        cfg: Config,
        *,
        store: Store | None = None,
        embedder: Any = None,
        index: Any = None,
        stale_days: int = DEFAULT_STALE_DAYS,
        min_coverage: float = DEFAULT_MIN_COVERAGE,
    ):
        self.cfg = cfg
        self._owns_store = store is None
        self.store = store or Store(cfg.paths.db)
        self.embedder = embedder
        self.index = index
        self.stale_days = stale_days
        self.min_coverage = min_coverage
        self._contested: set[str] | None = None

    # -- lifecycle ----------------------------------------------------------
    def close(self) -> None:
        for component in (self.index, self.embedder):
            if component is not None and hasattr(component, "close"):
                try:
                    component.close()
                except Exception:  # noqa: BLE001
                    pass
        if self._owns_store:
            self.store.close()

    def __enter__(self) -> RecallService:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def hybrid(self) -> bool:
        return self.embedder is not None and self.index is not None

    # -- reads --------------------------------------------------------------
    def _contested_ids(self) -> set[str]:
        if self._contested is not None:
            return self._contested
        ids: set[str] = set()
        try:
            rows = self.store.conn.execute(
                "SELECT payload_json FROM proposals WHERE status='pending'"
            ).fetchall()
        except sqlite3.Error:
            rows = []
        for row in rows:
            try:
                payload = json.loads(row["payload_json"])
            except (TypeError, ValueError):
                continue
            decision = payload.get("decision") or {}
            for target in decision.get("target_ids") or []:
                ids.add(str(target))
            if payload.get("memory_id"):
                ids.add(str(payload["memory_id"]))
        self._contested = ids
        return ids

    def _receipts(self, memory_id: str) -> list[dict[str, Any]]:
        rows = self.store.conn.execute(
            """SELECT e.episode_id, e.quote, e.verified, ep.spoken_at_utc, ep.role, ep.source
               FROM evidence e LEFT JOIN episodes ep ON ep.id = e.episode_id
               WHERE e.memory_id=?
               ORDER BY COALESCE(ep.spoken_at_utc, '') DESC""",
            (memory_id,),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            quote = str(row["quote"] or "")
            if not quote or quote == "[forgotten]":
                continue
            out.append(
                {
                    "episode_id": row["episode_id"],
                    "quote": quote,
                    "verified": bool(row["verified"]),
                    "spoken_at": row["spoken_at_utc"],
                    "role": row["role"],
                    "source": row["source"],
                }
            )
        return out

    def _superseded_by(self, memory_id: str) -> str | None:
        row = self.store.conn.execute(
            "SELECT id FROM memories WHERE supersedes_id=? AND status='active' LIMIT 1",
            (memory_id,),
        ).fetchone()
        return str(row["id"]) if row else None

    def _flags(self, memory: dict[str, Any], receipts: list[dict[str, Any]], now: datetime) -> tuple[list[str], str, str | None]:
        flags: list[str] = []
        caveats: list[str] = []
        superseded_by: str | None = None
        status = str(memory.get("status") or "active")
        mid = str(memory.get("id") or "")

        if status != "active":
            flags.append(f"{status}: not current")
            superseded_by = self._superseded_by(mid)
            when = memory.get("valid_to") or memory.get("superseded_at") or ""
            if superseded_by:
                replacement = self.store.get_memory(superseded_by) or {}
                caveats.append(
                    f"Replaced on {when} by {replacement.get('text', 'a newer entry')}; "
                    "use the current version, not this one."
                )
            elif status == "tombstone":
                caveats.append("The user asked for this to be forgotten.")
            else:
                caveats.append(f"No longer current after {when}.")

        if mid in self._contested_ids() and status == "active":
            flags.append("contested: awaiting review")
            caveats.append("This is contested and waiting for human review; treat it as provisional.")

        if receipts and not any(receipt["verified"] for receipt in receipts):
            flags.append("unverified: no checked quote")
        if not receipts:
            flags.append("no receipt")

        if status == "active" and str(memory.get("section")) not in DURABLE_SECTIONS:
            try:
                recorded = parse_timestamp(memory.get("recorded_at"))
                age_days = (now - recorded).total_seconds() / 86400
            except (TypeError, ValueError):
                age_days = 0.0
            if age_days > self.stale_days:
                flags.append("stale: older than the staleness window")
                caveats.append(
                    f"Recorded {int(age_days)} days ago; confirm it is still true before relying on it."
                )
        return flags, " ".join(caveats), superseded_by

    def _score(self, query_tokens: set[str], memory: dict[str, Any], now: datetime) -> float:
        text = " ".join(
            str(memory.get(key) or "") for key in ("text", "subject", "section")
        )
        overlap = _coverage(query_tokens, token_set(text))
        try:
            recorded = parse_timestamp(memory.get("recorded_at"))
            age_days = max(0.0, (now - recorded).total_seconds() / 86400)
        except (TypeError, ValueError):
            age_days = 365.0
        recency = 1.0 / (1.0 + age_days / 90.0)
        return round(
            0.55 * overlap
            + 0.20 * recency
            + 0.15 * float(memory.get("confidence") or 0.0)
            + 0.10 * float(memory.get("salience") or 0.0),
            4,
        )

    def recall(
        self,
        query: str,
        *,
        k: int = 6,
        sections: list[str] | None = None,
        include_superseded: bool = False,
    ) -> RecallResult:
        started = time.perf_counter()
        query = normalize_text(query or "")
        now = utc_now()
        query_tokens = token_set(query)

        if not query:
            rows = self.store.active_memories(limit=max(1, k))
        else:
            rows = self.store.search(query, limit=max(k * 4, 12))
        if sections:
            rows = [row for row in rows if str(row.get("section")) in sections]

        if include_superseded and query:
            rows = self._with_history(rows, k)

        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            status = str(row.get("status") or "active")
            if status != "active" and not include_superseded:
                continue
            scored.append((self._score(query_tokens, row, now), row))
        scored.sort(key=lambda pair: pair[0], reverse=True)

        # Abstain on the question, not on the top row: the best-covering hit is
        # often not the highest-scored one (a generic row can outrank a precise
        # answer), and abstaining there would hide a fact we actually hold.
        best_coverage = 0.0
        if query_tokens:
            for _score, row in scored[: max(1, k)]:
                best_coverage = max(
                    best_coverage,
                    _coverage(query_tokens, token_set(str(row.get("text", "")))),
                )
        hits: list[RecallHit] = []
        for score, row in scored[: max(1, k)]:
            receipts = self._receipts(str(row["id"]))
            flags, caveat, superseded_by = self._flags(row, receipts, now)
            try:
                age = human_age(parse_timestamp(row.get("recorded_at")), now)
            except (TypeError, ValueError):
                age = ""
            hits.append(
                RecallHit(
                    memory=row,
                    receipts=receipts,
                    flags=flags,
                    caveat=caveat,
                    superseded_by=superseded_by,
                    relevance=score,
                    age=age,
                )
            )

        abstained = False
        reason = ""
        if not hits:
            abstained, reason = True, "nothing stored matches"
        elif query_tokens and best_coverage < self.min_coverage:
            abstained, reason = True, "weak match; not enough of the question is in memory"
        elif query_tokens and hits[0].relevance < 0.2:
            abstained, reason = True, "weak match"
        if not query:
            abstained, reason = not hits, "empty ledger" if not hits else ""

        elapsed = int((time.perf_counter() - started) * 1000)
        return RecallResult(
            query=query,
            hits=hits,
            abstained=abstained,
            reason=reason,
            elapsed_ms=elapsed,
            detail="hybrid" if self.hybrid else "fast",
        )

    def _with_history(self, rows: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
        """Add superseded rows for the same subject, so 'it changed' is answerable."""
        seen = {str(row.get("id")) for row in rows}
        extra: list[dict[str, Any]] = []
        for row in rows[:k]:
            for old in self.store.conn.execute(
                "SELECT * FROM memories WHERE status IN ('superseded','archived') "
                "AND section=? AND valid_to IS NOT NULL ORDER BY superseded_at DESC LIMIT 2",
                (row.get("section"),),
            ).fetchall():
                if str(old["id"]) not in seen:
                    seen.add(str(old["id"]))
                    extra.append(dict(old))
        return rows + extra

    # -- writes -------------------------------------------------------------
    def remember(
        self,
        text: str,
        *,
        kind: str = "fact",
        section: str = "projects",
        quote: str = "",
        episode_id: str = "",
        valid_from: str | None = None,
        confidence: float = 0.7,
        origin: str = "agent",
        model: str = "",
    ) -> dict[str, Any]:
        """Record one memory immediately, without a model call.

        Exact duplicates are a NOOP. A near-duplicate in the same section
        supersedes the old row (never deletes it) so corrections keep history.
        """
        from ..apply.ledger import salience_for
        from ..ingest.redact import has_secret
        from ..util.text import is_probably_meta, is_retraction

        clean = normalize_text(text or "")
        problems: list[str] = []
        if not clean:
            problems.append("empty text")
        if len(clean) > 300:
            problems.append("text exceeds 300 characters")
        if has_secret(clean):
            problems.append("secret-shaped text")
        if is_probably_meta(clean):
            problems.append("conversational meta-state")
        if is_retraction(clean):
            problems.append("retraction; reconcile it against the existing memory instead")
        if problems:
            return {"ok": False, "errors": problems, "memory_id": None}

        evidence: list[dict[str, Any]] = []
        if quote:
            verified = False
            if episode_id:
                episode = self.store.get_episode(episode_id) or {}
                verified = quote_in_source(quote, str(episode.get("text_redacted") or ""))
            evidence.append(
                {
                    "episode_id": episode_id,
                    "quote": quote,
                    "verified": verified,
                    "char_start": None,
                    "char_end": None,
                }
            )

        existing = self.store.find_active_by_norm(clean, section)
        if existing:
            return {
                "ok": True,
                "op": "NOOP",
                "memory_id": existing["id"],
                "text": existing["text"],
                "reason": "already stored",
            }

        salience = salience_for(kind, section, confidence, len(evidence))
        supersedes: str | None = None
        if evidence:
            neighbours = self.store.search(clean, limit=3)
            for row in neighbours:
                if str(row.get("section")) != section:
                    continue
                if max(
                    containment(clean, str(row.get("text", ""))),
                    containment(str(row.get("text", "")), clean),
                ) >= 0.9:
                    supersedes = str(row["id"])
                    break
        now = iso_utc(utc_now())
        with self.store.transaction():
            memory_id = self.store.insert_memory(
                kind=kind,
                section=section,
                text=clean,
                evidence=evidence,
                valid_from=valid_from,
                confidence=confidence,
                salience=salience,
                origin=origin,
                created_by="agent",
                model=model or None,
                prompt_version="recall@1",
                supersedes_id=supersedes,
            )
            if supersedes:
                self.store.supersede(supersedes, valid_to=valid_from or now)
            self.store.conn.execute(
                """INSERT INTO ops (id, run_id, ts, op, target_ids_json, payload_json,
                   decision, rationale, validator_result, applied)
                   VALUES (?,?,?,?,?,?,?,?,?,1)""",
                (
                    f"op_{memory_id}",
                    None,
                    now,
                    "UPDATE" if supersedes else "ADD",
                    json.dumps([supersedes] if supersedes else []),
                    json.dumps({"memory_id": memory_id, "text": clean, "origin": origin}),
                    "remember",
                    "written by the agent at turn time",
                    "ok",
                ),
            )
        return {
            "ok": True,
            "op": "UPDATE" if supersedes else "ADD",
            "memory_id": memory_id,
            "text": clean,
            "supersedes": supersedes,
            "verified": bool(evidence and evidence[0]["verified"]),
        }

    def explain(self, memory_id: str) -> dict[str, Any] | None:
        memory = self.store.get_memory(memory_id)
        if not memory:
            return None
        receipts = self._receipts(memory_id)
        flags, caveat, superseded_by = self._flags(memory, receipts, utc_now())
        history = [
            dict(row)
            for row in self.store.conn.execute(
                "SELECT version, text, valid_from, valid_to, created_at FROM memory_versions "
                "WHERE memory_id=? ORDER BY version",
                (memory_id,),
            ).fetchall()
        ]
        chain: list[dict[str, Any]] = []
        current = memory.get("supersedes_id")
        guard = 0
        while current and guard < 20:
            guard += 1
            old = self.store.get_memory(str(current))
            if not old:
                break
            chain.append({"memory_id": old["id"], "text": old["text"], "status": old["status"],
                          "valid_to": old.get("valid_to")})
            current = old.get("supersedes_id")
        return {
            "memory": memory,
            "receipts": receipts,
            "flags": flags,
            "caveat": caveat,
            "superseded_by": superseded_by,
            "history": history,
            "supersedes_chain": chain,
        }

    def status(self) -> dict[str, Any]:
        counts = self.store.counts()
        last = self.store.conn.execute(
            "SELECT id, finished_at, status, counters_json FROM runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        return {
            "memories": counts,
            "hybrid": self.hybrid,
            "fts": self.store.fts_enabled,
            "last_run": dict(last) if last else None,
            "ledger": str(self.store.path),
        }
