"""SQLite ledger: bitemporal memories, evidence, ops, proposals, runs."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ..util.atomic import ensure_dir
from ..util.ids import new_id, stable_hash
from ..util.text import norm_key, normalize_text, tokens
from ..util.timeutil import iso_utc, utc_now

SCHEMA_VERSION = 1

_FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    text, subject, section,
    content='memories', content_rowid='seq',
    tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, text, subject, section)
    VALUES (new.seq, new.text, new.subject, new.section);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text, subject, section)
    VALUES ('delete', old.seq, old.text, old.subject, old.section);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text, subject, section)
    VALUES ('delete', old.seq, old.text, old.subject, old.section);
    INSERT INTO memories_fts(rowid, text, subject, section)
    VALUES (new.seq, new.text, new.subject, new.section);
END;
"""


class Store:
    """Thin, explicit data-access layer. Callers own transaction boundaries."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        ensure_dir(self.path.parent)
        self.conn = sqlite3.connect(str(self.path), isolation_level=None, timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.fts_enabled = False
        self._init_schema()

    # -- schema -------------------------------------------------------------
    def _init_schema(self) -> None:
        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise RuntimeError(
                f"ledger schema v{version} is newer than this build (v{SCHEMA_VERSION}); upgrade minicpm-dream"
            )
        schema = (Path(__file__).with_name("schema.sql")).read_text(encoding="utf-8")
        self.conn.executescript(schema)
        try:
            self.conn.executescript(_FTS_SQL)
            self.fts_enabled = True
        except sqlite3.OperationalError:
            self.fts_enabled = False
        self.conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield self.conn
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        else:
            self.conn.execute("COMMIT")

    # -- episodes -----------------------------------------------------------
    def insert_episode(self, episode: dict[str, Any]) -> bool:
        try:
            self.conn.execute(
                """INSERT INTO episodes
                   (id, source, session_id, role, spoken_at_utc, spoken_at_local, tz,
                    text_hash, text_redacted, trust_tier, flags, meta_json, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    episode["id"],
                    episode.get("source", ""),
                    episode.get("session_id"),
                    episode.get("role"),
                    episode["spoken_at_utc"],
                    episode.get("spoken_at_local"),
                    episode.get("tz"),
                    episode["text_hash"],
                    episode.get("text_redacted"),
                    episode.get("trust_tier", "trusted"),
                    json.dumps(episode.get("flags", [])),
                    json.dumps(episode.get("meta", {})),
                    iso_utc(utc_now()),
                ),
            )
            return True
        except sqlite3.IntegrityError:
            return False

    def get_episode(self, episode_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM episodes WHERE id=?", (episode_id,)).fetchone()
        return dict(row) if row else None

    # -- memories -----------------------------------------------------------
    def insert_memory(
        self,
        *,
        kind: str,
        section: str,
        text: str,
        evidence: list[dict[str, Any]],
        subject: str = "",
        entities: list[str] | None = None,
        valid_from: str | None = None,
        valid_to: str | None = None,
        confidence: float = 0.0,
        salience: float = 0.0,
        origin: str = "said",
        status: str = "active",
        sensitivity: str = "normal",
        created_by: str = "slm",
        model: str | None = None,
        prompt_version: str | None = None,
        supersedes_id: str | None = None,
        run_id: str | None = None,
        memory_id: str | None = None,
    ) -> str:
        mid = memory_id or new_id("mem")
        now = iso_utc(utc_now())
        norm = norm_key(text)
        content_hash = stable_hash(kind, norm, valid_from or "")
        self.conn.execute(
            """INSERT INTO memories
               (id, kind, section, text, norm_text, subject, entities_json, valid_from, valid_to,
                recorded_at, last_confirmed_at, confidence, salience, status, origin, sensitivity,
                created_by, model, prompt_version, content_hash, supersedes_id, run_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                mid, kind, section, text, norm, subject, json.dumps(entities or []), valid_from,
                valid_to, now, now, float(confidence), float(salience), status, origin, sensitivity,
                created_by, model, prompt_version, content_hash, supersedes_id, run_id,
            ),
        )
        self.conn.execute(
            """INSERT INTO memory_versions
               (memory_id, version, text, op_id, valid_from, valid_to, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (mid, 1, text, None, valid_from, valid_to, now),
        )
        for ev in evidence:
            quote = normalize_text(str(ev.get("quote", "")))
            self.conn.execute(
                """INSERT OR IGNORE INTO evidence
                   (memory_id, episode_id, quote, quote_hash, char_start, char_end,
                    source_path, source_hash, verified, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    mid,
                    ev.get("episode_id", ""),
                    quote,
                    stable_hash(quote),
                    ev.get("char_start"),
                    ev.get("char_end"),
                    ev.get("source_path"),
                    ev.get("source_hash"),
                    int(bool(ev.get("verified", False))),
                    now,
                ),
            )
        return mid

    def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
        return dict(row) if row else None

    def find_active_by_norm(self, text: str, section: str | None = None) -> dict[str, Any] | None:
        norm = norm_key(text)
        if section:
            row = self.conn.execute(
                "SELECT * FROM memories WHERE norm_text=? AND section=? AND status='active' LIMIT 1",
                (norm, section),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT * FROM memories WHERE norm_text=? AND status='active' LIMIT 1", (norm,)
            ).fetchone()
        return dict(row) if row else None

    def supersede(self, memory_id: str, *, valid_to: str | None, status: str = "superseded") -> None:
        self.conn.execute(
            "UPDATE memories SET status=?, valid_to=COALESCE(?, valid_to), superseded_at=? WHERE id=?",
            (status, valid_to, iso_utc(utc_now()), memory_id),
        )

    def archive(self, memory_id: str) -> None:
        self.supersede(memory_id, valid_to=None, status="archived")

    def forget(self, memory_id: str, *, reason: str = "user_request") -> bool:
        row = self.get_memory(memory_id)
        if not row:
            return False
        now = iso_utc(utc_now())
        self.conn.execute(
            """UPDATE memories SET status='tombstone', text='[forgotten]', norm_text='',
               subject='', entities_json='[]', superseded_at=?, sensitivity='normal' WHERE id=?""",
            (now, memory_id),
        )
        self.conn.execute(
            "UPDATE evidence SET quote='[forgotten]', verified=0 WHERE memory_id=?", (memory_id,)
        )
        self.conn.execute("DELETE FROM memory_vectors WHERE memory_id=?", (memory_id,))
        self.conn.execute(
            "INSERT OR REPLACE INTO kv(key, value, updated_at) VALUES (?,?,?)",
            (f"forgotten:{memory_id}", reason, now),
        )
        return True

    def active_memories(self, *, section: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM memories WHERE status='active'"
        args: list[Any] = []
        if section:
            sql += " AND section=?"
            args.append(section)
        sql += " ORDER BY salience DESC, recorded_at DESC"
        if limit:
            sql += " LIMIT ?"
            args.append(limit)
        return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def counts(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT status, COUNT(*) AS n, COALESCE(SUM(LENGTH(text)),0) AS chars "
            "FROM memories GROUP BY status"
        ).fetchall()
        out = {r["status"]: int(r["n"]) for r in rows}
        out["chars"] = sum(int(r["chars"]) for r in rows)
        out["active"] = out.get("active", 0)
        return out

    # -- retrieval ----------------------------------------------------------
    def search(self, query: str, limit: int = 6) -> list[dict[str, Any]]:
        toks = [t for t in tokens(query) if len(t) > 1][:24]
        if not toks:
            return self.active_memories(limit=limit)
        rows: list[sqlite3.Row] = []
        if self.fts_enabled:
            match = " OR ".join('"' + t.replace('"', '""') + '"' for t in toks)
            try:
                rows = self.conn.execute(
                    """SELECT m.*, bm25(memories_fts) AS rank
                       FROM memories_fts JOIN memories m ON m.seq = memories_fts.rowid
                       WHERE memories_fts MATCH ? AND m.status='active'
                       ORDER BY rank LIMIT ?""",
                    (match, max(limit * 4, limit)),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
        if not rows:
            like = f"%{toks[0]}%"
            rows = self.conn.execute(
                "SELECT * FROM memories WHERE status='active' AND (text LIKE ? OR norm_text LIKE ?) "
                "ORDER BY salience DESC LIMIT ?",
                (like, like, max(limit * 4, limit)),
            ).fetchall()
        now = utc_now()
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            d = dict(row)
            base = 1.0 / (1.0 + abs(float(d.get("rank") or 0.0)))
            try:
                recorded = datetime.fromisoformat(str(d["recorded_at"]))
                age_days = max(0.0, (now - recorded).total_seconds() / 86400)
            except (ValueError, TypeError, KeyError):
                age_days = 365.0
            recency = 1.0 / (1.0 + age_days / 30.0)
            score = base + 0.15 * recency + 0.1 * float(d.get("confidence") or 0.0)
            scored.append((score, d))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [d for _, d in scored[:limit]]

    # -- runs ---------------------------------------------------------------
    def record_run(self, *, run_id: str, source: str, window_start: str, window_end: str,
                   config_hash: str, model: str, backend: str) -> None:
        self.conn.execute(
            """INSERT INTO runs (id, source, window_start, window_end, config_hash, model, backend,
                                 started_at, status) VALUES (?,?,?,?,?,?,?,?, 'running')""",
            (run_id, source, window_start, window_end, config_hash, model, backend, iso_utc(utc_now())),
        )

    def finish_run(self, run_id: str, *, status: str, counters: dict[str, Any],
                   usage: dict[str, Any], error: str | None = None) -> None:
        self.conn.execute(
            "UPDATE runs SET finished_at=?, status=?, counters_json=?, usage_json=?, error=? WHERE id=?",
            (iso_utc(utc_now()), status, json.dumps(counters), json.dumps(usage), error, run_id),
        )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return dict(row) if row else None

    # -- proposals ----------------------------------------------------------
    def add_proposal(self, *, run_id: str, kind: str, target: str, payload: dict[str, Any],
                     cooldown_days: int = 0) -> str:
        pid = new_id("prop")
        now = utc_now()
        cooldown = iso_utc(now + timedelta(days=cooldown_days)) if cooldown_days else None
        self.conn.execute(
            """INSERT INTO proposals (id, run_id, ts, kind, target, payload_json, status,
                                      cooldown_until, created_at)
               VALUES (?,?,?,?,?,?, 'pending', ?, ?)""",
            (pid, run_id, iso_utc(now), kind, target, json.dumps(payload), cooldown, iso_utc(now)),
        )
        return pid

    def list_proposals(self, status: str | None = "pending") -> list[dict[str, Any]]:
        if status:
            rows = self.conn.execute(
                "SELECT * FROM proposals WHERE status=? ORDER BY ts", (status,)
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM proposals ORDER BY ts").fetchall()
        out = []
        for row in rows:
            d = dict(row)
            d["payload"] = json.loads(d.pop("payload_json"))
            out.append(d)
        return out

    def get_proposal(self, proposal_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM proposals WHERE id=?", (proposal_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["payload"] = json.loads(d.pop("payload_json"))
        return d

    def set_proposal_status(self, proposal_id: str, status: str, *, reviewer: str = "user",
                            reason: str | None = None) -> bool:
        cur = self.conn.execute(
            "UPDATE proposals SET status=?, reviewer=?, reason=? WHERE id=?",
            (status, reviewer, reason, proposal_id),
        )
        return cur.rowcount > 0

    def proposal_counts(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT status, COUNT(*) AS n FROM proposals GROUP BY status"
        ).fetchall()
        return {r["status"]: int(r["n"]) for r in rows}

    # -- kv -----------------------------------------------------------------
    def kv_get(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def kv_set(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO kv(key, value, updated_at) VALUES (?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, value, iso_utc(utc_now())),
        )

    # -- maintenance --------------------------------------------------------
    def archive_stale(self, *, older_than_days: int, min_confidence: float = 0.7) -> int:
        cutoff = iso_utc(utc_now() - timedelta(days=older_than_days))
        cur = self.conn.execute(
            "UPDATE memories SET status='archived', superseded_at=? "
            "WHERE status='active' AND COALESCE(last_confirmed_at, recorded_at) < ? "
            "AND confidence < ? AND section NOT IN ('identity','preferences')",
            (iso_utc(utc_now()), cutoff, min_confidence),
        )
        return cur.rowcount

    def cooldown(self, key: str) -> str | None:
        row = self.conn.execute("SELECT until FROM cooldowns WHERE key=?", (key,)).fetchone()
        return row["until"] if row else None

    def set_cooldown(self, key: str, kind: str, until: datetime) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO cooldowns(key, kind, until, created_at) VALUES (?,?,?,?)",
            (key, kind, iso_utc(until), iso_utc(datetime.now(UTC))),
        )
