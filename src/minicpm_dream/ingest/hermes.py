"""Read-only Hermes Agent adapter.

The Hermes state database is an internal schema that changes between releases.
This adapter introspects it, uses only columns that exist, and degrades with a
warning instead of guessing. It never writes to the database.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import Config
from ..util.logging import get_logger
from ..util.text import normalize_text
from ..util.timeutil import iso_utc, parse_timestamp
from .base import Episode, make_episode

log = get_logger("ingest.hermes")


@dataclass
class HermesInfo:
    path: str
    available: bool
    tables: list[str]
    session_columns: list[str]
    message_columns: list[str]
    note: str = ""


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    try:
        return [str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')]
    except sqlite3.DatabaseError:
        return []


class HermesAdapter:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.path = Path(cfg.hermes.state_db).expanduser()

    def inspect(self) -> HermesInfo:
        if not self.path.is_file():
            return HermesInfo(str(self.path), False, [], [], [], "state.db not found")
        try:
            conn = self._connect()
        except sqlite3.DatabaseError as exc:
            return HermesInfo(str(self.path), False, [], [], [], f"cannot open: {exc}")
        try:
            tables = [
                str(r[0])
                for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            ]
            msg_cols = _table_columns(conn, "messages")
            ses_cols = _table_columns(conn, "sessions")
            missing = []
            if "messages" not in tables:
                missing.append("messages")
            if "sessions" not in tables:
                missing.append("sessions")
            for col in ("session_id", "role", "content", "timestamp"):
                if col not in msg_cols:
                    missing.append(f"messages.{col}")
            note = "" if not missing else "missing: " + ", ".join(missing)
            return HermesInfo(
                str(self.path), not missing, tables, ses_cols, msg_cols, note
            )
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        uri = f"file:{self.path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def iter_episodes(
        self,
        *,
        since: Any,
        until: Any,
        limit: int | None = None,
        tz: str = "UTC",
    ) -> Iterator[Episode]:
        info = self.inspect()
        if not info.available:
            raise RuntimeError(f"Hermes state.db is not usable: {info.note}")
        start = since if isinstance(since, str) else iso_utc(since)
        end = until if isinstance(until, str) else iso_utc(until)
        start_epoch = parse_timestamp(start, tz).timestamp()
        end_epoch = parse_timestamp(end, tz).timestamp()

        msg_cols = set(info.message_columns)
        ses_cols = set(info.session_columns)
        where = ["m.role IN ('user','assistant')", "m.timestamp >= ?", "m.timestamp < ?"]
        if "active" in msg_cols:
            where.append("COALESCE(m.active, 1) = 1")
        if "compacted" in msg_cols:
            where.append("COALESCE(m.compacted, 0) = 0")
        if "_compressed_summary" in msg_cols:
            where.append("COALESCE(m._compressed_summary, 0) = 0")

        select = [
            "m.session_id AS session_id",
            "m.role AS role",
            "m.content AS content",
            "m.timestamp AS timestamp",
            "s.source AS source",
        ]
        optional = {
            "m.id": "message_id",
            "s.chat_type": "chat_type",
            "s.chat_id": "chat_id",
            "s.thread_id": "thread_id",
            "s.parent_session_id": "parent_session_id",
            "s.hidden": "hidden",
            "s.title": "title",
        }
        for expr, alias in optional.items():
            col = expr.split(".")[1]
            table_cols = msg_cols if expr.startswith("m.") else ses_cols
            if col in table_cols:
                select.append(f"{expr} AS {alias}")
        if "parent_session_id" in ses_cols:
            where.append("s.parent_session_id IS NULL")
        if "hidden" in ses_cols:
            where.append("COALESCE(s.hidden, 0) = 0")

        sql = (
            f"SELECT {', '.join(select)} FROM messages m "
            "LEFT JOIN sessions s ON s.id = m.session_id "
            f"WHERE {' AND '.join(where)} ORDER BY m.timestamp ASC"
        )
        if limit:
            sql += f" LIMIT {int(limit)}"

        conn = self._connect()
        try:
            for row in conn.execute(sql, (start_epoch, end_epoch)):
                text = row["content"]
                if not isinstance(text, str):
                    continue
                text = normalize_text(text)
                if len(text) < 3:
                    continue
                source = str(row["source"] or "unknown")
                if source in self.cfg.ingest.exclude_sources:
                    continue
                if source not in self.cfg.ingest.trusted_sources:
                    continue
                meta = {}
                for key in ("chat_type", "chat_id", "thread_id", "title"):
                    if key in row.keys() and row[key] is not None:
                        meta[key] = row[key]
                yield make_episode(
                    source=source,
                    session_id=str(row["session_id"] or ""),
                    role=str(row["role"] or "user"),
                    timestamp=float(row["timestamp"]),
                    text=text,
                    tz=tz,
                    message_id=row["message_id"] if "message_id" in row.keys() else None,
                    meta=meta,
                )
        finally:
            conn.close()

    def read_memory_limits(self) -> tuple[int, int]:
        """Best-effort read of the user's configured char limits."""
        limits = (self.cfg.hermes.memory_char_limit, self.cfg.hermes.user_char_limit)
        config = Path(self.cfg.hermes.home) / "config.yaml"
        if not config.is_file():
            return limits
        try:
            text = config.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return limits
        import re

        def find(key: str, default: int) -> int:
            match = re.search(rf"^\s*{key}\s*:\s*(\d+)\s*$", text, re.M)
            return int(match.group(1)) if match else default

        return find("memory_char_limit", limits[0]), find("user_char_limit", limits[1])
