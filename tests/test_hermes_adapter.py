from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

from minicpm_dream.ingest.hermes import HermesAdapter


def _make_db(path):
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY, source TEXT, chat_type TEXT, chat_id TEXT,
            thread_id TEXT, parent_session_id TEXT, hidden INTEGER, started_at REAL,
            title TEXT
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT,
            timestamp REAL, active INTEGER, compacted INTEGER, _compressed_summary INTEGER
        );
        """
    )
    base = datetime(2026, 11, 5, 9, 0, tzinfo=UTC).timestamp()
    con.execute(
        "INSERT INTO sessions VALUES ('s1','desktop','dm','chat1',NULL,NULL,0,?, 'work')",
        (base,),
    )
    con.execute(
        "INSERT INTO sessions VALUES ('s2','cron',NULL,NULL,NULL,NULL,0,?, 'cron')",
        (base,),
    )
    con.execute(
        "INSERT INTO sessions VALUES ('s3','subagent','dm','chat1',NULL,'s1',0,?, 'child')",
        (base,),
    )
    con.execute(
        "INSERT INTO sessions VALUES ('s4','desktop','dm','chat2',NULL,NULL,1,?, 'hidden')",
        (base,),
    )
    rows = [
        (1, "s1", "user", "I prefer dark mode in editors.", base, 1, 0, 0),
        (2, "s1", "assistant", "Noted.", base + 10, 1, 0, 0),
        (3, "s1", "tool", "tool output", base + 20, 1, 0, 0),
        (4, "s2", "user", "cron prompt text", base + 30, 1, 0, 0),
        (5, "s3", "user", "child agent text", base + 40, 1, 0, 0),
        (6, "s4", "user", "hidden session text", base + 50, 1, 0, 0),
        (7, "s1", "user", "compacted old text", base + 60, 1, 1, 0),
    ]
    con.executemany("INSERT INTO messages VALUES (?,?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()


def test_inspect_and_iter(tmp_path, cfg):
    path = tmp_path / "state.db"
    _make_db(path)
    cfg.hermes.state_db = str(path)
    adapter = HermesAdapter(cfg)
    info = adapter.inspect()
    assert info.available
    since = datetime(2026, 11, 5, 0, 0, tzinfo=UTC)
    until = datetime(2026, 11, 6, 0, 0, tzinfo=UTC)
    episodes = list(adapter.iter_episodes(since=since, until=until, tz="UTC"))
    texts = [ep.text for ep in episodes]
    assert "I prefer dark mode in editors." in texts
    assert "Noted." in texts
    assert "cron prompt text" not in texts
    assert "child agent text" not in texts
    assert "hidden session text" not in texts
    assert "compacted old text" not in texts
    assert all(ep.id.startswith("ep_") for ep in episodes)


def test_read_memory_limits(tmp_path, cfg):
    cfg.hermes.home = str(tmp_path / "hermes")
    (tmp_path / "hermes").mkdir(exist_ok=True)
    (tmp_path / "hermes" / "config.yaml").write_text(
        "memory:\n  memory_char_limit: 3000\n  user_char_limit: 2000\n", encoding="utf-8"
    )
    adapter = HermesAdapter(cfg)
    assert adapter.read_memory_limits() == (3000, 2000)
