-- minicpm-dream ledger schema (version 1)
-- Bitemporal: valid_from/valid_to = world time; recorded_at/superseded_at = system time.

CREATE TABLE IF NOT EXISTS episodes (
    id              TEXT PRIMARY KEY,
    source          TEXT NOT NULL,
    session_id      TEXT,
    role            TEXT,
    spoken_at_utc   TEXT NOT NULL,
    spoken_at_local TEXT,
    tz              TEXT,
    text_hash       TEXT NOT NULL,
    text_redacted   TEXT,
    trust_tier      TEXT DEFAULT 'trusted',
    flags           TEXT DEFAULT '[]',
    meta_json       TEXT DEFAULT '{}',
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_episodes_ts ON episodes(spoken_at_utc);
CREATE INDEX IF NOT EXISTS idx_episodes_hash ON episodes(text_hash);

CREATE TABLE IF NOT EXISTS memories (
    seq             INTEGER PRIMARY KEY AUTOINCREMENT,
    id              TEXT NOT NULL UNIQUE,
    kind            TEXT NOT NULL,
    section         TEXT NOT NULL,
    text            TEXT NOT NULL,
    norm_text       TEXT NOT NULL,
    subject         TEXT DEFAULT '',
    entities_json   TEXT DEFAULT '[]',
    valid_from      TEXT,
    valid_to        TEXT,
    recorded_at     TEXT NOT NULL,
    superseded_at   TEXT,
    last_confirmed_at TEXT,
    confidence      REAL DEFAULT 0.0,
    salience        REAL DEFAULT 0.0,
    status          TEXT NOT NULL DEFAULT 'active',
    origin          TEXT NOT NULL DEFAULT 'said',
    sensitivity     TEXT NOT NULL DEFAULT 'normal',
    created_by      TEXT NOT NULL DEFAULT 'slm',
    model           TEXT,
    prompt_version  TEXT,
    content_hash    TEXT NOT NULL,
    supersedes_id   TEXT,
    run_id          TEXT
);
CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);
CREATE INDEX IF NOT EXISTS idx_memories_section ON memories(section, status);
CREATE INDEX IF NOT EXISTS idx_memories_norm ON memories(norm_text);
CREATE INDEX IF NOT EXISTS idx_memories_hash ON memories(content_hash);

CREATE TABLE IF NOT EXISTS memory_versions (
    memory_id       TEXT NOT NULL,
    version         INTEGER NOT NULL,
    text            TEXT NOT NULL,
    op_id           TEXT,
    valid_from      TEXT,
    valid_to        TEXT,
    created_at      TEXT NOT NULL,
    PRIMARY KEY (memory_id, version)
);

CREATE TABLE IF NOT EXISTS evidence (
    memory_id       TEXT NOT NULL,
    episode_id      TEXT NOT NULL,
    quote           TEXT NOT NULL,
    quote_hash      TEXT NOT NULL,
    char_start      INTEGER,
    char_end        INTEGER,
    source_path     TEXT,
    source_hash     TEXT,
    verified        INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    PRIMARY KEY (memory_id, episode_id, quote_hash)
);
CREATE INDEX IF NOT EXISTS idx_evidence_memory ON evidence(memory_id);

CREATE TABLE IF NOT EXISTS ops (
    id              TEXT PRIMARY KEY,
    run_id          TEXT,
    ts              TEXT NOT NULL,
    op              TEXT NOT NULL,
    target_ids_json TEXT DEFAULT '[]',
    payload_json    TEXT DEFAULT '{}',
    decision        TEXT,
    rationale       TEXT,
    validator_result TEXT,
    applied         INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_ops_run ON ops(run_id);

CREATE TABLE IF NOT EXISTS proposals (
    id              TEXT PRIMARY KEY,
    run_id          TEXT,
    ts              TEXT NOT NULL,
    kind            TEXT NOT NULL,
    target          TEXT NOT NULL,
    payload_json    TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    cooldown_until  TEXT,
    reask_after     TEXT,
    reviewer        TEXT,
    reason          TEXT,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_proposals_status ON proposals(status, ts);

CREATE TABLE IF NOT EXISTS runs (
    id              TEXT PRIMARY KEY,
    source          TEXT,
    window_start    TEXT,
    window_end      TEXT,
    config_hash     TEXT,
    model           TEXT,
    backend         TEXT,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    status          TEXT NOT NULL DEFAULT 'running',
    counters_json   TEXT DEFAULT '{}',
    usage_json      TEXT DEFAULT '{}',
    error           TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at);

CREATE TABLE IF NOT EXISTS kv (
    key             TEXT PRIMARY KEY,
    value           TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rejects (
    key             TEXT PRIMARY KEY,
    memory_id       TEXT,
    text            TEXT,
    reason          TEXT,
    created_at      TEXT NOT NULL,
    reask_after     TEXT
);

CREATE TABLE IF NOT EXISTS cooldowns (
    key             TEXT PRIMARY KEY,
    kind            TEXT,
    until           TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_vectors (
    memory_id       TEXT PRIMARY KEY,
    model           TEXT NOT NULL,
    dim             INTEGER NOT NULL,
    vector          BLOB NOT NULL
);
