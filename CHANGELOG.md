# Changelog

## 0.1.0 — initial release

- Hermes and JSONL ingest adapters, read-only and schema-tolerant.
- Deterministic secret redaction and prompt-injection quarantine.
- Token-budgeted chunking with code-resolved relative-date anchors.
- JSON-schema constrained extraction with evidence-quote validation.
- Evidence entailment verifier.
- Hybrid FTS5 retrieval and typed reconciliation
  (`ADD`/`UPDATE`/`SUPERSEDE`/`MERGE`/`NOOP`/`DEFER`).
- Bitemporal SQLite ledger, append-only versions, evidence, ops, proposals,
  cooldowns.
- Transactional apply with a 25% memory-loss guard.
- Per-run budgets and a review queue for uncertain decisions.
- `DREAMS.md` diary and Hermes-format candidate exports.
- Reviewed Hermes write-back with lock, backup, mtime check, and char limits.
- Hermes cron `wakeAgent` gate installer.
- `llama_cpp`, `openai`, and scripted `mock` backends; MiniCPM5-2B/1B and
  MiniCPM4.1-8B model registry with SHA-256 verification.
- Eval harness with synthetic fixtures and metrics.
- Tests, ruff, mypy, toy demo, and documentation.
