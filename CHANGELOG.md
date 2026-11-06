# Changelog

## 0.2.0 — OpenBMB retrieval stack and real-memory validation

- Default retrieval is now the complete OpenBMB stack:
  `MiniCPM-Embedding-Light` (440M, 1024-dim, 8K tokens) plus
  `MiniCPM-Reranker-Light` (1.2B cross-encoder), with cached SQLite vectors and
  reciprocal rank fusion over FTS5.
- Optional `[openbmb]` extra, `setup --download-embeddings`, `index` command,
  and `doctor` cache/dependency checks.
- Cloud-compatible `json_object` fallback for endpoints that reject
  `json_schema`; `reasoning_effort` support; evidence normalization for string
  arrays.
- `--hermes-home` isolation, `--max-minutes`, and memory-pressure reporting in
  `review.json` and the cron gate.
- Hermes action fitting now simulates the final file and drops actions that
  would exceed the char limit.
- Added the 24-turn / 14-fact `full-day` suite and a local-vs-cloud comparison
  script (`scripts/compare_models.py`).
- Real Hermes sandbox validation and the fine-tuning decision recorded in
  `docs/models.md`.

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
