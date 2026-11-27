# Changelog

## 1.1.0 — the read path

The nightly pass was a diary nobody read. This release makes memory something an
agent uses mid-conversation.

- `minicpm-dream serve`: stdlib-only MCP stdio server with `recall`, `remember`,
  `explain`, `forget` and `memory_status`. Works with Claude Code, Cursor, DSH
  and any other MCP client; no new dependency.
- `minicpm-dream recall` / `remember`: the same lookup and write for shell hooks
  and scripts that are not MCP clients.
- Turn-time recall (`recall/`): FTS5 with prefix matching, lexical coverage
  scoring, no model and no reranker, sub-millisecond on a small ledger. Every
  hit carries its receipts and flags (`superseded`, `contested`, `stale`,
  `unverified`) with a caveat, and the service abstains instead of guessing.
- `eval-qa`: end-to-end QA evaluation. Builds a ledger from the transcript, asks
  questions through the real recall path, and scores answers with and without
  memory, plus retrieval hit rate, recall latency and tokens. Shipped `qa.json`
  fixtures for `basic` and `full-day`.
- Profile compaction (`apply/profile.py`): pinned sections
  (`identity`, `preferences`) plus human-written entries are never evicted;
  low-salience entries are evicted instead of refusing new facts. Every eviction
  is reported in `review.json` and `DREAMS.md`; the ledger keeps the history.
- `minicpm-dream compact`, and known memory pressure now gets an explicit
  eviction note rather than a silent skip.
- Cost: `extract.salience_gate` skips chunks with no durable cues before the
  model call, and `run --incremental` resumes from the last successful pass
  (30-minute overlap) instead of re-extracting the whole day.
- Ingest: `--source claude_code` reads `~/.claude/projects/**/*.jsonl`, with
  tool results (the actual work) and sidechain turns tagged as `subagent`; the
  JSONL adapter accepts nested OpenAI/Anthropic-style `message.content` blocks.
- Removed the declared-but-empty `server` extra (FastAPI/uvicorn); the MCP
  server needs neither.
- 118 tests; ruff and mypy clean.

## 1.0.0-rc1 — release candidate

- `python -m minicpm_dream` entrypoint; clean-wheel install verified.
- Hermes gate installer and `--hermes-home` isolation tested.
- Optimized local run on a real Hermes sandbox: 12 chunks, 10 ledger writes,
  1 `MEMORY.md` addition, OpenBMB reranker active, live files untouched.
- `SECURITY.md`, `CONTRIBUTING.md`, and [docs/release.md](docs/release.md)
  release checklist.
- CI matrix for Python 3.11 and 3.12.
- 82 tests passing; ruff and mypy clean.

## 0.4.0 — scalable eval and training scaffold

- `minicpm-dream generate-eval` creates deterministic multi-day suites with
  gold facts, updates, dates, Chinese turns, noise, injections, and secrets.
- Labels retain verbatim quotes and final/superseded status for both evaluation
  and SFT.
- `training/build_dataset.py` builds extraction and reconciliation JSONL with
  the exact production prompts/schemas.
- `training/train_lora.py` provides a plain transformers+peft LoRA loop, plus
  `training/README.md` with GPU, vLLM evaluation, and acceptance gates.
- Tests for the generator and dataset builder.

## 0.3.0 — quality optimisation campaign

- Session-aware chunking, long-turn sentence splitting, and per-message fallback
  for short user turns, with an empty-message retry.
- Deterministic derived facts (relationships, migrations, subagent lessons) that
  bypass LLM reconciliation and are protected from model supersede.
- Sequential reconcile/apply in one transaction, so later candidates see
  earlier memories and intra-run updates are handled.
- Deterministic compaction of duplicates, stale SQLite facts, visiting events,
  and question echoes, with evidence preservation.
- Static system prompt for llama.cpp prefix caching; section/kind heuristics;
  status-echo and meta filters; lenient quote-overlap verification.
- Destructive-op guards: cross-section rewrites, identity loss, and low-overlap
  supersede become review proposals.
- `--quality` / `--fast` modes, 1500-token default chunks, and
  `docs/optimizations.md` with measured results.
- Fine-tuning decision recorded: not needed for average quality; GPU use is for
  an 8B baseline and a targeted LoRA only if the larger suite shows a floor.

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
