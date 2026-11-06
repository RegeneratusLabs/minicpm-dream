# Architecture

`minicpm-dream` is a batch memory consolidator. It is deliberately not a
conversational agent and not a vector database: it is the maintenance pass that
turns raw session history into a small, auditable, bitemporal memory.

## Pipeline

```
 ingest ──► prepare ──► extract ──► verify ──► retrieve ──► reconcile ──► apply
   │           │           │           │           │            │            │
 filters     chunks      typed       entailment   hybrid       ADD/UPDATE   SQLite
 redaction   date refs   JSON        check        FTS(+vec)    SUPERSEDE    ledger
                                                               MERGE/NOOP   proposals
                                                                  │            │
                                                                  └──► synthesize ──► DREAMS.md
                                                                       export ──► Hermes candidates
```

### 1. Ingest (`ingest/`)

- **Hermes adapter**: opens `~/.hermes/state.db` read-only (`mode=ro`,
  `PRAGMA query_only=ON`, in-memory temp store), introspects the schema, and
  uses only columns that exist. It selects `user`/`assistant` messages in a UTC
  window, excluding cron, subagent, tool, hidden, compacted, and
  compression-summary messages by default.
- **JSONL adapter**: the stable integration surface. One JSON object per line
  with `session_id`, `role`, `timestamp`, `content`, optional `source`/`tz`.
- **Filters**: source allow/deny lists, role allow-list, empty/short messages,
  per-message length cap, exact-duplicate suppression, secret redaction, and
  prompt-injection quarantine. Everything is counted and reported.
- **Stable episode IDs**: derived from source, session, role, timestamp, and
  message identity, so re-ingesting the same day is idempotent.

The Hermes database is an internal schema. The adapter degrades with a warning
and a clear error instead of guessing; see `docs/hermes-integration.md`.

### 2. Prepare (`prepare/`)

- Chunks episodes under a token budget (default 6,000 estimated tokens, 10%
  overlap), never splitting a turn.
- **Date anchors**: a deterministic resolver turns relative expressions
  ("yesterday", "last Saturday", "3 days ago", "in March") into numbered
  `d0..d4` refs with absolute ISO dates, anchored to the episode timestamp and
  timezone. The model may only *select* a ref. Code inserts the date into the
  stored text.
- This exists because temporal grounding is the first capability that degrades
  in small models (see `docs/prior-art.md`). It is cheaper and safer to do date
  arithmetic in Python.

### 3. Extract (`extract/`)

One constrained JSON call per chunk (`TASK: EXTRACT`). The model returns typed
items:

`fact`, `event`, `preference`, `task`, `decision`, `self`, `question`
mapped to sections `identity`, `preferences`, `projects`, `people`,
`environment`, `procedures`, `lessons`, `open_loops`.

Each item has a short atomic statement, a subject, optional entities, a
`date_ref`, confidence, and **one or two verbatim evidence quotes**.

Deterministic post-validation (`extract/validate.py`) rejects:

- quotes that are not substrings of the cited episode (after whitespace/case
  normalization) — the episode may be inferred if exactly one episode contains
  the quote;
- conversational meta-states ("the user is asking…");
- unresolved relative-time words without a date ref;
- secret-shaped text;
- text over 240 characters;
- missing evidence.

The verifier (`extract/verify.py`) then checks entailment: the quote, in a
small context window, must `supports` the claim, not merely be related. A
`contradicts` or `insufficient` verdict drops the candidate.

### 4. Retrieve (`consolidate/retrieve.py`)

For each verified candidate, retrieve up to `top_k` active memories with hybrid
ranking:

- SQLite FTS5 BM25 over `text`, `subject`, `section`;
- MiniCPM-Embedding-Light dense vectors from `memory_vectors`, cached per
  memory and refreshed only for new/changed rows;
- reciprocal rank fusion of the two channels;
- MiniCPM-Reranker-Light cross-encoder over the top candidates.

Both OpenBMB models are local and optional; FTS5 alone remains a working
fallback.

If FTS5 is unavailable, a token/LIKE fallback keeps the system usable.

### 5. Reconcile (`consolidate/reconcile.py`)

Deterministic pre-checks before any model call:

- exact normalized duplicate → `NOOP`;
- same fact with different numbers/dates → let the model choose
  `UPDATE`/`SUPERSEDE`;
- empty retrieval → `ADD` with no model call.

Otherwise one narrow JSON call (`TASK: RECONCILE`) selects exactly one operation:

| Op | Meaning | Write |
|---|---|---|
| `ADD` | new memory | insert |
| `UPDATE` | refine an existing memory | supersede old, insert new |
| `SUPERSEDE` | corrected/changed fact | supersede old, insert new |
| `MERGE` | duplicates/fragments of one fact | supersede all targets, insert merged |
| `NOOP` | already stored | nothing |
| `DEFER` | ambiguous | review proposal |

Validators enforce: target IDs must exist in the retrieval set; `MERGE` needs
≥ 2 targets; `SUPERSEDE` cannot use reason `new`; merged text must not introduce
digits or dates absent from the candidate, targets, or evidence; confidence
below threshold becomes review.

Budgets: max 6 changes and 3 new memories per run (highest confidence first),
max ~40 model calls, max 20 minutes. Over-budget items become `DEFER`.

### 6. Apply (`apply/ledger.py`)

All decisions are applied in **one SQLite transaction**. Every operation writes
an `ops` row with the candidate, decision, validator result, model, prompt
version, and run id.

- `SUPERSEDE`/`UPDATE`/`MERGE` set the old memory's `status='superseded'`,
  `valid_to` (when the world changed), and `superseded_at` (when we learned it).
  The old row and its evidence remain; a new row points back via
  `supersedes_id`.
- A loss guard computes active entry/char counts before and after; if more than
  25% would disappear, the transaction rolls back and the run is marked
  `blocked`.
- `DEFER` and validation failures create `proposals` for review.
- `forget` is explicit only: text/evidence becomes `[forgotten]`, the id and
  hash become a tombstone.

### 7. Synthesize and export (`synth/`, `apply/exports.py`)

- `DREAMS.md` gets one section per run: promoted/changed/deferred/skipped,
  evidence quotes, safety counters, duration.
- `MEMORY.candidate.md` / `USER.candidate.md` are rendered in Hermes format
  (`§` separators) within the configured char limits.
- `review.json` contains ready-to-run `memory` tool actions (`add`, `replace`
  with a unique `old_text` anchor, `remove`) plus skipped entries and limits.
- `apply-hermes` is opt-in and requires `hermes.write_mode = "apply"`; it takes
  a lock, checks the file mtime, writes a backup, and refuses a result over the
  character limit.

## Data model

See `src/minicpm_dream/store/schema.sql`. The important tables:

| Table | Purpose |
|---|---|
| `episodes` | normalized source turns (redacted text optional) |
| `memories` | current and historical memory rows, bitemporal |
| `memory_versions` | append-only text history per memory |
| `evidence` | verbatim quotes tied to episodes, with hashes |
| `ops` | every decision and whether it was applied |
| `proposals` | review queue with cooldowns |
| `runs` | window, config hash, model, counters, usage, status |
| `rejects` / `cooldowns` | anti-nag lifecycle state |
| `kv` | cursors and small state |
| `memories_fts` | FTS5 index (external content, triggers) |

Bitemporal semantics:

- `valid_from` / `valid_to` = when the fact was true in the world.
- `recorded_at` / `superseded_at` = when the system learned/retired it.
- Retrieval defaults to `status='active' AND valid_to IS NULL`.

## Invariants

1. No model output is written without a code-side schema and evidence check.
2. No memory is hard-deleted by the pipeline.
3. Every applied change is one transaction and one `ops` row.
4. A run cannot silently lose more than 25% of active memory.
5. Hermes writes require an explicit mode, a backup, and a lock.
6. Secrets are redacted before prompting and rejected before writing.
7. The gate reads only the ledger, never the Hermes state database.

## Extension points

- **New harness**: implement an iterator of `Episode` and register a CLI source;
  everything downstream is source-agnostic.
- **New model**: add a `ModelSpec` in `models.py` and use `backend=openai` with
  any OpenAI-compatible server, or `llama_cpp` for a local GGUF.
- **New task**: add a JSON schema in `schemas/`, a prompt in `prompts/`, and a
  validator. Keep the schema small; grammar-constrained decoding guarantees
  syntax, not meaning.
- **Retrieval**: `Retriever.retrieve()` is the seam for embeddings/rerankers.
