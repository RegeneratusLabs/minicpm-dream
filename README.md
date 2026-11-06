# minicpm-dream

**Local MiniCPM memory consolidation for agent harnesses.**

An agent's day leaves behind thousands of messages: facts, preferences, decisions,
open loops, corrections, and a lot of noise. `minicpm-dream` reads a window of
that history, asks a small local MiniCPM model to extract typed memories, checks
every claim against its source quote, reconciles it with what is already known,
and writes a bitemporal SQLite ledger plus a human-readable `DREAMS.md` diary.

Hermes Agent is the first adapter. The core is harness-agnostic: any transcript
that can be exported to JSONL works.

```text
Hermes state.db ─┐
JSONL export ────┤→ filter/redact → chunk + date anchors → MiniCPM extract
                 │                                          ↓
                 │                                    evidence verifier
                 │                                          ↓
                 └──────────────→ hybrid retrieval → reconcile (ADD/UPDATE/
                                    SUPERSEDE/MERGE/NOOP/DEFER)
                                                           ↓
                              SQLite ledger + DREAMS.md + reviewed Hermes proposals
```

## Why this exists

- **hermes-dreaming** is deterministic and wakes the host agent to apply
  proposals. It is excellent, but it deliberately calls no LLM.
- **OpenClaw dreaming** uses light/deep/REM phases and a dream diary, powered by
  the runtime model.
- **Letta sleep-time compute** shows consolidation belongs off the interactive
  path.
- **mem0 v3** moved to ADD-only and dropped write-time contradiction handling.
  This project keeps write-time **supersede** with full history.
- A 2026 LoCoMo study of small local memory managers found that a **narrow typed
  protocol** is what lets a small model manage memory, that **absolute-date
  grounding at write time** carries the temporal advantage, and that periodic
  consolidation/merging is the missing maintenance layer. This project is built
  around those findings.

The honest constraint: a 2B model should not be trusted to invent dates, silently
delete memories, or decide merges alone. So dates are resolved in code, every
memory needs a verbatim quote, merges are checked for invented numbers, and
uncertain operations become review items instead of writes.

## Quickstart (no model required)

```bash
git clone <this repo> minicpm-memory && cd minicpm-memory
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# 1. See what the tool can see
minicpm-dream doctor

# 2. Run the full pipeline with scripted mock generations
minicpm-dream run --source jsonl \
  --input evals/synthetic-days/basic/episodes.jsonl \
  --backend mock --mock-script evals/synthetic-days/basic/mock_script.json

# 3. Inspect the ledger and the diary
minicpm-dream search "dentist"
minicpm-dream explain <memory-id>
cat ~/.local/share/minicpm-dream/DREAMS.md
```

The mock run proves the plumbing. It does not prove model quality; that is what
`minicpm-dream eval` and `docs/models.md` are for.

**Status:** v0.1.0. On the reference 6-core CPU, MiniCPM5-2B Q4_K_M scores
P=1.0 / R=1.0 / schema=1.0 / temporal=1.0 on the 8-message synthetic fixture in
~49 s. That is one small case with three gold facts, not a benchmark. The repo
also documents the interface fixes the 2B needed (thinking mode, date refs,
retractions, task typing) in `docs/models.md`.

## Local MiniCPM setup

Default model: **MiniCPM5-2B Q4_K_M** (2.52B parameters, 131K context,
Apache-2.0, ~1.56 GB).

```bash
# Download + checksum-verify the GGUF into the data directory
minicpm-dream setup --download

# Install llama.cpp (one option)
#   https://github.com/ggml-org/llama.cpp  → put `llama-server` on PATH
# or point at an existing binary:
#   model.llama_server = "/opt/llama.cpp/build/bin/llama-server"

minicpm-dream run --source hermes --window day --dry-run
minicpm-dream run --source hermes --window day
```

On a CPU-only machine the nightly pass is a batch job; a typical day is a handful
of 6K-token chunks. For a GPU endpoint (vLLM/SGLang/Ollama/LM Studio):

```toml
[model]
backend = "openai"
base_url = "http://127.0.0.1:8000/v1"
name = "openai/MiniCPM5-2B"   # whatever the server serves
```

## Hermes integration

`minicpm-dream` reads `~/.hermes/state.db` read-only and imports
`MEMORY.md` / `USER.md` only to compute diffs. It never edits Hermes memory
unless you explicitly run with `--apply-hermes` **and** set
`hermes.write_mode = "apply"`.

The default path is reviewed proposals:

```bash
minicpm-dream run --source hermes --window day
# writes:
#   ~/.local/share/minicpm-dream/exports/MEMORY.candidate.md
#   ~/.local/share/minicpm-dream/exports/USER.candidate.md
#   ~/.local/share/minicpm-dream/exports/review.json   ← ready-to-run memory tool calls
```

A Hermes cron gate is included:

```bash
minicpm-dream install-hermes-gate
# ~/.hermes/scripts/minicpm-dream-gate.sh emits {"wakeAgent": false} when there
# is nothing to review, and {"wakeAgent": true, "context": {...}} otherwise.
```

See `docs/hermes-integration.md` for the two cron patterns (no-agent nightly pass
and gate + review turn).

## Safety model

- **Local by default.** No network calls except the explicit model download or a
  backend you configure.
- **Data, not instructions.** Transcripts are screened for prompt injection and
  invisible Unicode before they reach the model; model output is never executed.
- **Secrets are redacted** before prompting and rejected before writing.
- **Evidence or nothing.** Every stored memory has a verbatim quote checked
  against the source turn.
- **Bitemporal, never destructive.** Superseded facts keep their history;
  `forget` leaves a tombstone. A run that would drop more than 25% of active
  memory is rolled back.
- **Reviewed writes.** New/contested Hermes entries go through
  `exports/review.json`; `apply-hermes` requires an explicit mode and creates a
  backup.

## Model stack

| Tier | Model | When |
|---|---|---|
| Default | MiniCPM5-2B Q4_K_M | CPU/local nightly pass |
| Cheap verifier | MiniCPM5-1B Q4_K_M | second opinion / triage |
| Quality | MiniCPM4.1-8B Q4_K_M | GPU or weekly deep pass |
| Long context | MiniCPM-SALA | 1M-token whole-week pass on a large GPU |
| Embeddings | BAAI/bge-small-en-v1.5 (MIT) | optional hybrid retrieval |

MiniCPM-Embedding is deliberately not used: it is 2.4B, capped at 512 tokens, and
has a separate model license. See `docs/models.md` for the evaluation table and
licensing notes.

## Commands

```text
minicpm-dream doctor
minicpm-dream setup [--download|--check|--list-models]
minicpm-dream ingest --source hermes --window day --out episodes.jsonl
minicpm-dream run --source hermes --window day [--dry-run] [--apply-hermes]
minicpm-dream review list|show|approve|reject
minicpm-dream export [--apply-hermes]
minicpm-dream gate
minicpm-dream search "query"
minicpm-dream explain <memory-id>
minicpm-dream forget <memory-id> --yes
minicpm-dream maintain --archive-days 180
minicpm-dream eval --suite evals/synthetic-days
```

## Development

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```

The test suite uses a deterministic mock backend; integration tests that need a
real model are marked `integration` and skipped when the GGUF/llama-server is
absent.

## Limitations

- A 2B model can still miss a fact, over-compress a quote, or choose a wrong
  operation. The design turns those failures into rejections or review items,
  not silent bad memory.
- English is the primary tested language; MiniCPM is bilingual CN/EN, but other
  languages need their own eval fixtures.
- No encryption at rest. The ledger and exports are `0600` in a `0700` directory;
  use full-disk encryption if that matters.
- Fine-tuning is not part of v1. `docs/models.md` records where zero-shot fails
  before any training work starts.

## License

Code: MIT. Model weights are downloaded separately under their own licenses;
see `NOTICE`. This is a community project, not affiliated with OpenBMB.
