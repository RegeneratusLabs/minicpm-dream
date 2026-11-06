# minicpm-dream

**Your agent forgets the day. This gives it back.**

An agent is good at the conversation in front of it and bad at everything before
it. A day of work leaves hundreds of messages behind — a preference mentioned
once, a decision, a name, an open loop, a correction — and tomorrow the context
window has moved on.

`minicpm-dream` is a nightly pass over that day. It reads the transcript, asks a
small [MiniCPM](https://huggingface.co/openbmb) model to pull out what is worth
keeping, checks every claim against the words that produced it, and writes the
result to a private ledger on your machine. In the morning there is a memory your
agent can use and a diary you can read.

It runs on your own hardware. No cloud account, no subscription, and no
conversation leaves your disk.

## What you get

| | |
|---|---|
| **A memory that persists** | Facts, preferences, projects, people, decisions and open loops, stored with absolute dates and full history. |
| **A diary, not a dump** | `DREAMS.md` says what was promoted, what changed and what needs a human — in sentences. |
| **Receipts** | Every stored memory carries the verbatim quote it came from. No quote, no memory. |
| **Corrections without erasure** | A changed fact supersedes the old one; nothing is silently deleted. `forget` leaves a tombstone. |
| **Reviewed writes** | New or contested entries go to a review file first. Applying them needs an explicit flag and makes a backup. |
| **Local by default** | Model, retrieval, ledger and exports all live on your machine. |

## Try it in two minutes

No model download: the repo ships a scripted mock that runs the whole pipeline
offline.

```bash
git clone https://github.com/RegeneratusLabs/minicpm-dream
cd minicpm-dream
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"

minicpm-dream run --source jsonl \
  --input evals/synthetic-days/basic/episodes.jsonl \
  --backend mock --mock-script evals/synthetic-days/basic/mock_script.json
```

```text
run run_… ok: 6 episodes → 3 candidates → 3 verified → 3 changes (0 deferred)
diary:   ~/.local/share/minicpm-dream/DREAMS.md
exports: ~/.local/share/minicpm-dream/exports/review.json
```

Then read what it wrote, and ask it why:

```bash
cat ~/.local/share/minicpm-dream/DREAMS.md
minicpm-dream search "dentist"
minicpm-dream explain <memory-id>
```

The mock proves the plumbing, not the model. Model quality and its measured
limits are in [docs/models.md](docs/models.md).

## A diary you can read

Here is a real mock run, trimmed of ids:

```markdown
## 2026-11-06 14:07 — 2026-11-04T14:00:00+00:00 → 2026-11-05T14:00:00+00:00

Run `run_1793…` · model `mock` · 8 episodes → 3 candidates → 3 verified · 3 changes

### Promoted
- **preferences** — Josh prefers dark mode in all editors.
  ↳ "I prefer dark mode in all my editors"
- **projects** — The miniCPM-memory project uses Python 3.12 and SQLite for storage.
  ↳ "The miniCPM-memory project uses Python 3.12 and SQLite for storage."
- **open_loops** — Josh has a dentist appointment. (on 2026-11-06)
  ↳ "dentist tomorrow at 3pm"

### Safety
- loss guard: ok (0→3 entries, 0→152 chars)
- ingest: kept 8/8, redactions 1, quarantined 0, duplicates 0
```

The last memory shows the split of work. The model was allowed to read
"tomorrow"; code resolved it to 2026-11-06 before anything was stored, so the
memory does not go stale when read next week. The quote stays as it was said,
because it is the receipt.

## How it works

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

- **Ingest** filters noise, redacts secrets, quarantines prompt injection and
  de-duplicates. It reads Hermes `state.db` read-only, or any JSONL transcript.
- **Prepare** chunks the day and resolves relative dates ("tomorrow", "last
  Saturday") into absolute dates in code, not in the model.
- **Extract** asks the model for typed memories — facts, events, preferences,
  tasks, decisions, open loops — one constrained JSON call per chunk.
- **Verify** rejects any memory whose quote is not present in the source turn.
- **Reconcile** decides ADD, UPDATE, SUPERSEDE, MERGE, NOOP or DEFER against
  what is already known, and escalates uncertain decisions for review.
- **Apply** writes a bitemporal SQLite ledger, a daily `DREAMS.md`, and (for
  Hermes) reviewed proposals rather than direct edits.

A deeper tour is in [docs/architecture.md](docs/architecture.md).

## Built for a small model

The default is a 2B model, and a 2B model should not be trusted to invent dates,
silently delete memories or decide merges alone. So the protocol is narrow and
the safeguards are outside the model: dates are resolved in code, every memory
needs a verbatim quote, merges are checked for invented numbers, supersession
keeps history, and uncertain operations become review items instead of writes.
A run that would drop more than 25% of active memory is rolled back.

The design borrows from hermes-dreaming, OpenClaw dreaming, Letta sleep-time
compute and mem0, and from a 2026 LoCoMo study of small memory managers. The
short version of that research: a narrow typed protocol plus absolute dates at
write time is what lets a small model manage memory at all. See
[docs/prior-art.md](docs/prior-art.md).

Hermes Agent is the first adapter. The core is harness-agnostic: any transcript
that exports to JSONL works.

## Install for real

Default model: **MiniCPM5-2B Q4_K_M** (2.52B parameters, 131K context,
Apache-2.0, ~1.56 GB).

```bash
# Download + checksum-verify the GGUF into the data directory
minicpm-dream setup --download
minicpm-dream setup --download-embeddings   # optional OpenBMB retrieval models

# Install llama.cpp (one option)
#   https://github.com/ggml-org/llama.cpp  → put `llama-server` on PATH
# or point at an existing binary:
#   model.llama_server = "/opt/llama.cpp/build/bin/llama-server"

# Check the environment, then run a day
minicpm-dream doctor
minicpm-dream run --source hermes --window day --dry-run
minicpm-dream run --source hermes --window day
```

On a CPU-only machine the nightly pass is a batch job; a typical day is a handful
of 6K-token chunks. On the reference machine (6-core i5-8500T, 15 GB RAM, no
GPU) a dense day takes roughly 5–7 minutes in quality mode; `--fast` trades
recall for time.

For a GPU endpoint (vLLM/SGLang/Ollama/LM Studio):

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

See [docs/hermes-integration.md](docs/hermes-integration.md) for the two cron
patterns (no-agent nightly pass, and gate + review turn).

## Privacy and safety

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

The full data-flow and threat model are in [docs/privacy.md](docs/privacy.md).

## Model stack

| Tier | Model | When |
|---|---|---|
| Default | MiniCPM5-2B Q4_K_M | CPU/local nightly pass |
| Cheap verifier | MiniCPM5-1B Q4_K_M | second opinion / triage |
| Quality | MiniCPM4.1-8B Q4_K_M | GPU or weekly deep pass |
| Long context | MiniCPM-SALA | 1M-token whole-week pass on a large GPU |
| Embeddings | **MiniCPM-Embedding-Light** (440M, 1024-dim) | local hybrid retrieval |
| Reranker | **MiniCPM-Reranker-Light** (1.2B) | local reranking |

Retrieval is a complete OpenBMB stack: **MiniCPM-Embedding-Light** (440M,
1024-dim, 8K tokens, dense + sparse) and **MiniCPM-Reranker-Light** (1.2B
cross-encoder). Both run locally; the models use the MiniCPM model license
(free academic, commercial registration). See [docs/models.md](docs/models.md)
for CPU timings and the comparison with a cloud model.

## Commands

```text
minicpm-dream doctor
minicpm-dream setup [--download|--check|--list-models]
minicpm-dream ingest --source hermes --window day --out episodes.jsonl
minicpm-dream run --source hermes --window day [--quality|--fast] [--dry-run] [--apply-hermes]
minicpm-dream review list|show|approve|reject
minicpm-dream export [--apply-hermes]
minicpm-dream gate
minicpm-dream search "query"
minicpm-dream explain <memory-id>
minicpm-dream forget <memory-id> --yes
minicpm-dream maintain --archive-days 180
minicpm-dream index                 # build/refresh OpenBMB memory vectors
minicpm-dream eval --suite evals/synthetic-days/full-day
minicpm-dream generate-eval --out evals/generated/my-suite --days 10 --facts 120
```

## Status and limits

**v1.0.0-rc1, code-complete for v1.** The nightly pass, reviewed Hermes writes,
safety guards, OpenBMB retrieval, CLI, packaging and tests are in place. The
remaining release-confidence work is an expanded 100–300 fact evaluation, an
optimised multi-session Hermes pass, and macOS/Windows smoke tests. The release
checklist and known gaps are in [docs/release.md](docs/release.md).

Measured on the 24-message / 14-fact day: the optimised 2B pipeline reaches
**0.80–0.97 F1** (best run P 0.93 / R 1.00 / F1 0.97 in 334 s), versus
**0.81–0.90** for DeepSeek Flash. The remaining gap is run-to-run variance, not
average quality. These are regression comparisons on one synthetic scenario, not
a benchmark; read [docs/models.md](docs/models.md) before quoting them.

Known limits:

- A 2B model can still miss a fact, over-compress a quote or choose a wrong
  operation. The design turns those failures into rejections or review items,
  not silent bad memory.
- Quality mode costs several minutes for a dense day on a CPU. Cloud models are
  10–20× faster.
- English is the primary tested language; MiniCPM is bilingual CN/EN, but other
  languages need their own eval fixtures.
- No encryption at rest. The ledger and exports are `0600` in a `0700` directory;
  use full-disk encryption if that matters.
- `MEMORY.md` is often near its 2,200-char limit; the tool reports pressure and
  refuses over-budget writes, but does not automatically compact the live profile.
- No HTTP server or web UI; the CLI and JSONL are the integration surface.
- Fine-tuning is not part of v1. [docs/models.md](docs/models.md) records where
  zero-shot fails before any training work starts.

## Development

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```

The test suite uses a deterministic mock backend; integration tests that need a
real model are marked `integration` and skipped when the GGUF/llama-server is
absent.

## Scaling the evaluation and training

Generate a larger deterministic suite and build SFT data with the production
prompts:

```bash
minicpm-dream generate-eval --out evals/generated/gpu-eval --seed 42 --days 10 --facts 120
.venv/bin/python training/build_dataset.py --suite evals/generated/gpu-eval --out training/data/gpu-eval
```

[training/README.md](training/README.md) has the GPU LoRA recipe, evaluation
command and acceptance gates. Fine-tuning is optional: the optimised zero-shot
pipeline already reaches cloud-level average quality; use the GPU first for the
expanded suite and an 8B baseline, then a targeted LoRA only if the floor is
still too low.

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).

## License

Code: MIT. Model weights are downloaded separately under their own licenses; see
[NOTICE](NOTICE). This is a community project, not affiliated with OpenBMB.
