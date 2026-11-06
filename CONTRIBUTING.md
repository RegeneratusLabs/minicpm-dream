# Contributing

## Setup

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
make check
```

For retrieval work, also install `.[openbmb]`; those models are optional and must
not be required for the test suite.

## What to run before a PR

```bash
make check            # ruff + mypy src + pytest
make demo             # offline mock pipeline
make eval             # mock eval smoke test
```

If you change extraction, reconciliation, chunking, or prompts, also run a real
model pass on `evals/synthetic-days/full-day` and include the before/after
precision, recall, F1, temporal accuracy, and wall time in the PR. The suite is a
regression baseline, not a benchmark.

## Rules of the road

- Keep model output out of trust paths. Every write goes through the schema,
  evidence, target-ID, digit/date, and budget validators.
- Never add a path where model output can be executed, used as a filesystem path,
  or written to memory without review.
- Keep Hermes writes opt-in, locked, backed up, and char-budget checked.
- Add a test for every new guard or parser branch; use synthetic fixtures only.
- Prefer deterministic rules over extra model calls when the rule is unambiguous.
- Do not commit real transcripts, credentials, model weights, or generated data.

## Commit style

Small, descriptive commits. `CHANGELOG.md` gets a line for user-visible changes.
