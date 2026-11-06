# Release checklist

## What ships

| Artifact | Status |
|---|---|
| `minicpm-dream` Python package (wheel + sdist) | builds |
| CLI: run, ingest, review, export, gate, search, explain, forget, maintain, index, eval, generate-eval, install-hermes-gate | implemented |
| `python -m minicpm_dream` | implemented |
| Default model registry + SHA-256 download | implemented |
| OpenBMB retrieval stack (optional extra) | implemented |
| Hermes adapter (read-only) + reviewed proposals | implemented |
| Docs, examples, synthetic suites | included |
| LoRA training scaffold | included, not run |
| Fine-tuned weights | not included (not required) |

## v1.0 gate

- [x] 79 tests, ruff, mypy
- [x] wheel builds and installs in a clean venv
- [x] offline mock demo
- [x] real MiniCPM5-2B run on the 8-message and 24-message fixtures
- [x] real Hermes session in a sandbox copy, cloud comparison
- [x] optimized local full-stack pass on real Hermes sandbox memory
      (12 chunks, 10 ledger writes, 1 MEMORY.md addition, 699 s)
- [x] secret/injection/retention/forget tests
- [x] loss guard and destructive-op guards
- [x] reviewed Hermes write path with lock, backup, and char limits
- [x] cron wake gate
- [x] release, security, contributing docs
- [ ] 100–300 fact local-vs-cloud evaluation (generator ready; run pending)
- [ ] macOS/Windows runtime smoke test

The unchecked items are confidence work, not missing implementation. Do not claim
"production benchmark" until the first one is run.

## Known limitations

- The 2B is stochastic: the 14-fact suite ranged 0.80–0.97 F1 across seeds.
- Quality mode costs ~5–7 minutes for a dense day on the reference CPU; cloud is
  ~10–20× faster.
- `MEMORY.md` is often near its 2,200-char limit; the tool reports pressure and
  refuses over-budget writes, but does not automatically compact the live profile.
- No encryption at rest; rely on disk encryption and a dedicated OS account.
- The Hermes `state.db` is an internal schema; the adapter introspects and degrades,
  but a future release can still require an adapter update.
- The OpenBMB Light models are optional and separately licensed for commercial use.
- No HTTP server or web UI; the CLI and JSONL are the integration surface.

## Publishing

```bash
make check
make demo
uv build
uv tool install dist/minicpm_dream-<version>-py3-none-any.whl
minicpm-dream doctor
```

Tag the version, attach the wheel/sdist, and keep `CHANGELOG.md` aligned.
