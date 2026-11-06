# Operations

## Requirements

- Python 3.11+ with SQLite FTS5 (the standard CPython builds include it).
- `uv` (recommended) or pip.
- llama.cpp `llama-server` for the local backend, or any OpenAI-compatible
  endpoint.
- Disk: ~1.6 GB for MiniCPM5-2B Q4_K_M, plus ledger/diary (small).

## Install

```bash
git clone <repo> minicpm-memory
cd minicpm-memory
uv venv .venv
source .venv/bin/activate
uv pip install -e .
minicpm-dream doctor
```

For development:

```bash
uv pip install -e ".[dev]"
uv run pytest
uv run ruff check .
uv run mypy src
```

## llama.cpp

The tool expects a `llama-server` binary with JSON-schema `response_format`
support. Three options:

1. **Package manager / existing build** — put `llama-server` on `PATH`.
2. **Pinned prebuilt release** (what this repo was tested with):

   ```bash
   mkdir -p data/tools
   curl -L -o data/tools/llama.tar.gz \
     https://github.com/ggml-org/llama.cpp/releases/download/b12144/llama-b12144-bin-ubuntu-x64.tar.gz
   tar -xzf data/tools/llama.tar.gz -C data/tools
   export LLAMA_SERVER="$PWD/data/tools/llama-b12144/llama-server"
   ```

3. **Build from source**:

   ```bash
   git clone https://github.com/ggml-org/llama.cpp
   cmake -B llama.cpp/build -DGGML_NATIVE=ON
   cmake --build llama.cpp/build -j --target llama-server
   ```

Pin the build you validated. Grammar behavior and chat-template handling change
between builds.

Configure the path in `config.toml` or `LLAMA_SERVER`:

```toml
[model]
llama_server = "/opt/llama.cpp/build/bin/llama-server"
```

## Model

```bash
minicpm-dream setup --list-models
minicpm-dream setup --download
minicpm-dream setup --check
```

The download is size-checked, SHA-256 verified against the pinned hash, and
written with `0600` permissions. For offline machines:

```bash
minicpm-dream setup --model-path /media/usb/MiniCPM5-2B-Q4_K_M.gguf
```

If upstream legitimately re-releases a file, set
`MINICPM_DREAM_MODEL_SHA256` to the published hash rather than disabling checks.

## CPU tuning

The tool starts llama-server with:

```text
-m <model> --host 127.0.0.1 --port <free> -c <context> -t <threads> \
-ngl <gpu_layers> --jinja --parallel 1
```

Recommended starting points:

| Machine | threads | context | gpu_layers |
|---|---|---|---|
| 4-core laptop | 3 | 4096 | 0 |
| 6-core desktop (reference) | 4–5 | 8192 | 0 |
| 8+ core | physical cores − 1 | 8192–16384 | 0 |
| small GPU (8 GB) | 4 | 8192 | 99 (all layers) |

Threads should be physical cores, not hyperthreads. More context costs KV cache
memory; 8K is enough because the pipeline chunks at ~6K tokens. Do not raise the
context to 128K on a small machine just because the model supports it.

`--max-minutes` (config `budgets.max_minutes`) stops a run that is taking too
long; the remaining candidates are deferred rather than half-applied.

## GPU endpoint

For vLLM/SGLang/Ollama/LM Studio:

```toml
[model]
backend = "openai"
base_url = "http://127.0.0.1:8000/v1"
name = "openbmb/MiniCPM5-2B"
api_key_env = ""        # or an env var name, never the key itself
temperature = 0.2
max_tokens = 1200
timeout_s = 180
```

The server must accept `response_format: {"type": "json_schema", ...}`. If it
does not, it will still work but JSON validity is no longer guaranteed by the
server; the parser and validators remain the backstop.

## Scheduling

### Hermes cron

See `docs/hermes-integration.md`. The short version:

```bash
minicpm-dream install-hermes-gate
hermes cron create "0 3 * * *" "Review minicpm-dream proposals" \
  --script minicpm-dream-gate.sh
```

### systemd timer (no Hermes)

```ini
# ~/.config/systemd/user/minicpm-dream.service
[Unit]
Description=minicpm-dream nightly pass

[Service]
Type=oneshot
WorkingDirectory=/home/you/miniCPM-memory
Environment=MINICPM_DREAM_DATA_DIR=/home/you/.local/share/minicpm-dream
Environment=LLAMA_SERVER=/home/you/llama.cpp/build/bin/llama-server
ExecStart=/home/you/miniCPM-memory/.venv/bin/minicpm-dream run --source hermes --window day
Nice=10
IOSchedulingClass=idle
```

```ini
# ~/.config/systemd/user/minicpm-dream.timer
[Unit]
Description=Run minicpm-dream nightly

[Timer]
OnCalendar=*-*-* 03:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now minicpm-dream.timer
journalctl --user -u minicpm-dream
```

## Backups and restore

Back up the data directory (ledger + diary + exports) and, if you use direct
writes, the Hermes memory directory.

```bash
sqlite3 ~/.local/share/minicpm-dream/ledger.db ".backup '/backup/ledger.db'"
cp ~/.local/share/minicpm-dream/DREAMS.md /backup/
cp ~/.hermes/memories/MEMORY.md /backup/
```

Restore the ledger by stopping the agent/cron, replacing `ledger.db` (and
`-wal`/`-shm`), and starting again. Hermes memory restore is just the file plus
a restart of any open session (the system prompt snapshot is taken at session
start).

## Upgrades

- The ledger has an explicit `PRAGMA user_version`. A newer schema refuses to
  open in an older build rather than corrupting data.
- Re-run `minicpm-dream doctor` after every upgrade.
- If you use the Hermes gate, re-run `install-hermes-gate --check`; a stale copy
  will not include new behavior.
- `DREAMS.md` is append-only and trimmed by section count; it is safe to keep.

## Troubleshooting

| Symptom | Check |
|---|---|
| `model file not found` | `minicpm-dream setup --check`; or pass `--model-path` |
| `llama-server not found` | `LLAMA_SERVER`, `model.llama_server`, or `PATH` |
| server starts then exits | `data/logs/llama-server.log`; usually insufficient RAM or a bad GGUF |
| every candidate rejected | run `--verbose`; inspect `rejected` reasons in the JSON output |
| no episodes | timezone/window; run `minicpm-dream ingest ... --out -` |
| gate always wakes | stale `review.json` actions; apply/reject them |
| slow run | reduce `threads`, `max_chunk_tokens`, or the number of episodes; check swap |
| JSON schema errors | server may not support `response_format`; use llama.cpp or a recent vLLM |

## Performance notes

- The pass is CPU-bound on the model and I/O-bound on SQLite; it is a batch job,
  not interactive.
- The dominant cost is generation, not retrieval. Cap `max_items` and
  `max_tokens` before buying a bigger machine.
- A quiet day with no durable memories still costs one extractor call per chunk.
  The wake gate only helps if you run the pass conditionally; this tool always
  needs to read the day to know it was quiet.
- On the reference 6-core CPU, measure with `time minicpm-dream run ...`; do not
  infer throughput from the model card.
