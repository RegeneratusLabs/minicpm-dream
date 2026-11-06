# Hermes Agent integration

Hermes is the first adapter, but the integration is intentionally shallow:
**read transcripts read-only, write candidates for review**. Nothing in this
document requires patching Hermes.

## What is read

| Path | Access | Purpose |
|---|---|---|
| `~/.hermes/state.db` | read-only SQLite URI, `PRAGMA query_only=ON` | sessions and messages |
| `~/.hermes/memories/MEMORY.md` | read-only | current agent notes, diff target |
| `~/.hermes/memories/USER.md` | read-only | current user profile, diff target |
| `~/.hermes/config.yaml` | read-only, narrow regex | char limits only |
| `~/.hermes/cron/jobs.json` | **never** | not needed |

The adapter introspects `PRAGMA table_info` and uses only columns that exist:
`messages.active`, `compacted`, `_compressed_summary`, `sessions.hidden`,
`parent_session_id`, `chat_type`, etc. If a future Hermes release changes the
schema, `doctor` and `ingest` say exactly what is missing instead of returning
wrong data.

Default filters:

- roles: `user`, `assistant` (tool output is excluded);
- exclude session sources: `cron`, `subagent`, `oneshot`, `tool`;
- exclude child sessions (`parent_session_id` not null) and hidden sessions;
- exclude compacted messages and compression summaries;
- exact-duplicate suppression within a window;
- secrets redacted before the model sees anything.

Tune with:

```toml
[ingest]
trusted_sources = ["telegram", "desktop", "tui", "cli", "local"]
exclude_sources = ["cron", "subagent", "oneshot", "tool"]
include_tool_messages = false
```

## Hermes memory format

Hermes stores plain entries separated by a line containing `§`:

```text
Entry one.
§
Entry two.
```

Limits in a default install:

- `MEMORY.md`: 2,200 chars
- `USER.md`: 1,375 chars

The adapter reads the real limits from `config.yaml` (`memory.memory_char_limit`,
`memory.user_char_limit`) when available. Candidate files are budgeted to those
limits; entries that do not fit are listed under `skipped` in `review.json`
rather than silently dropped.

Section mapping:

| Ledger section | Export target |
|---|---|
| `identity`, `preferences` | `USER.md` |
| `environment`, `projects`, `procedures`, `lessons`, `people` | `MEMORY.md` |
| `open_loops` | `DREAMS.md` only |

Open loops are intentionally not promoted into a bounded profile file; they are
time-sensitive and would crowd out durable facts.

## Review workflow

```bash
minicpm-dream run --source hermes --window day
```

This writes:

```text
<data>/exports/MEMORY.candidate.md
<data>/exports/USER.candidate.md
<data>/exports/review.json
```

`review.json` contains actions in Hermes `memory` tool shape:

```json
{
  "target": "user",
  "action": "replace",
  "old_text": "prefers dark mode in VS Code",
  "content": "prefers dark mode in all editors",
  "memory_id": "mem_...",
  "reason": "supersede"
}
```

Three ways to apply:

1. **Agent review turn** (recommended). The Hermes agent reads `review.json`
   and calls its own `memory` tool. It can ask you about ambiguous items and its
   write path is the one Hermes already audits.
2. **`minicpm-dream export --apply-hermes`**. Direct file write with a lock,
   backup, mtime check, and char-limit enforcement. Requires
   `hermes.write_mode = "apply"`.
3. **Manual**. Read the candidate files and copy what you want.

`review approve <proposal-id>` is for deferred *ledger* decisions. Proposals
created by a run are visible in `minicpm-dream review list`; approving one marks
it approved. Hermes file actions are the separate `review.json` path.

## Cron patterns

Hermes cron scripts must live in `~/.hermes/scripts/` (Hermes refuses symlinks).
`minicpm-dream install-hermes-gate` copies a small wrapper there.

### Pattern A — no-agent nightly pass (cheapest)

The script does everything with the local model; Hermes only delivers the
report. Create a normal Hermes cron job with `--script`; the script's stdout is
the report, no big-model turn.

```bash
hermes cron create "0 3 * * *" \
  "Run the local memory consolidation pass" \
  --script minicpm-dream-nightly.sh
```

Wrapper:

```bash
#!/usr/bin/env bash
set -euo pipefail
exec /home/you/miniCPM-memory/.venv/bin/minicpm-dream run \
  --source hermes --window day
```

### Pattern B — gate then review turn

Use this when a human or the Hermes agent should decide contested items.

```bash
#!/usr/bin/env bash
# ~/.hermes/scripts/minicpm-dream-gate.sh (installed by minicpm-dream)
exec /path/to/python -m minicpm_dream.cli gate --source hermes "$@"
```

- If nothing needs review, the gate prints `{"wakeAgent": false}` and Hermes
  skips the agent entirely (zero LLM tokens).
- If there are proposals or review actions, it prints
  `{"wakeAgent": true, "context": {...}}` and the job prompt can tell the agent
  to read `<data>/exports/review.json` and apply only items it can justify.

Example prompt:

```text
Read {data_dir}/exports/review.json. For each action:
- if it is a clear, evidence-backed improvement, apply it with the memory tool;
- if it would remove or contradict something the user confirmed, ask instead;
- if unsure, change nothing.
Report what you applied, what you left, and why. Never invent an old_text anchor.
```

The gate never touches `state.db`; it reads the ledger's `proposals` table and
`review.json`. This follows Hermes' own warning that `state.db` is an internal
schema.

### Schedules

- Nightly at 03:00 local is a good default.
- A weekly `--window week` run is useful after a busy week; it is more expensive.
- Do not run two passes concurrently: a file lock in the data directory
  serializes them.

## Direct-write mode

Set this only if you accept the blast radius:

```toml
[hermes]
write_mode = "apply"
memory_char_limit = 2200
user_char_limit = 1375
```

`--apply-hermes` then writes both candidate files, with:

- a lock on `MEMORY.md.lock` / `USER.md.lock` (the same files Hermes uses);
- an mtime check to detect a concurrent Hermes write;
- a timestamped backup in `<data>/backups/`;
- atomic replacement and 0600 permissions;
- refusal if the result exceeds the configured limit.

If Hermes is configured with `memory.write_approval: true`, direct writes bypass
that approval queue. Prefer proposals in that setup.

## Concurrency and safety

- One consolidation run at a time (`<data>/run.lock`).
- Hermes keeps its own memory lock; we take the same lock file name.
- Every applied change is in one SQLite transaction.
- A 25% active-memory loss rolls back the whole run and marks it `blocked`.
- Secrets are screened before prompting and before writing.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `state.db is not usable: missing: messages.active` | Hermes schema drift | upgrade or use `minicpm-dream ingest --source jsonl` with an export |
| `apply-hermes requested but hermes.write_mode=proposals` | safety default | set `write_mode = "apply"` or use the review turn |
| `replace anchor matched 0 entries` | memory changed since export | re-run `export`, then apply |
| gate always wakes | stale `review.json` with actions | apply/reject them, then `export` again |
| no episodes | wrong timezone or window | pass `--since/--until` or check `config.timezone` |
