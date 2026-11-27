# Claude Code integration

Two ways to give Claude Code the memory, depending on whether you want it in the
loop or in front of it.

## 1. MCP server (turn-time, recommended)

`minicpm-dream serve` speaks MCP over stdio and needs no extra dependency. Add
the server to your Claude Code config (`mcp.json` next to this file):

```json
{
  "mcpServers": {
    "minicpm-dream": {
      "command": "minicpm-dream",
      "args": ["serve"]
    }
  }
}
```

The agent then has `recall`, `remember`, `explain`, `forget` and `memory_status`.
`recall` hits render like this, receipts included:

```text
- [mem_1793…] preferences/preference, 2mo ago: Josh prefers dark mode in all editors.
    ↳ 2026-11-04T23:12:00+00:00 "I prefer dark mode in all my editors"
```

A hit that is no longer current says so, so the agent does not state it as fact:

```text
- [mem_1793…] projects/fact: The ledger uses SQLite for storage.
    ! superseded: not current
    ! Replaced on 2026-11-04 by the ledger uses PostgreSQL; use the current version.
```

## 2. SessionStart hook (memory before the first token)

Copy `session-start-hook.sh` to `~/.claude/hooks/`, make it executable, and
register it in `~/.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          { "type": "command", "command": "~/.claude/hooks/minicpm-dream-recall.sh" }
        ]
      }
    ]
  }
}
```

The hook exits silently when the ledger has nothing relevant, so it never
blocks a session. Both approaches use the same ledger; running both is fine.

## Feeding Claude Code transcripts back in

```bash
minicpm-dream run --source claude_code --include-tool-messages --window day
```

This reads `~/.claude/projects/**/*.jsonl` (override with
`ingest.claude_code_dir`). Tool results are included because they are usually
where the day's real decisions live; sidechain/subagent turns are tagged as
`subagent` and excluded by default. Set `--fast --incremental` in cron.
