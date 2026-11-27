#!/usr/bin/env bash
# Claude Code SessionStart hook: put relevant memory in front of the agent before
# it answers. Wire it up in ~/.claude/settings.json under hooks.SessionStart with
#   {"hooks": [{"type": "command", "command": "~/.claude/hooks/minicpm-dream-recall.sh"}]}
#
# Reads the hook payload on stdin, asks minicpm-dream for a few memories, and
# emits additionalContext. Exits 0 silently when there is nothing relevant, so a
# cold ledger never blocks a session.
set -euo pipefail

payload="$(cat || true)"
prompt="$(printf '%s' "$payload" | python3 -c 'import json,sys
try:
    print(json.load(sys.stdin).get("prompt", ""))
except Exception:
    print("")')"

if [ -z "${prompt//[[:space:]]/}" ]; then
  exit 0
fi

context="$(minicpm-dream recall "$prompt" --k 6 --stats 2>/dev/null || true)"
if [ -z "${context//[[:space:]]/}" ]; then
  exit 0
fi

MINICPM_CONTEXT="$context" python3 - <<'PY'
import json
import os

context = os.environ["MINICPM_CONTEXT"]
print(json.dumps({
    "hookSpecificOutput": {
        "hookEventName": "SessionStart",
        "additionalContext": (
            "Private memory from earlier sessions. Use it only where it is relevant, "
            "cite the receipts, and treat anything flagged stale, contested or "
            "superseded as history rather than current fact.\n\n" + context
        ),
    }
}))
PY
