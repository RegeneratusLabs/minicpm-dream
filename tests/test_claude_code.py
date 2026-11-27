from __future__ import annotations

import json
from pathlib import Path

from minicpm_dream.ingest.claude_code import iter_claude_code, session_files
from minicpm_dream.ingest.jsonl import read_jsonl


def _write_session(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "project" / "session-1.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "type": "user",
            "sessionId": "sess-1",
            "timestamp": "2026-11-05T09:00:00+10:00",
            "cwd": "/home/josh/projects/x",
            "message": {"role": "user", "content": "I prefer dark mode in all my editors."},
        },
        {
            "type": "assistant",
            "sessionId": "sess-1",
            "timestamp": "2026-11-05T09:00:30+10:00",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Noted, dark mode everywhere."},
                    {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
                ],
            },
        },
        {
            "type": "user",
            "sessionId": "sess-1",
            "timestamp": "2026-11-05T09:01:00+10:00",
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "total 12\ndrwxr-xr-x 2 josh josh 4096 ."}
                ],
            },
        },
        {
            "type": "summary",
            "sessionId": "sess-1",
            "timestamp": "2026-11-05T09:05:00+10:00",
            "summary": "ignored",
        },
        {
            "type": "user",
            "sessionId": "sess-1",
            "isSidechain": True,
            "timestamp": "2026-11-05T09:02:00+10:00",
            "message": {"role": "user", "content": "subagent scratch note"},
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    return path


def test_session_files_and_message_extraction(tmp_path):
    _write_session(tmp_path / "projects")
    files = session_files(tmp_path / "projects")
    assert len(files) == 1
    episodes = list(
        iter_claude_code(tmp_path / "projects", tz="Australia/Brisbane", include_tool_results=True)
    )
    roles = [ep.role for ep in episodes]
    sources = [ep.source for ep in episodes]
    assert "tool" in roles
    assert "subagent" in sources
    assert all(ep.role != "assistant" or ep.text for ep in episodes)
    # The tool_use block is intent, not evidence, and must not become an episode.
    assert not any("called Bash" in ep.text for ep in episodes)
    assert any("total 12" in ep.text for ep in episodes)


def test_tool_results_can_be_excluded(tmp_path):
    _write_session(tmp_path / "projects")
    episodes = list(
        iter_claude_code(tmp_path / "projects", tz="Australia/Brisbane", include_tool_results=False)
    )
    assert all(ep.role != "tool" for ep in episodes)
    assert not any("total 12" in ep.text for ep in episodes)


def test_jsonl_adapter_accepts_nested_message_records(tmp_path):
    path = tmp_path / "nested.jsonl"
    path.write_text(
        json.dumps(
            {
                "message": {"role": "user", "content": [{"type": "text", "text": "I use SQLite."}]},
                "timestamp": "2026-11-05T09:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    episodes = list(read_jsonl(path, tz="UTC"))
    assert len(episodes) == 1
    assert episodes[0].text == "I use SQLite."
    assert episodes[0].role == "user"


def test_collect_episodes_keeps_tool_output_and_drops_sidechains(cfg, tmp_path):
    from minicpm_dream.pipeline import collect_episodes
    from minicpm_dream.util.timeutil import parse_timestamp

    root = tmp_path / "claude"
    _write_session(root)
    cfg.ingest.claude_code_dir = str(root)
    cfg.ingest.include_tool_messages = True
    since = parse_timestamp("2026-11-05T00:00:00", cfg.timezone)
    until = parse_timestamp("2026-11-06T00:00:00", cfg.timezone)
    episodes, stats = collect_episodes(
        cfg, source="claude_code", since=since, until=until, input_path=None, limit=None
    )
    assert any(ep.role == "tool" for ep in episodes)
    assert not any(ep.source == "subagent" for ep in episodes)
    assert stats.as_dict()["dropped_source"] >= 1
