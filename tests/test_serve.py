from __future__ import annotations

import io
import json

from minicpm_dream.serve.server import serve_stdio
from minicpm_dream.store.db import Store


def _run(cfg, lines: list[dict]) -> list[dict]:
    stdin = io.StringIO("".join(json.dumps(line) + "\n" for line in lines))
    stdout = io.StringIO()
    serve_stdio(cfg, stdin=stdin, stdout=stdout)
    return [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]


def test_initialize_and_tools_list(cfg):
    responses = _run(
        cfg,
        [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "ping"},
        ],
    )
    assert len(responses) == 3  # the notification is not answered
    init = responses[0]["result"]
    assert init["protocolVersion"] == "2025-06-18"
    assert init["serverInfo"]["name"] == "minicpm-dream"
    assert init["capabilities"]["tools"]
    names = [tool["name"] for tool in responses[1]["result"]["tools"]]
    assert {"recall", "remember", "explain", "forget", "memory_status"} <= set(names)
    assert responses[2]["result"] == {}


def test_tool_call_recall_returns_text_and_structured_content(cfg):
    with Store(cfg.paths.db) as store:
        store.insert_memory(
            kind="preference",
            section="preferences",
            text="Josh prefers dark mode in all editors.",
            evidence=[{"episode_id": "e", "quote": "dark mode everywhere", "verified": True}],
            salience=0.9,
            confidence=0.9,
        )
    responses = _run(
        cfg,
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "recall", "arguments": {"query": "editor theme"}},
            }
        ],
    )
    result = responses[0]["result"]
    assert "dark mode" in result["content"][0]["text"]
    assert "dark mode everywhere" in result["content"][0]["text"]
    assert result["structuredContent"]["hits"][0]["receipts"][0]["quote"] == "dark mode everywhere"


def test_tool_call_remember_then_recall(cfg):
    responses = _run(
        cfg,
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "remember",
                    "arguments": {"text": "The homelab runs Ubuntu 24.04.", "section": "environment"},
                },
            },
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "recall", "arguments": {"query": "what does the homelab run"}},
            },
        ],
    )
    assert responses[0]["result"]["structuredContent"]["op"] == "ADD"
    hits = responses[1]["result"]["structuredContent"]["hits"]
    assert any("Ubuntu 24.04" in hit["text"] for hit in hits)


def test_remember_rejects_a_secret_with_is_error(cfg):
    responses = _run(
        cfg,
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "remember", "arguments": {"text": "key sk-abcdefghijklmnopqrstuvwxyz123456"}},
            }
        ],
    )
    result = responses[0]["result"]
    assert result["isError"] is True
    assert "Not stored" in result["content"][0]["text"]


def test_unknown_method_and_bad_params_are_rpc_errors(cfg):
    responses = _run(
        cfg,
        [
            {"jsonrpc": "2.0", "id": 1, "method": "resources/list"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "recall", "arguments": {}}},
        ],
    )
    assert responses[0]["error"]["code"] == -32601
    assert responses[1]["error"]["code"] == -32602


def test_forget_requires_confirmation(cfg):
    with Store(cfg.paths.db) as store:
        mid = store.insert_memory(
            kind="fact", section="projects", text="A secret plan.", evidence=[], salience=0.5
        )
    responses = _run(
        cfg,
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "forget", "arguments": {"memory_id": mid}},
            },
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "forget", "arguments": {"memory_id": mid, "confirm": True}},
            },
        ],
    )
    assert responses[0]["result"]["isError"] is True
    assert responses[1]["result"]["structuredContent"]["forgotten"] is True
