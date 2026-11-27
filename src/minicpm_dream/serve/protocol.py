"""Minimal JSON-RPC 2.0 framing for the MCP stdio transport.

MCP over stdio is newline-delimited JSON: one JSON-RPC message per line, no
Content-Length header (that is the LSP convention). This module is deliberately
small and dependency-free so the server ships in the base install.
"""

from __future__ import annotations

import json
from typing import Any, TextIO

JSONRPC_VERSION = "2.0"
DEFAULT_PROTOCOL_VERSION = "2025-06-18"
SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class RpcError(Exception):
    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data

    def as_dict(self, request_id: Any) -> dict[str, Any]:
        error: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            error["data"] = self.data
        return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "error": error}


def result_response(request_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": result}


def read_message(stream: TextIO) -> dict[str, Any] | None:
    """Read one JSON-RPC message; None at EOF. Blank lines are skipped."""
    while True:
        line = stream.readline()
        if not line:
            return None
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RpcError(PARSE_ERROR, f"invalid JSON: {exc}") from exc
        if not isinstance(message, dict):
            raise RpcError(INVALID_REQUEST, "a JSON-RPC message must be an object")
        return message


def write_message(stream: TextIO, payload: dict[str, Any]) -> None:
    stream.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    stream.flush()


def is_notification(message: dict[str, Any]) -> bool:
    return "id" not in message


def validate_request(message: dict[str, Any]) -> tuple[Any, str]:
    """Return (id, method) or raise for a malformed request object."""
    if message.get("jsonrpc") != JSONRPC_VERSION:
        raise RpcError(INVALID_REQUEST, "jsonrpc must be '2.0'")
    method = message.get("method")
    if not isinstance(method, str) or not method:
        raise RpcError(INVALID_REQUEST, "method is required")
    return message.get("id"), method
