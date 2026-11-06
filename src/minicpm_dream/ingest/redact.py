"""Secret redaction and prompt-injection screening (deterministic, pre-model)."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S)),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("github_token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("aws_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("google_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    (
        "credential_assignment",
        re.compile(
            r"(?i)\b(api[_-]?key|secret|token|password|passwd|client[_-]?secret)\b\s*[:=]\s*"
            r"([\"']?)[^\s\"',;]{8,}\2"
        ),
    ),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    (
        "phone",
        re.compile(r"(?i)\b(?:phone|mobile|cell|tel|whatsapp)\b\s*[:=]?\s*(\+?\d[\d\s().-]{6,}\d)"),
    ),
    ("phone_intl", re.compile(r"\+\d{1,3}[\s.-]?(?:\(?\d{1,4}\)?[\s.-]?){2,5}\d{2,4}")),
]

_INJECTION: list[tuple[str, re.Pattern[str]]] = [
    ("ignore_instructions", re.compile(r"(?i)\b(ignore|disregard|forget)\b[^.\n]{0,40}\b(previous|prior|above|system)\b")),
    ("system_prompt", re.compile(r"(?i)\b(system prompt|developer message|hidden instructions)\b")),
    ("role_hijack", re.compile(r"(?i)\b(you are now|act as|pretend to be)\b[^.\n]{0,40}\b(admin|root|developer|system)\b")),
    ("chat_template", re.compile(r"<\|(?:im_start|im_end|system|assistant|user)\|>")),
    ("tool_call_spoof", re.compile(r"(?i)<\s*(?:tool_call|function_call)\b")),
]

_INVISIBLE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]")


@dataclass
class RedactionResult:
    text: str
    hits: Counter = field(default_factory=Counter)

    @property
    def clean(self) -> bool:
        return not self.hits


def redact(text: str) -> RedactionResult:
    hits: Counter = Counter()
    out = text
    for name, pattern in _PATTERNS:
        def repl(match: re.Match[str], _name: str = name) -> str:
            hits[_name] += 1
            return f"[REDACTED:{_name}]"

        out = pattern.sub(repl, out)
    return RedactionResult(text=out, hits=hits)


def injection_flags(text: str) -> list[str]:
    flags = [name for name, pattern in _INJECTION if pattern.search(text)]
    if _INVISIBLE.search(text):
        flags.append("invisible_unicode")
    return flags


def strip_invisible(text: str) -> str:
    return _INVISIBLE.sub("", text)


def quarantine_reason(text: str) -> str | None:
    flags = injection_flags(text)
    if flags:
        return ",".join(flags)
    return None


def has_secret(text: str) -> bool:
    return bool(redact(text).hits)


def scrubbed_for_prompt(text: str) -> tuple[str, bool]:
    """Return prompt-safe text and whether anything was removed."""
    result = redact(text)
    cleaned = strip_invisible(result.text)
    return cleaned, bool(result.hits) or cleaned != result.text


def safe_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)
