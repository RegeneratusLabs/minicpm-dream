"""Load versioned JSON schemas and prompts from package data."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

_PKG = Path(__file__).resolve().parent.parent
_SCHEMA_DIR = _PKG / "schemas"
_PROMPT_DIR = _PKG / "prompts"


@lru_cache(maxsize=8)
def load_schema(name: str) -> dict:
    return json.loads((_SCHEMA_DIR / name).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_PROMPT_DIR)),
        undefined=StrictUndefined,
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


def render_prompt(name: str, **context: object) -> str:
    return _env().get_template(name).render(**context)


def prompt_version(name: str) -> str:
    return f"{name}@1"
