"""Ledger storage."""

from .db import SCHEMA_VERSION, Store

__all__ = ["Store", "SCHEMA_VERSION"]
