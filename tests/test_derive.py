from __future__ import annotations

from minicpm_dream.extract.derive import derive_candidates
from tests.conftest import make_chunk, make_ep


def test_derives_relationship():
    ep = make_ep("My brother Cameron is visiting next week.")
    candidates = derive_candidates(make_chunk(ep))
    assert any("Cameron is the user's brother" in c.text for c in candidates)
    assert all(c.origin == "derived" for c in candidates)


def test_derives_migration_fact():
    ep = make_ep("We moved the miniCPM-memory ledger from SQLite to PostgreSQL on 2026-11-04.")
    candidates = derive_candidates(make_chunk(ep))
    migration = [c for c in candidates if "PostgreSQL" in c.text]
    assert migration
    assert migration[0].valid_from == "2026-11-04"
    assert migration[0].evidence[0].quote in ep.text
