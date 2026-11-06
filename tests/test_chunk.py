from __future__ import annotations

from minicpm_dream.config import ExtractCfg, IngestCfg
from minicpm_dream.prepare.chunk import build_chunks, estimate_tokens
from tests.conftest import TZ, make_ep


def test_single_chunk_keeps_all_episodes():
    episodes = [make_ep("I prefer dark mode."), make_ep("The project uses SQLite.")]
    chunks = build_chunks(episodes, extract=ExtractCfg(), ingest=IngestCfg(), tz=TZ)
    assert len(chunks) == 1
    assert len(chunks[0].episodes) == 2
    assert all(ep.id in chunks[0].text for ep in episodes)


def test_chunks_respect_budget_and_overlap():
    episodes = [make_ep("word " * 800 + str(i)) for i in range(4)]
    cfg = ExtractCfg(max_chunk_tokens=700, overlap_ratio=0.1)
    chunks = build_chunks(episodes, extract=cfg, ingest=IngestCfg(), tz=TZ)
    assert len(chunks) > 1
    seen = {ep.id for chunk in chunks for ep in chunk.episodes}
    assert seen == {ep.id for ep in episodes}
    assert all(chunk.episodes for chunk in chunks)


def test_date_refs_are_attached_to_chunk():
    chunks = build_chunks(
        [make_ep("Remind me I have the dentist tomorrow at 3pm.")],
        extract=ExtractCfg(),
        ingest=IngestCfg(),
        tz=TZ,
    )
    refs = chunks[0].date_refs
    assert refs and refs[0].iso == "2026-11-06"
    assert "d0=2026-11-06" in chunks[0].text or refs[0].id == "d0"


def test_estimate_tokens_is_positive():
    assert estimate_tokens("hello world") >= 1
