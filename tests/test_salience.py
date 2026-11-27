from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from minicpm_dream.config import ExtractCfg, IngestCfg
from minicpm_dream.ingest.base import make_episode
from minicpm_dream.prepare.chunk import build_chunks
from minicpm_dream.prepare.salience import chunk_score, gate_chunks, salience_score

TZ = "Australia/Brisbane"


def _chunk(text: str, role: str = "user"):
    ep = make_episode(
        source="desktop",
        session_id="s1",
        role=role,
        timestamp=datetime(2026, 11, 5, 10, 0, tzinfo=ZoneInfo(TZ)),
        text=text,
        tz=TZ,
        message_id=text[:12],
    )
    return build_chunks([ep], extract=ExtractCfg(), ingest=IngestCfg(), tz=TZ)[0]


def test_preference_scores_above_a_health_check_log():
    preference = salience_score("I prefer dark mode in all my editors, it's easier on my eyes.")
    assert preference > 0.12
    log = "\n".join(
        f"GET /api/v1/health 200 12ms 200 - {index}" for index in range(40)
    )
    assert salience_score(log) < 0.12
    assert salience_score("") == 0.0


def test_gate_skips_only_the_noise_chunk():
    good = _chunk("I decided to move the ledger to PostgreSQL next week.")
    noise = _chunk("\n".join(f"GET /api/v1/health 200 12ms {i}" for i in range(60)))
    kept, stats = gate_chunks([good, noise], threshold=0.12)
    assert [chunk.index for chunk in kept] == [good.index]
    assert stats.skipped == 1
    assert stats.skipped_tokens > 0
    assert chunk_score(good) > chunk_score(noise)


def test_gate_always_keeps_at_least_one_chunk():
    noise = _chunk("\n".join(f"GET /api/v1/health 200 12ms {i}" for i in range(60)))
    kept, stats = gate_chunks([noise], threshold=0.9, keep_min=1)
    assert kept == [noise]
    assert stats.skipped == 0
