from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from minicpm_dream.config import Config, ExtractCfg, IngestCfg, load_config
from minicpm_dream.extract.validate import Candidate, Evidence
from minicpm_dream.ingest.base import make_episode
from minicpm_dream.prepare.chunk import Chunk, build_chunks

TZ = "Australia/Brisbane"


@pytest.fixture
def cfg(tmp_path) -> Config:
    config = load_config(tmp_path / "missing.toml", data_dir=str(tmp_path / "data"))
    # Fixtures use Brisbane timestamps; pin the zone so tests do not depend on
    # the machine's /etc/timezone (CI runs in UTC).
    config.timezone = TZ
    config.hermes.home = str(tmp_path / "hermes")
    config.resolve()
    (tmp_path / "hermes" / "memories").mkdir(parents=True, exist_ok=True)
    (tmp_path / "hermes" / "memories" / "MEMORY.md").write_text("", encoding="utf-8")
    (tmp_path / "hermes" / "memories" / "USER.md").write_text("", encoding="utf-8")
    return config


def make_ep(text: str, *, role: str = "user", source: str = "desktop", session: str = "s1",
            ts: datetime | None = None):
    ts = ts or datetime(2026, 11, 5, 10, 0, tzinfo=ZoneInfo(TZ))
    return make_episode(
        source=source, session_id=session, role=role, timestamp=ts, text=text, tz=TZ,
        message_id=f"{session}:{role}:{text[:10]}",
    )


def make_chunk(*episodes) -> Chunk:
    return build_chunks(
        list(episodes), extract=ExtractCfg(), ingest=IngestCfg(), tz=TZ
    )[0]


def make_candidate(text: str, *, kind: str = "fact", section: str = "projects",
                   episode=None, confidence: float = 0.9) -> Candidate:
    ev = Evidence(episode.id if episode else "ep_x", "quote text here", verified=True)
    return Candidate(
        kind=kind,
        section=section,
        text=text,
        subject="",
        entities=[],
        date_ref="none",
        valid_to_ref="none",
        confidence=confidence,
        evidence=[ev],
        chunk_index=0,
    )
