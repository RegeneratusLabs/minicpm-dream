from __future__ import annotations

from minicpm_dream.backends.mock import MockBackend
from minicpm_dream.config import ExtractCfg
from minicpm_dream.extract.extract import extract_chunk
from minicpm_dream.extract.schema import load_schema
from minicpm_dream.extract.verify import verify_candidate
from tests.conftest import make_chunk, make_ep


def _item(text: str, quote: str, **overrides):
    item = {
        "kind": "event",
        "section": "open_loops",
        "text": text,
        "subject": "dentist",
        "entities": [],
        "date_ref": "d0",
        "valid_to_ref": "none",
        "confidence": 0.9,
        "evidence": [{"episode_id": "ep_unknown", "quote": quote}],
    }
    item.update(overrides)
    return item


def test_extract_resolves_relative_date():
    ep = make_ep("Remind me I have the dentist tomorrow at 3pm.")
    chunk = make_chunk(ep)
    backend = MockBackend({"extract": [{"items": [_item("Josh has a dentist appointment.", "dentist tomorrow at 3pm")]}]})
    result = extract_chunk(backend, chunk, ExtractCfg())
    assert not result.rejected
    assert len(result.candidates) == 1
    cand = result.candidates[0]
    assert cand.valid_from == "2026-11-06"
    assert "2026-11-06" in cand.text
    assert cand.evidence[0].verified


def test_extract_rejects_non_verbatim_quote():
    ep = make_ep("I prefer dark mode in my editors.")
    chunk = make_chunk(ep)
    backend = MockBackend({"extract": [{"items": [_item("Josh likes pizza.", "I love pizza", date_ref="none")]}]})
    result = extract_chunk(backend, chunk, ExtractCfg())
    assert not result.candidates
    assert "verbatim" in result.rejected[0]["reason"]


def test_extract_rejects_meta_state():
    ep = make_ep("The user is asking about dinner.")
    chunk = make_chunk(ep)
    backend = MockBackend(
        {
            "extract": [
                {"items": [_item("The user is asking about dinner.", "The user is asking about dinner", date_ref="none")]}
            ]
        }
    )
    result = extract_chunk(backend, chunk, ExtractCfg())
    assert not result.candidates
    assert "meta" in result.rejected[0]["reason"]


def test_extract_rejects_unresolvable_relative_time():
    ep = make_ep("I will do it tomorrow.")
    chunk = make_chunk(ep)
    chunk.date_refs = []  # simulate a resolver that could not anchor the phrase
    backend = MockBackend(
        {"extract": [{"items": [_item("Josh will do it tomorrow.", "I will do it tomorrow", date_ref="none")]}]}
    )
    result = extract_chunk(backend, chunk, ExtractCfg())
    assert not result.candidates
    assert "relative time" in result.rejected[0]["reason"]


def test_schema_is_loadable():
    schema = load_schema("extract.json")
    assert "items" in schema["properties"]


def test_verify_uses_the_verifier():
    ep = make_ep("I prefer dark mode.")
    chunk = make_chunk(ep)
    extract_backend = MockBackend(
        {
            "extract": [
                {
                    "items": [
                        _item(
                            "Josh prefers dark mode.",
                            "I prefer dark mode",
                            kind="preference",
                            section="preferences",
                            date_ref="none",
                        )
                    ]
                }
            ]
        }
    )
    cand = extract_chunk(extract_backend, chunk, ExtractCfg()).candidates[0]
    verifier = MockBackend({"verify": [{"verdict": "contradicts", "confidence": 0.95}]})
    result = verify_candidate(verifier, cand, chunk, ExtractCfg())
    assert not result.supported
    assert result.verdict == "contradicts"


def test_meta_summary_is_rejected():
    ep = make_ep("The user is asking about dinner plans.")
    chunk = make_chunk(ep)
    backend = MockBackend(
        {"extract": [{"items": [_item("The user mentioned dinner plans but no details were provided.",
                                     "The user is asking about dinner plans.", date_ref="none")]}]}
    )
    result = extract_chunk(backend, chunk, ExtractCfg())
    assert not result.candidates
    assert "meta" in result.rejected[0]["reason"]


def test_reminder_cue_forces_task_and_open_loops():
    ep = make_ep("Remind me the dentist is tomorrow at 3pm.")
    chunk = make_chunk(ep)
    backend = MockBackend(
        {"extract": [{"items": [_item("The user has a dentist appointment tomorrow at 3pm.",
                                     "dentist is tomorrow at 3pm",
                                     kind="preference", section="preferences")]}]}
    )
    result = extract_chunk(backend, chunk, ExtractCfg())
    assert result.candidates
    cand = result.candidates[0]
    assert cand.kind == "task"
    assert cand.section == "open_loops"
    assert cand.valid_from == "2026-11-06"


def test_accepts_flat_string_evidence_from_cloud_models():
    ep = make_ep("I prefer dark mode in all my editors.")
    chunk = make_chunk(ep)
    backend = MockBackend(
        {
            "extract": [
                {
                    "items": [
                        {
                            "kind": "preference",
                            "section": "preferences",
                            "text": "Josh prefers dark mode in all editors.",
                            "date_ref": "none",
                            "confidence": 0.9,
                            "episode_id": ep.id,
                            "evidence": ["I prefer dark mode in all my editors"],
                        }
                    ]
                }
            ]
        }
    )
    result = extract_chunk(backend, chunk, ExtractCfg())
    assert len(result.candidates) == 1
    assert result.candidates[0].evidence[0].episode_id == ep.id
