from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from minicpm_dream.prepare.timeanchor import has_relative_time, resolve_dates

TZ = "Australia/Brisbane"
BASE = datetime(2026, 11, 5, 10, 0, tzinfo=ZoneInfo(TZ))


def dates(text: str) -> dict[str, str]:
    return {ref.phrase.lower(): ref.iso for ref in resolve_dates(text, BASE, TZ)}


def test_relative_dates():
    assert dates("I did it yesterday")["yesterday"] == "2026-11-04"
    assert dates("3 days ago")["3 days ago"] == "2026-11-02"
    assert dates("Tomorrow at 3pm")["tomorrow"] == "2026-11-06"
    assert dates("last night was rough")["last night"] == "2026-11-04"


def test_weekday_dates():
    # 2026-11-05 is a Thursday
    assert dates("last Saturday")["last saturday"] == "2026-10-31"
    assert dates("next Monday")["next monday"] == "2026-11-09"
    assert dates("on Friday")["on friday"] == "2026-10-30"


def test_month_and_iso_dates():
    assert dates("in March")["in march"] == "2027-03-01"
    assert dates("November 20")["november 20"] == "2026-11-20"
    assert dates("2026-12-01")["2026-12-01"] == "2026-12-01"


def test_has_relative_time():
    assert has_relative_time("I will do it tomorrow")
    assert not has_relative_time("Josh prefers dark mode")
