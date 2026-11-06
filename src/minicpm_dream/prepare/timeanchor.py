"""Deterministic relative-date resolution.

The SLM never does date arithmetic. Code resolves expressions against the
episode's own timestamp/timezone and offers the model a small enum of refs.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

_MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}
_WEEKDAYS = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))
_WEEKDAY_ALT = "|".join(sorted(_WEEKDAYS, key=len, reverse=True))

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("iso", re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")),
    ("today", re.compile(r"\btoday\b", re.I)),
    ("yesterday", re.compile(r"\byesterday\b", re.I)),
    ("tomorrow", re.compile(r"\btomorrow\b", re.I)),
    ("last_night", re.compile(r"\blast night\b", re.I)),
    ("tonight", re.compile(r"\b(?:tonight|this evening)\b", re.I)),
    ("ago", re.compile(r"\b(\d{1,3})\s+(day|week|month|year)s?\s+ago\b", re.I)),
    ("last_weekday", re.compile(rf"\blast\s+({_WEEKDAY_ALT})\b", re.I)),
    ("next_weekday", re.compile(rf"\bnext\s+({_WEEKDAY_ALT})\b", re.I)),
    ("on_weekday", re.compile(rf"\bon\s+({_WEEKDAY_ALT})\b", re.I)),
    ("in_month", re.compile(rf"\bin\s+({_MONTH_ALT})\b", re.I)),
    ("month_day", re.compile(rf"\b({_MONTH_ALT})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", re.I)),
    ("day_month", re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH_ALT})\b", re.I)),
    ("next_week", re.compile(r"\bnext week\b", re.I)),
    ("next_month", re.compile(r"\bnext month\b", re.I)),
    ("next_year", re.compile(r"\bnext year\b", re.I)),
    ("this_week", re.compile(r"\bthis week\b", re.I)),
    ("this_month", re.compile(r"\bthis month\b", re.I)),
    ("this_year", re.compile(r"\bthis year\b", re.I)),
    ("last_week", re.compile(r"\blast week\b", re.I)),
    ("last_month", re.compile(r"\blast month\b", re.I)),
    ("last_year", re.compile(r"\blast year\b", re.I)),
]

_RELATIVE_WORDS = re.compile(
    r"(?i)\b(today|yesterday|tomorrow|tonight|last night|last week|last month|last year|"
    r"this week|this month|this year|next week|next month|next year|\d+\s+(?:day|week|month|year)s?\s+ago|"
    rf"last\s+({_WEEKDAY_ALT})|next\s+({_WEEKDAY_ALT})|on\s+({_WEEKDAY_ALT})|in\s+({_MONTH_ALT}))\b"
)


@dataclass(frozen=True)
class DateRef:
    id: str
    phrase: str
    iso: str
    source: str = "relative"

    def render(self) -> str:
        return f"{self.id}={self.iso} ({self.source}: {self.phrase!r})"


def _add_months(d: date, months: int) -> date:
    total = d.month - 1 + months
    year = d.year + total // 12
    month = total % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _weekday_on_or_before(d: date, weekday: int) -> date:
    delta = (d.weekday() - weekday) % 7
    return d - timedelta(days=delta)


def _weekday_after(d: date, weekday: int) -> date:
    delta = (weekday - d.weekday()) % 7
    return d + timedelta(days=delta or 7)


def _next_month_day(base: date, month: int, day: int) -> date:
    day = min(day, calendar.monthrange(base.year, month)[1])
    candidate = date(base.year, month, day)
    if candidate < base - timedelta(days=1):
        day = min(day, calendar.monthrange(base.year + 1, month)[1])
        candidate = date(base.year + 1, month, day)
    return candidate


def resolve_dates(text: str, spoken_at: datetime, tz: str, *, limit: int = 5) -> list[DateRef]:
    local = spoken_at.astimezone(ZoneInfo(tz))
    base = local.date()
    found: list[tuple[str, date]] = []

    for kind, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            phrase = match.group(0)
            value: date | None = None
            if kind == "iso":
                try:
                    value = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
                except ValueError:
                    value = None
            elif kind == "today":
                value = base
            elif kind in {"yesterday", "last_night"}:
                value = base - timedelta(days=1)
            elif kind == "tomorrow":
                value = base + timedelta(days=1)
            elif kind == "tonight":
                value = base
            elif kind == "ago":
                n = int(match.group(1))
                unit = match.group(2).lower()
                if unit == "day":
                    value = base - timedelta(days=n)
                elif unit == "week":
                    value = base - timedelta(weeks=n)
                elif unit == "month":
                    value = _add_months(base, -n)
                elif unit == "year":
                    value = _add_months(base, -12 * n)
            elif kind == "last_weekday":
                value = _weekday_on_or_before(base - timedelta(days=1), _WEEKDAYS[match.group(1).lower()])
            elif kind == "next_weekday":
                value = _weekday_after(base, _WEEKDAYS[match.group(1).lower()])
            elif kind == "on_weekday":
                target = _WEEKDAYS[match.group(1).lower()]
                value = base - timedelta(days=(base.weekday() - target) % 7)
            elif kind == "in_month":
                month = _MONTHS[match.group(1).lower()]
                value = _next_month_day(base, month, 1)
            elif kind == "month_day":
                month = _MONTHS[match.group(1).lower()]
                value = _next_month_day(base, month, int(match.group(2)))
            elif kind == "day_month":
                month = _MONTHS[match.group(2).lower()]
                value = _next_month_day(base, month, int(match.group(1)))
            elif kind == "next_week":
                value = base + timedelta(weeks=1)
            elif kind == "next_month":
                value = _add_months(base, 1)
            elif kind == "next_year":
                value = _add_months(base, 12)
            elif kind == "this_week":
                value = _weekday_on_or_before(base, 0)
            elif kind == "this_month":
                value = base.replace(day=1)
            elif kind == "this_year":
                value = base.replace(month=1, day=1)
            elif kind == "last_week":
                value = base - timedelta(weeks=1)
            elif kind == "last_month":
                value = _add_months(base, -1)
            elif kind == "last_year":
                value = _add_months(base, -12)
            if value is not None:
                found.append((phrase, value))

    refs: list[DateRef] = []
    seen: set[tuple[str, str]] = set()
    for phrase, value in found:
        key = (value.isoformat(), phrase.casefold())
        if key in seen:
            continue
        seen.add(key)
        refs.append(DateRef(id=f"d{len(refs)}", phrase=phrase, iso=value.isoformat()))
        if len(refs) >= limit:
            break
    return refs


def has_relative_time(text: str) -> bool:
    return bool(_RELATIVE_WORDS.search(text))


def render_refs(refs: list[DateRef]) -> str:
    if not refs:
        return "(none)"
    return "\n".join(ref.render() for ref in refs)
