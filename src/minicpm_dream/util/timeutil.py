"""Time helpers: UTC storage, local display, deterministic day windows."""

from __future__ import annotations

import calendar
from datetime import UTC, date, datetime, timedelta
from datetime import time as dtime
from zoneinfo import ZoneInfo


def utc_now() -> datetime:
    return datetime.now(UTC)


def parse_timestamp(value: object, default_tz: str = "UTC") -> datetime:
    """Parse epoch seconds/millis or an ISO-8601 string into an aware datetime."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=ZoneInfo(default_tz)).astimezone(UTC)
        return value.astimezone(UTC)
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds > 1e12:  # milliseconds
            seconds /= 1000.0
        return datetime.fromtimestamp(seconds, tz=UTC)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("empty timestamp")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(default_tz))
    return dt.astimezone(UTC)


def iso_utc(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat(timespec="seconds")


def iso_local(dt: datetime, tz: str) -> str:
    return dt.astimezone(ZoneInfo(tz)).isoformat(timespec="seconds")


def local_date(dt: datetime, tz: str) -> date:
    return dt.astimezone(ZoneInfo(tz)).date()


def day_window(target: date, tz: str) -> tuple[datetime, datetime]:
    """Return [start, end) UTC for a local calendar day."""
    zone = ZoneInfo(tz)
    start = datetime.combine(target, dtime.min, tzinfo=zone)
    end = datetime.combine(target + timedelta(days=1), dtime.min, tzinfo=zone)
    return start.astimezone(UTC), end.astimezone(UTC)


def previous_day(tz: str, now: datetime | None = None) -> date:
    now = now or utc_now()
    return local_date(now, tz) - timedelta(days=1)


def week_window(end_day: date, tz: str, days: int = 7) -> tuple[datetime, datetime]:
    start = end_day - timedelta(days=days - 1)
    start_dt, _ = day_window(start, tz)
    _, end_dt = day_window(end_day, tz)
    return start_dt, end_dt


def human_age(then: datetime, now: datetime | None = None) -> str:
    now = now or utc_now()
    seconds = max(0, int((now - then).total_seconds()))
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    days = seconds // 86400
    if days < 30:
        return f"{days}d ago"
    if days < 365:
        return f"{days // 30}mo ago"
    return f"{days // 365}y ago"


def month_bounds(year: int, month: int) -> tuple[date, date]:
    last = calendar.monthrange(year, month)[1]
    return date(year, month, 1), date(year, month, last)
