"""Время: ISO-8601 со смещением, дата курса, сторона границы периода."""
from datetime import datetime, timedelta, timezone


def parse_ts(text: str) -> datetime:
    """Разбор ISO-8601. Время без смещения запрещено."""
    ts = datetime.fromisoformat(text)
    if ts.tzinfo is None:
        raise ValueError("timestamp without offset")
    return ts


def fmt_ts(ts: datetime) -> str:
    return ts.isoformat()


def tz_from_offset(text: str) -> timezone:
    """'+05:00' -> timezone. Фиксированные смещения: не нужна база tzdata (на Windows её нет)."""
    sign = -1 if text.startswith("-") else 1
    hours, minutes = text.lstrip("+-").split(":")
    return timezone(sign * timedelta(hours=int(hours), minutes=int(minutes)))


def offset_text(tz: timezone) -> str:
    seconds = int(tz.utcoffset(None).total_seconds())
    sign = "-" if seconds < 0 else "+"
    seconds = abs(seconds)
    return f"{sign}{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}"


def fx_date(ts: datetime, rule: str, company_tz: timezone) -> str:
    """Дата курса по правилу политики: 'company_date' или 'utc_date'."""
    if rule == "company_date":
        return ts.astimezone(company_tz).date().isoformat()
    if rule == "utc_date":
        return ts.astimezone(timezone.utc).date().isoformat()
    raise ValueError(f"unknown fx rule: {rule}")


def period_side(ts: datetime, start: datetime, end: datetime) -> int:
    """-1 раньше периода, 0 внутри [start, end), +1 позже. Сравнение по мгновению, не по строке."""
    if ts < start:
        return -1
    if ts >= end:
        return 1
    return 0
