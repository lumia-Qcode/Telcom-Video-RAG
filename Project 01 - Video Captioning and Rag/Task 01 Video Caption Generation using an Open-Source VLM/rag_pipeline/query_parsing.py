"""
Lightweight natural-language query understanding.

Pulls out structural filters (date range, time-of-day range, label
mentions, verdict/classification keywords) from a free-text query so the
retriever can narrow the candidate set before semantic/BM25 ranking runs.
This is heuristic, not an LLM call - it keeps filtering fast and free,
and complex/ambiguous language is still handled by the LLM at answer time
since we pass the raw query through untouched as well.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import config

try:
    from dateutil import parser as dateparser
except ImportError:
    dateparser = None


@dataclass
class QueryFilters:
    date_start: Optional[datetime] = None
    date_end: Optional[datetime] = None
    hour_start: Optional[int] = None   # 0-23, inclusive
    hour_end: Optional[int] = None     # 0-23, inclusive
    labels: List[str] = field(default_factory=list)
    verdict_keywords: List[str] = field(default_factory=list)  # e.g. "false positive", "hallucination"


WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

VERDICT_TERMS = {
    "hallucinat": "False Positive",
    "false positive": "False Positive",
    "false negative": "False Negative",
    "true positive": "True Positive",
    "missed": "False Negative",
    "all matched": "All Matched",
    "unclassified": "Unclassified",
    "no ground truth": "Unclassified",
}


def _day_bounds(d: datetime) -> Tuple[datetime, datetime]:
    start = d.replace(hour=0, minute=0, second=0, microsecond=0)
    return start, start + timedelta(days=1) - timedelta(seconds=1)


def _parse_relative_dates(q: str, today: datetime) -> Optional[Tuple[datetime, datetime]]:
    ql = q.lower()

    if "today" in ql:
        return _day_bounds(today)
    if "yesterday" in ql:
        return _day_bounds(today - timedelta(days=1))
    if "this week" in ql:
        start = today - timedelta(days=today.weekday())
        return _day_bounds(start)[0], today
    if "last week" in ql:
        this_monday = today - timedelta(days=today.weekday())
        last_monday = this_monday - timedelta(days=7)
        last_sunday = this_monday - timedelta(seconds=1)
        return _day_bounds(last_monday)[0], last_sunday
    if "last month" in ql:
        start = today - timedelta(days=30)
        return start, today
    if "this month" in ql:
        start = today.replace(day=1)
        return start, today

    m = re.search(r"last (\d+) days?", ql)
    if m:
        n = int(m.group(1))
        return today - timedelta(days=n), today

    m = re.search(r"past (\d+) days?", ql)
    if m:
        n = int(m.group(1))
        return today - timedelta(days=n), today

    for i, day_name in enumerate(WEEKDAYS):
        if day_name in ql:
            # most recent occurrence of that weekday, on/before today
            delta = (today.weekday() - i) % 7
            target = today - timedelta(days=delta)
            return _day_bounds(target)

    return None


def _parse_explicit_date(q: str) -> Optional[Tuple[datetime, datetime]]:
    if dateparser is None:
        return None
    # look for common explicit date patterns e.g. "July 7", "2026-07-07", "7/7"
    m = re.search(
        r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}(?:/\d{2,4})?|"
        r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?)\b",
        q,
        re.IGNORECASE,
    )
    if not m:
        return None
    try:
        dt = dateparser.parse(m.group(0), default=config.TODAY)
        return _day_bounds(dt)
    except (ValueError, OverflowError):
        return None


_TIME_WORD_RANGES = {
    "morning": (6, 11),
    "afternoon": (12, 16),
    "evening": (17, 20),
    "night": (20, 23),
    "noon": (12, 12),
}


def _parse_time_range(q: str) -> Tuple[Optional[int], Optional[int]]:
    ql = q.lower()

    m = re.search(r"between\s+(\d{1,2})\s*(am|pm)?\s*(?:and|-|to)\s*(\d{1,2})\s*(am|pm)?", ql)
    if m:
        h1, ap1, h2, ap2 = m.groups()
        h1, h2 = int(h1), int(h2)
        # "between 4 and 5 PM" - a trailing am/pm with no leading one applies to both
        if not ap1 and ap2:
            ap1 = ap2
        ap2 = ap2 or ap1  # if only leading has am/pm, share it with the end too
        if ap1 == "pm" and h1 != 12:
            h1 += 12
        if ap2 == "pm" and h2 != 12:
            h2 += 12
        return h1, h2

    m = re.search(r"after\s+(\d{1,2})\s*(am|pm)?", ql)
    if m:
        h, ap = m.groups()
        h = int(h)
        if ap == "pm" and h != 12:
            h += 12
        return h, 23

    m = re.search(r"before\s+(\d{1,2})\s*(am|pm)?", ql)
    if m:
        h, ap = m.groups()
        h = int(h)
        if ap == "pm" and h != 12:
            h += 12
        return 0, h

    for word, (h1, h2) in _TIME_WORD_RANGES.items():
        if word in ql:
            return h1, h2

    return None, None


def _parse_labels(q: str) -> List[str]:
    ql = q.lower()
    found = []
    for canonical, synonyms in config.LABEL_SYNONYMS.items():
        for syn in synonyms:
            if syn in ql:
                found.append(canonical)
                break
    return found


def _parse_verdict_keywords(q: str) -> List[str]:
    ql = q.lower()
    found = []
    for phrase, canonical in VERDICT_TERMS.items():
        if phrase in ql and canonical not in found:
            found.append(canonical)
    return found


def parse_query(query: str, today: datetime = None) -> QueryFilters:
    today = today or config.TODAY

    date_range = _parse_relative_dates(query, today) or _parse_explicit_date(query)
    hour_start, hour_end = _parse_time_range(query)

    return QueryFilters(
        date_start=date_range[0] if date_range else None,
        date_end=date_range[1] if date_range else None,
        hour_start=hour_start,
        hour_end=hour_end,
        labels=_parse_labels(query),
        verdict_keywords=_parse_verdict_keywords(query),
    )


if __name__ == "__main__":
    tests = [
        "show me technicians working last week",
        "any videos with solar panels between 4 and 5 PM?",
        "what happened yesterday morning",
        "false positives involving utility towers",
        "videos from July 7",
        "cell tower footage after 3pm",
    ]
    for t in tests:
        print(t, "->", parse_query(t))
