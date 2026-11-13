import logging
import math
import re
import dateparser

from app.diagnostics import log_failure


def clean_timestamp(raw_ts: str) -> str:
    """Strip forum noise suffixes while preserving date/year/time commas."""
    if not isinstance(raw_ts, str):
        return ""
    clean_ts = re.sub(r',\s+in\b.*$', '', raw_ts, flags=re.I)  # "4 hours ago, in Thread" -> "4 hours ago"
    clean_ts = re.sub(r'\binc/.*$', '', clean_ts, flags=re.I)
    clean_ts = re.sub(r'(?:Joined|Posts|Comments)\b.*$', '', clean_ts, flags=re.I)
    return clean_ts.strip(" ,")


def parse_timestamp(clean_ts: str):
    """Parse an English timestamp string deterministically in UTC."""
    if not isinstance(clean_ts, str) or not clean_ts.strip():
        return None
    dt = dateparser.parse(
        clean_ts,
        languages=['en'],
        settings={
            'TIMEZONE': 'UTC',
            'TO_TIMEZONE': 'UTC',
            'RETURN_AS_TIMEZONE_AWARE': True,
        },
    )
    return int(dt.timestamp()) if dt else None


def is_valid_raw_timestamp(raw_ts) -> bool:
    """Check whether raw_ts is a plausible non-empty timestamp string or numeric epoch."""
    if isinstance(raw_ts, bool):
        return False
    if isinstance(raw_ts, (int, float)):
        return math.isfinite(raw_ts) and raw_ts >= 0
    return isinstance(raw_ts, str) and bool(raw_ts.strip()) and raw_ts.strip() != "N/A"


def normalize_timestamps_with_stats(posts):
    """Normalize timestamps to deterministic UTC epochs and return (posts, stats)."""
    normalized_posts = [dict(p) for p in posts]
    stats = {"total": len(normalized_posts), "attempted": 0, "parsed": 0, "failed": 0}
    sample_failures = []

    for idx, post in enumerate(normalized_posts):
        raw_ts = post.get("timestamp", "")
        if not is_valid_raw_timestamp(raw_ts):
            continue
        if isinstance(raw_ts, (int, float)):
            post["timestamp"] = int(raw_ts)
            stats["attempted"] += 1
            stats["parsed"] += 1
            continue
        if isinstance(raw_ts, str):
            stats["attempted"] += 1
            cleaned = clean_timestamp(raw_ts)
            parsed = parse_timestamp(cleaned)
            if parsed is not None:
                post["timestamp"] = parsed
                stats["parsed"] += 1
            else:
                stats["failed"] += 1
                if len(sample_failures) < 5:
                    sample_failures.append(raw_ts)
                log_failure(
                    "Timestamp parse failed",
                    level=logging.DEBUG,
                    logger_name="date_utils",
                    index=idx,
                    raw=repr(raw_ts),
                    cleaned=repr(cleaned),
                )

    if stats["failed"] > 0:
        log_failure(
            f"normalize_timestamps: {stats['failed']}/{stats['attempted']} timestamp(s) failed parsing",
            level=logging.WARNING,
            logger_name="date_utils",
            total_posts=stats["total"],
            samples=sample_failures,
        )
    return normalized_posts, stats


def normalize_timestamps(posts):
    """
    Convert raw timestamp strings (or numeric epochs) in a list of posts
    into deterministic integer UTC epoch timestamps.
    If parsing fails, logs failure diagnostics and keeps the original value.
    """
    normalized_posts, _ = normalize_timestamps_with_stats(posts)
    return normalized_posts
