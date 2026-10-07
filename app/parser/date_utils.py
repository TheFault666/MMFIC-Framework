import dateparser
import re

def clean_timestamp(raw_ts: str) -> str:
    clean_ts = re.sub(r',.*$', '', raw_ts) # "4 hours ago, in" -> "4 hours ago"
    clean_ts = re.sub(r'inc/.*$', '', clean_ts)
    clean_ts = re.sub(r'(?:Joined|Posts|Comments).*$', '', clean_ts, flags=re.I)
    return clean_ts.strip()

def parse_timestamp(clean_ts: str):
    dt = dateparser.parse(clean_ts, settings={'TIMEZONE': 'UTC', 'RETURN_AS_TIMEZONE_AWARE': True})
    return int(dt.timestamp()) if dt else None

def is_valid_raw_timestamp(raw_ts) -> bool:
    return bool(raw_ts and raw_ts != "N/A")

def normalize_timestamps(posts):
    """
    Convert raw timestamp strings in a list of posts into integer epoch timestamps.
    If parsing fails, the original string is kept.
    """
    normalized_posts = [dict(p) for p in posts]
    for post in normalized_posts:
        raw_ts = post.get("timestamp", "")
        if is_valid_raw_timestamp(raw_ts):
            cleaned = clean_timestamp(raw_ts)
            parsed = parse_timestamp(cleaned)
            if parsed is not None:
                post["timestamp"] = parsed
    return normalized_posts
