"""Shared crawler utilities — deduplicated helpers used across crawler and parser modules."""

import hashlib
import json
import logging

from app.diagnostics import log_failure

MIN_POST_WORDS = 5


def is_substantial_post(text, min_words=MIN_POST_WORDS):
    """Return True if text contains at least min_words whitespace-separated tokens."""
    return bool(text and len(str(text).split()) >= min_words)


def hash_text_snippet(text, prefix="", length=200):
    """Compute MD5 digest over the leading snippet of text with an optional prefix."""
    snippet = str(text or "")[:length]
    payload = f"{prefix}{snippet}" if prefix else snippet
    return hashlib.md5(payload.encode()).hexdigest()


def content_hash(post):
    """Create a hash from post content + username for deduplication."""
    username = post.get("username", "")
    return hash_text_snippet(post.get("post", ""), prefix=f"{username}|", length=200)


def sanitize_filename(name):
    """Convert a username or arbitrary string into a safe filename."""
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in name)


def filter_substantial_posts(rows):
    """Filter posts to only those with MIN_POST_WORDS+ words of content."""
    return [r["post"] for r in rows if is_substantial_post(r.get("post"))]


def append_to_json_file(data, filepath):
    """Append data to a JSON file (read-extend-write pattern)."""
    existing = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            existing = json.load(f)
    except FileNotFoundError:
        pass
    except (json.JSONDecodeError, OSError) as exc:
        log_failure(f"Failed reading existing JSON file '{filepath}'", exc, level=logging.WARNING)

    existing.extend(data)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, default=str, ensure_ascii=False)

    return existing


def persist_crawl_results(results, json_filepath="crawl_data.json", normalize_fn=None, save_fn=None):
    """Normalize timestamps, save posts to SQLite, and append to JSON.

    Returns tuple (normalized_results, all_json_records).
    """
    if normalize_fn is None:
        from app.parser.date_utils import normalize_timestamps as normalize_fn
    if save_fn is None:
        from app.database import save_posts as save_fn

    normalized = normalize_fn(results)
    save_fn(normalized)
    all_records = append_to_json_file(normalized, json_filepath)
    return normalized, all_records
