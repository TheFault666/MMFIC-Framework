"""
Shared crawler utilities — deduplicated helpers used across crawler modules.

Extracted during code pruning to eliminate identical implementations that
existed in crawler.py, user_crawler.py, main.py, and dashboard.py.
"""

import hashlib
import json
import os


def content_hash(post):
    """Create a hash from post content + username for deduplication."""
    key = f"{post.get('username', '')}|{post.get('post', '')[:200]}"
    return hashlib.md5(key.encode()).hexdigest()


def sanitize_filename(name):
    """Convert a username or arbitrary string into a safe filename."""
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in name)


def filter_substantial_posts(rows):
    """Filter posts to only those with 5+ words of content.
    
    Args:
        rows: list of dicts with a 'post' key
    
    Returns:
        list of post strings that have >= 5 words
    """
    return [r["post"] for r in rows if r.get("post") and len(r["post"].split()) >= 5]


def append_to_json_file(data, filepath):
    """Append data to a JSON file (read-extend-write pattern).
    
    Args:
        data: list of dicts to append
        filepath: path to the JSON file
    
    Returns:
        list: the full combined data
    """
    existing = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            existing = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    existing.extend(data)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, default=str, ensure_ascii=False)

    return existing
