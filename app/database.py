from contextlib import closing
import os
import sqlite3

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "Forum_profiles.db")


def _connect():
    """Open and return a new SQLite connection."""
    return sqlite3.connect(DB_PATH)


def init_db():
    with closing(_connect()) as conn:
        c = conn.cursor()

        c.execute("""
        CREATE TABLE IF NOT EXISTS posts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            forum       TEXT NOT NULL,
            username    TEXT NOT NULL,
            content     TEXT,
            timestamp   TEXT,
            url         TEXT,
            crawled_at  TEXT DEFAULT CURRENT_TIMESTAMP 
        )""")

        c.execute("""
        CREATE TABLE IF NOT EXISTS identity_pairs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username_a  TEXT,
            forum_a     TEXT,
            username_b  TEXT,
            forum_b     TEXT,
            same_person INTEGER,  -- 1=yes, 0=no, NULL=unknown
            confidence  TEXT      -- 'manual', 'inferred', 'auto'
        )""")

        conn.commit()

    from Model.fingerprint_store import init_fingerprint_tables
    init_fingerprint_tables()


def get_user_posts(username, forum=None):
    with closing(_connect()) as conn:
        c = conn.cursor()

        query = "SELECT content, timestamp FROM posts WHERE username=?"
        params = [username]
        if forum:
            query += " AND forum=?"
            params.append(forum)

        c.execute(query, tuple(params))
        rows = c.fetchall()
    return [{"post": r[0], "timestamp": r[1]} for r in rows]


def get_all_users():
    with closing(_connect()) as conn:
        c = conn.cursor()
        c.execute("SELECT DISTINCT username, forum FROM posts WHERE username != 'unknown'")
        rows = c.fetchall()
    return rows


def _resolve_forum(url):
    """Derive the forum identifier from a URL or plain string."""
    from urllib.parse import urlparse
    if url.startswith("http"):
        return urlparse(url).netloc
    return url or "unknown"


def save_posts(data):
    """Save a list of post dictionaries to the database."""
    if not data:
        return

    with closing(_connect()) as conn:
        c = conn.cursor()

        for item in data:
            username = item.get("username", "unknown")
            content = item.get("post", "")
            timestamp = item.get("timestamp", "N/A")
            url = item.get("source", "")
            forum = _resolve_forum(url)

            c.execute("""
                INSERT INTO posts (forum, username, content, timestamp, url)
                VALUES (?, ?, ?, ?, ?)
            """, (forum, username, content, timestamp, url))

        conn.commit()


def get_crawled_urls():
    """Retrieve all URLs that have already been successfully parsed."""
    with closing(_connect()) as conn:
        c = conn.cursor()
        c.execute("SELECT DISTINCT url FROM posts WHERE url IS NOT NULL")
        rows = c.fetchall()
    return {r[0] for r in rows}
