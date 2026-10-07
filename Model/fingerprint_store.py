"""
Fingerprint Store — precomputed author fingerprints stored in SQLite.

This is the production-grade approach:
  1. When posts are crawled, compute a 138-dim fingerprint per user@forum
  2. Store the fingerprint in the database
  3. When new posts arrive for the same user, UPDATE the fingerprint
  4. For similarity checks, compare fingerprints using the trained model
  5. Returns ranked matches with confidence scores

This integrates with the existing database.py schema.

Recovery: If this approach doesn't work well, the original pairwise
comparison (Model.model.AuthorSimilarityModel.predict_similarity)
still works exactly as before — it doesn't depend on this module.
"""

import os
import sqlite3
import json
import numpy as np
from datetime import datetime
from collections import defaultdict
from urllib.parse import urlparse

from .features import extract_user_features, get_feature_count

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "Forum_profiles.db")


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_fingerprint_tables():
    """Create fingerprint tables. Call once at startup.
    
    Adds to the existing database without modifying existing tables.
    """
    conn = _get_conn()
    c = conn.cursor()
    
    c.execute("""
    CREATE TABLE IF NOT EXISTS fingerprints (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        username        TEXT NOT NULL,
        forum           TEXT NOT NULL,
        fingerprint     TEXT NOT NULL,
        post_count      INTEGER DEFAULT 0,
        total_words     INTEGER DEFAULT 0,
        first_seen      TEXT DEFAULT CURRENT_TIMESTAMP,
        last_updated    TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(username, forum)
    )""")
    
    c.execute("""
    CREATE TABLE IF NOT EXISTS training_log (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        trained_at      TEXT DEFAULT CURRENT_TIMESTAMP,
        n_users         INTEGER,
        n_posts_total   INTEGER,
        n_pairs         INTEGER,
        n_positive      INTEGER,
        n_negative      INTEGER,
        cv_roc_auc      REAL,
        cv_accuracy     REAL,
        train_roc_auc   REAL,
        model_version   TEXT
    )""")
    
    c.execute("""
    CREATE TABLE IF NOT EXISTS similarity_results (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        username_a      TEXT NOT NULL,
        forum_a         TEXT NOT NULL,
        username_b      TEXT NOT NULL,
        forum_b         TEXT NOT NULL,
        score           REAL NOT NULL,
        computed_at     TEXT DEFAULT CURRENT_TIMESTAMP,
        model_version   TEXT,
        UNIQUE(username_a, forum_a, username_b, forum_b)
    )""")
    
    conn.commit()
    conn.close()


# ── DB read queries ──────────────────────────────────────────────

def _query_fingerprint(username, forum):
    """Read one fingerprint row from the database.
    
    Returns the raw row tuple (fingerprint_json, post_count, total_words, last_updated),
    or None if not found.
    """
    conn = _get_conn()
    c = conn.cursor()
    c.execute("""
        SELECT fingerprint, post_count, total_words, last_updated
        FROM fingerprints WHERE username = ? AND forum = ?
    """, (username, forum))
    row = c.fetchone()
    conn.close()
    return row


def _query_all_fingerprints():
    """Read all fingerprint rows from the database.
    
    Returns a list of raw row tuples
    (username, forum, fingerprint_json, post_count, total_words, last_updated).
    """
    conn = _get_conn()
    c = conn.cursor()
    c.execute("""
        SELECT username, forum, fingerprint, post_count, total_words, last_updated
        FROM fingerprints ORDER BY last_updated DESC
    """)
    rows = c.fetchall()
    conn.close()
    return rows


def _query_distinct_users():
    """Read all distinct (username, forum) pairs from the posts table."""
    conn = _get_conn()
    c = conn.cursor()
    c.execute("""
        SELECT DISTINCT username, forum FROM posts
        WHERE username != 'unknown' AND content IS NOT NULL
    """)
    users = c.fetchall()
    conn.close()
    return users


def _query_posts_for_user(username, forum):
    """Read all post rows for a given user from the posts table.
    
    Returns a list of (content, timestamp, url) tuples.
    """
    conn = _get_conn()
    c = conn.cursor()
    c.execute("""
        SELECT content, timestamp, url FROM posts
        WHERE username = ? AND forum = ? AND content IS NOT NULL
    """, (username, forum))
    rows = c.fetchall()
    conn.close()
    return rows


# ── DB write commands ────────────────────────────────────────────

def _upsert_fingerprint(username, forum, fp_json, post_count, total_words):
    """Write or update a fingerprint row in the database."""
    conn = _get_conn()
    c = conn.cursor()
    
    c.execute("""
        INSERT INTO fingerprints (username, forum, fingerprint, post_count, total_words, last_updated)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(username, forum) DO UPDATE SET
            fingerprint = excluded.fingerprint,
            post_count = excluded.post_count,
            total_words = excluded.total_words,
            last_updated = excluded.last_updated
    """, (username, forum, fp_json, post_count, total_words, datetime.now().isoformat()))
    
    conn.commit()
    conn.close()


def _delete_fingerprint_row(username, forum):
    """Delete one fingerprint row. Returns True if a row was deleted."""
    conn = _get_conn()
    c = conn.cursor()
    c.execute("DELETE FROM fingerprints WHERE username = ? AND forum = ?", (username, forum))
    deleted = c.rowcount
    conn.commit()
    conn.close()
    return deleted > 0


def _insert_similarity_results(results, model_version):
    """Bulk-insert similarity result rows into the database."""
    conn = _get_conn()
    c = conn.cursor()
    
    for r in results:
        user_a, forum_a = r["user_a"].rsplit("@", 1)
        user_b, forum_b = r["user_b"].rsplit("@", 1)
        
        c.execute("""
            INSERT OR REPLACE INTO similarity_results
            (username_a, forum_a, username_b, forum_b, score, computed_at, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (user_a, forum_a, user_b, forum_b, r["score"],
              datetime.now().isoformat(), model_version))
    
    conn.commit()
    conn.close()


def _insert_training_log_row(n_users, n_posts_total, metrics, model_version):
    """Write one training log row to the database."""
    conn = _get_conn()
    c = conn.cursor()
    
    c.execute("""
        INSERT INTO training_log
        (n_users, n_posts_total, n_pairs, n_positive, n_negative,
         cv_roc_auc, cv_accuracy, train_roc_auc, model_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        n_users,
        n_posts_total,
        metrics.get("n_pairs", 0),
        metrics.get("n_positive", 0),
        metrics.get("n_negative", 0),
        metrics.get("cv_roc_auc_mean", 0),
        metrics.get("cv_accuracy_mean", 0),
        metrics.get("train_roc_auc", 0),
        model_version,
    ))
    
    conn.commit()
    conn.close()


# ── Fingerprint CRUD ─────────────────────────────────────────────

def _load_lda_components_from_model():
    """Load LDA model, vectorizer, and board_vocab from the saved AuthorSimilarityModel.
    
    Returns (lda_model, lda_vectorizer, board_vocab), each may be None on failure.
    """
    try:
        from .model import AuthorSimilarityModel
        model = AuthorSimilarityModel.load()
        return model.lda_model, model.lda_vectorizer, model.board_vocab
    except Exception:
        return None, None, None


def store_fingerprint(username, forum, posts, timestamps=None, sources=None, lda_model=None, lda_vectorizer=None, board_vocab=None):
    """Compute and store/update a user's fingerprint.
    
    If the user already exists, recomputes from ALL posts (old + new).
    
    Args:
        username: the forum username
        forum: forum domain or identifier
        posts: list of post strings
        timestamps: optional list of timestamp strings
        sources: optional list of source URLs
        lda_model: fitted LatentDirichletAllocation object
        lda_vectorizer: fitted CountVectorizer object
        board_vocab: list of board names
    
    Returns:
        dict with fingerprint metadata
    """
    if not posts:
        return None
    
    timestamps = timestamps or ["N/A"] * len(posts)
    sources = sources or ["manual"] * len(posts)
    
    if lda_model is None or lda_vectorizer is None or board_vocab is None:
        lda_model, lda_vectorizer, board_vocab = _load_lda_components_from_model()
    
    fingerprint = extract_user_features(posts, timestamps, sources, lda_model, lda_vectorizer, board_vocab)
    fingerprint = np.nan_to_num(fingerprint, nan=0.0, posinf=0.0, neginf=0.0)
    
    fp_json = json.dumps(fingerprint.tolist())
    total_words = sum(len(p.split()) for p in posts)
    
    _upsert_fingerprint(username, forum, fp_json, len(posts), total_words)
    
    meta = {
        "username": username,
        "forum": forum,
        "post_count": len(posts),
        "total_words": total_words,
        "fingerprint_dims": len(fingerprint),
    }
    
    return meta


def update_fingerprint(username, forum, lda_model=None, lda_vectorizer=None, board_vocab=None):
    """Recompute a user's fingerprint from all their posts in the DB.
    
    Reads posts from the existing 'posts' table, recomputes features,
    and updates the fingerprints table.
    """
    rows = _query_posts_for_user(username, forum)
    if not rows:
        return None
    
    posts = [r[0] for r in rows if r[0] and len(r[0].split()) >= 5]
    if not posts:
        return None

    timestamps = [r[1] or "N/A" for r in rows]
    sources = [r[2] or "manual" for r in rows]
    
    return store_fingerprint(username, forum, posts, timestamps, sources, lda_model, lda_vectorizer, board_vocab)


def get_fingerprint(username, forum):
    """Retrieve a stored fingerprint as a numpy array."""
    row = _query_fingerprint(username, forum)
    
    if not row:
        return None
    
    return {
        "fingerprint": np.array(json.loads(row[0])),
        "post_count": row[1],
        "total_words": row[2],
        "last_updated": row[3],
    }


def get_all_fingerprints():
    """Retrieve all stored fingerprints.
    
    Returns:
        list of dicts with username, forum, fingerprint (numpy), post_count
    """
    return [
        {
            "username": row[0],
            "forum": row[1],
            "fingerprint": np.array(json.loads(row[2])),
            "post_count": row[3],
            "total_words": row[4],
            "last_updated": row[5],
        }
        for row in _query_all_fingerprints()
    ]


def delete_fingerprint(username, forum):
    """Remove a fingerprint from the store."""
    return _delete_fingerprint_row(username, forum)


# ── Bulk operations ──────────────────────────────────────────────

def rebuild_all_fingerprints():
    """Recompute ALL fingerprints from the posts table.
    
    Use after retraining or if fingerprints seem stale.
    """
    users = _query_distinct_users()
    lda_model, lda_vectorizer, board_vocab = _load_lda_components_from_model()
    
    updated = 0
    for username, forum in users:
        result = update_fingerprint(username, forum, lda_model, lda_vectorizer, board_vocab)
        if result:
            updated += 1
    
    return updated


def _extract_domain_for_ingestion(source):
    if source.startswith("http"):
        return urlparse(source).netloc
    return source


def ingest_crawl_data(filepath="crawl_data.json"):
    """Load crawl data and store fingerprints for all users.
    
    This is the bridge between crawling and the fingerprint store.
    Groups posts by username+domain, then stores each fingerprint.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    profiles = defaultdict(lambda: {"posts": [], "timestamps": [], "sources": []})
    
    for entry in data:
        user = entry.get("username", "unknown")
        if user == "unknown":
            continue
        
        post = entry.get("post", "")
        if len(post.split()) < 5:
            continue
        
        source = entry.get("source", "manual")
        domain = _extract_domain_for_ingestion(source)
        
        key = f"{user}@{domain}"
        profiles[key]["posts"].append(post)
        profiles[key]["timestamps"].append(entry.get("timestamp", "N/A"))
        profiles[key]["sources"].append(source)
        profiles[key]["username"] = user
        profiles[key]["forum"] = domain
    
    lda_model, lda_vectorizer, board_vocab = _load_lda_components_from_model()
        
    stored = 0
    for key, data in profiles.items():
        if len(data["posts"]) >= 3:
            store_fingerprint(data["username"], data["forum"], data["posts"], data["timestamps"], data.get("sources"), lda_model, lda_vectorizer, board_vocab)
            stored += 1
    
    return stored


# ── Similarity comparison ────────────────────────────────────────

def _score_with_model(fp_a, fp_b, model):
    """Compute similarity score using the trained ML model.
    
    Returns (score, verdict).
    """
    diff = np.abs(fp_a - fp_b).reshape(1, -1)
    diff = np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
    score, verdict, _ = model.predict_pair_twd(diff)
    return score, verdict


def _score_with_cosine(fp_a, fp_b):
    """Compute similarity score using cosine similarity as a fallback.
    
    Returns (score, verdict).
    """
    from sklearn.metrics.pairwise import cosine_similarity
    score = float(cosine_similarity(fp_a.reshape(1, -1), fp_b.reshape(1, -1))[0][0])
    if score > 0.8:
        verdict = "MATCH"
    elif score < 0.3:
        verdict = "MISMATCH"
    else:
        verdict = "UNCERTAIN"
    return score, verdict


def _compute_similarity_score(fp_a, fp_b, model=None):
    """Compute similarity score between two fingerprint vectors.
    
    Uses trained model if available, falls back to cosine similarity.
    
    Returns:
        tuple: (score, verdict)
    """
    if model and model.is_trained:
        return _score_with_model(fp_a, fp_b, model)
    return _score_with_cosine(fp_a, fp_b)


def find_similar_users(username, forum, model=None, top_n=10):
    """Compare one user against ALL stored fingerprints.
    
    Args:
        username: target user
        forum: target user's forum
        model: trained AuthorSimilarityModel (if None, uses cosine similarity)
        top_n: return top N matches
    
    Returns:
        list of (username, forum, score, post_count) sorted by score desc
    """
    target = get_fingerprint(username, forum)
    if target is None:
        return []
    
    all_fps = get_all_fingerprints()
    target_fp = target["fingerprint"]
    
    results = []
    for fp_data in all_fps:
        if fp_data["username"] == username and fp_data["forum"] == forum:
            continue
        
        other_fp = fp_data["fingerprint"]
        score, verdict = _compute_similarity_score(target_fp, other_fp, model)
        
        results.append({
            "username": fp_data["username"],
            "forum": fp_data["forum"],
            "score": round(score, 4),
            "verdict": verdict,
            "post_count": fp_data["post_count"],
        })
    
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]


def _identify_matches_above_threshold(all_fps, threshold, model):
    results = []
    n = len(all_fps)
    for i in range(n):
        for j in range(i + 1, n):
            fp_a = all_fps[i]
            fp_b = all_fps[j]
            score, verdict = _compute_similarity_score(fp_a["fingerprint"], fp_b["fingerprint"], model)
            if score >= threshold:
                results.append({
                    "user_a": f"{fp_a['username']}@{fp_a['forum']}",
                    "user_b": f"{fp_b['username']}@{fp_b['forum']}",
                    "score": round(score, 4),
                    "verdict": verdict,
                })
    return results


def compare_all_users(model=None, threshold=0.5):
    """Compare ALL user pairs and return matches above threshold.
    
    Args:
        model: trained model (uses cosine similarity if None)
        threshold: minimum score to include
    
    Returns:
        list of matches sorted by score
    """
    all_fps = get_all_fingerprints()
    if len(all_fps) < 2:
        return []
    
    results = _identify_matches_above_threshold(all_fps, threshold, model)
    results.sort(key=lambda x: x["score"], reverse=True)
    
    _insert_similarity_results(results, model_version="v1")
    
    return results


# ── Training log ─────────────────────────────────────────────────

def log_training(metrics, n_users=0, n_posts_total=0, model_version="v1"):
    """Log a training run to the database.
    
    Called automatically by the updated model.train().
    """
    _insert_training_log_row(n_users, n_posts_total, metrics, model_version)


def get_training_history():
    """Get all training runs."""
    conn = _get_conn()
    c = conn.cursor()
    
    c.execute("""
        SELECT trained_at, n_users, n_posts_total, n_pairs,
               cv_roc_auc, cv_accuracy, train_roc_auc, model_version
        FROM training_log ORDER BY trained_at DESC
    """)
    
    rows = c.fetchall()
    conn.close()
    
    return [{
        "trained_at": r[0],
        "n_users": r[1],
        "n_posts_total": r[2],
        "n_pairs": r[3],
        "cv_roc_auc": r[4],
        "cv_accuracy": r[5],
        "train_roc_auc": r[6],
        "model_version": r[7],
    } for r in rows]


# ── Stats ────────────────────────────────────────────────────────

def get_store_stats():
    """Get summary stats about the fingerprint store."""
    conn = _get_conn()
    c = conn.cursor()
    
    c.execute("SELECT COUNT(*) FROM fingerprints")
    n_fingerprints = c.fetchone()[0]
    
    c.execute("SELECT SUM(post_count), SUM(total_words) FROM fingerprints")
    row = c.fetchone()
    total_posts = row[0] or 0
    total_words = row[1] or 0
    
    c.execute("SELECT COUNT(DISTINCT forum) FROM fingerprints")
    n_forums = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM training_log")
    n_trainings = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM similarity_results")
    n_cached_results = c.fetchone()[0]
    
    conn.close()
    
    return {
        "fingerprints": n_fingerprints,
        "total_posts": total_posts,
        "total_words": total_words,
        "forums": n_forums,
        "training_runs": n_trainings,
        "cached_comparisons": n_cached_results,
    }
