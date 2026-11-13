"""
dashboard.py - Flask backend for the MMFIC web dashboard.

Serves as the API backend for a single-page application that provides
identity-comparison analytics, crawl management, and ML model training.

SECURITY NOTICE:
    NEVER expose this dashboard beyond localhost (127.0.0.1). It controls active
    network crawlers (.onion / clearnet) and holds sensitive OSINT datasets.
    All /api/* endpoints strictly require an X-API-Token header (configured via
    DASHBOARD_API_TOKEN or generated ephemerally per startup).
"""

import argparse
from collections import deque
from contextlib import closing
from datetime import datetime
import hmac
import io
import itertools
import json
import logging
import os
from pathlib import Path
import secrets
import sqlite3
import sys
import threading
import time
import traceback
from urllib.parse import urlparse
import warnings

import numpy as np
from dotenv import load_dotenv
from flask import Flask, render_template, jsonify, request
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from app.database import DB_PATH, get_user_posts
from app.crawler.crawler import crawl
from app.crawler.user_crawler import crawl_user
from app.crawler.twitter_adapter import is_twitter_url, crawl_twitter_user, extract_username_from_url
from Model.fingerprint_store import (
    get_fingerprint, get_all_fingerprints, find_similar_users, init_fingerprint_tables,
)
from Model.model import AuthorSimilarityModel
from Model.data_prep import create_training_data_from_file
from Model.features import get_feature_names, CANONICAL_FEATURE_DIM, CANONICAL_LDA_TOPICS
from Model.thresholds import (
    MATCH_THRESHOLD, MISMATCH_THRESHOLD, COSINE_TIE_BREAKER,
    DEFAULT_CANDIDATE_THRESHOLD, VERDICT_MATCH, VERDICT_MISMATCH, VERDICT_UNCERTAIN
)
import app.crawler.browser_fetcher as bf
from app.crawler.crawl_utils import persist_crawl_results
from app.diagnostics import (
    add_verbosity_args, configure_logging, crawl_callbacks, log_failure, sanitize_finite_array
)

app = Flask(__name__)


def _init_api_token():
    """Load DASHBOARD_API_TOKEN from env or generate an ephemeral session token."""
    env_token = (os.getenv("DASHBOARD_API_TOKEN") or "").strip()
    if env_token:
        return env_token, True
    return secrets.token_urlsafe(32), False


_API_TOKEN, _TOKEN_FROM_ENV = _init_api_token()
_ALLOWED_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
_RATE_WINDOW_SEC = 60.0
_rate_lock = threading.Lock()
_rate_buckets = {}  # (client_ip, tier) -> deque[monotonic timestamp]


def _is_rate_limited(bucket_key, max_requests, window_sec=_RATE_WINDOW_SEC, record_hit=True):
    """Thread-safe sliding-window rate limiter."""
    now = time.monotonic()
    cutoff = now - window_sec
    with _rate_lock:
        dq = _rate_buckets.setdefault(bucket_key, deque())
        while dq and dq[0] <= cutoff:
            dq.popleft()
        if len(dq) >= max_requests:
            return True
        if record_hit:
            dq.append(now)
        return False


def _is_allowed_origin():
    """Verify Origin/Referer headers (when present) target localhost only."""
    for hdr_name in ("Origin", "Referer"):
        hdr = request.headers.get(hdr_name)
        if not hdr:
            continue
        try:
            host = (urlparse(hdr).hostname or "").lower()
            if host not in _ALLOWED_LOCAL_HOSTS:
                return False
        except Exception:
            return False
    return True


def _is_valid_identifier(value, max_len=128):
    """Reject empty, oversized, or control-character strings."""
    if not value or len(value) > max_len:
        return False
    return not any(ord(ch) < 32 or ord(ch) == 127 for ch in value)


def _server_error(context, exc):
    """Log detailed exception server-side without leaking internals to clients."""
    print(f"[ERROR] {context}: {exc}", file=sys.stderr)
    traceback.print_exc()
    return jsonify({"error": "Internal server error. Check server console logs."}), 500


@app.before_request
def _enforce_api_security():
    """Enforce origin checks, mandatory X-API-Token auth, and rate limits on /api/*."""
    if not request.path.startswith("/api/"):
        return None

    client_ip = request.remote_addr or "local"

    if not _is_allowed_origin():
        return jsonify({"error": "Forbidden: cross-origin request blocked."}), 403

    if _is_rate_limited((client_ip, "auth_fail"), max_requests=10, record_hit=False):
        return jsonify({"error": "Too many failed authentication attempts. Retry later."}), 429

    provided = (request.headers.get("X-API-Token") or "").strip()
    if not provided or not _API_TOKEN or not hmac.compare_digest(provided, _API_TOKEN):
        _is_rate_limited((client_ip, "auth_fail"), max_requests=10, record_hit=True)
        return jsonify({"error": "Unauthorized. Missing or invalid X-API-Token header."}), 401

    if request.path in (
        "/api/crawl/general", "/api/crawl/user", "/api/model/train",
        "/api/compare/pair", "/api/compare/all"
    ):
        if _is_rate_limited((client_ip, "heavy"), max_requests=15):
            return jsonify({"error": "Rate limit exceeded for heavy operation."}), 429
    elif request.path.startswith("/api/user/") or request.path.startswith("/api/users/"):
        if _is_rate_limited((client_ip, "user_lookup"), max_requests=20):
            return jsonify({"error": "Rate limit exceeded for user lookup."}), 429
    else:
        if _is_rate_limited((client_ip, "standard"), max_requests=120):
            return jsonify({"error": "Rate limit exceeded."}), 429

    return None


@app.after_request
def _add_security_headers(response):
    """Attach baseline browser security headers."""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    if request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response

_crawl_lock = threading.Lock()
_train_lock = threading.Lock()
_fingerprint_sync_lock = threading.Lock()
_skipped_short_users = set()

crawl_state = {
    "running": False, "status": "idle", "result": None, "error": None,
    "posts_found": 0, "started_at": None, "log": [],
}
train_state = {"running": False, "status": "idle", "error": None, "metrics": None}
_crawl_thread = None


def _update_crawl_state(**fields):
    """Thread-safely update fields in crawl_state."""
    with _crawl_lock:
        crawl_state.update(fields)


def _snapshot_crawl_state():
    """Return a consistent point-in-time snapshot of crawl_state."""
    with _crawl_lock:
        snap = dict(crawl_state)
        snap["log"] = list(crawl_state.get("log", []))
        return snap


def _update_train_state(**fields):
    """Thread-safely update fields in train_state."""
    with _train_lock:
        train_state.update(fields)


def _snapshot_train_state():
    """Return a consistent point-in-time snapshot of train_state."""
    with _train_lock:
        return dict(train_state)

_model_lock = threading.Lock()
_cached_model = None


def _get_model():
    """Return the cached model, loading it lazily in a thread-safe way."""
    global _cached_model
    if _cached_model is None:
        with _model_lock:
            if _cached_model is None:
                _cached_model = AuthorSimilarityModel.load()
    return _cached_model


def _db():
    """Open a new SQLite connection with Row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _read_model_meta():
    """Read model_meta.json and return (last_trained str | None)."""
    meta_path = PROJECT_ROOT / "Model" / "model_meta.json"
    if not meta_path.exists():
        return None
    with open(meta_path, "r") as f:
        return json.load(f).get("trained_at")


def _query_max_crawled_at():
    """Return the latest crawled_at timestamp from the posts table, or None."""
    try:
        with closing(_db()) as conn:
            return conn.execute("SELECT MAX(crawled_at) FROM posts").fetchone()[0]
    except Exception as exc:
        log_failure("Failed querying max crawled_at", exc, level=logging.WARNING, logger_name="dashboard")
        return None


def _check_model_staleness():
    """Check if the ML model is trained and if it's stale compared to the DB."""
    last_trained = _read_model_meta()
    if not last_trained:
        return False, False, None

    model_stale = False
    max_crawled = _query_max_crawled_at()
    if max_crawled and last_trained < max_crawled:
        model_stale = True
    return True, model_stale, last_trained


@app.route("/")
def index():
    """Serve the main SPA page."""
    return render_template("index.html")


def _query_db_stats():
    """Query platforms, profiles, posts count and max crawled_at from DB."""
    with closing(_db()) as conn:
        platforms = conn.execute("SELECT COUNT(DISTINCT forum) FROM posts").fetchone()[0]
        profiles = conn.execute("SELECT COUNT(DISTINCT username || forum) FROM posts").fetchone()[0]
        posts = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
        max_crawled = conn.execute("SELECT MAX(crawled_at) FROM posts").fetchone()[0]
    return platforms, profiles, posts, max_crawled


@app.route("/api/stats")
def api_stats():
    """Return high-level database statistics and model staleness flag."""
    try:
        platforms, profiles, posts, _ = _query_db_stats()
        model_trained, model_stale, _ = _check_model_staleness()
        return jsonify({
            "platforms": platforms,
            "profiles": profiles,
            "posts": posts,
            "model_trained": model_trained,
            "model_stale": model_stale,
        })
    except Exception as e:
        return _server_error("api_stats", e)


def _query_all_users_grouped():
    """Return all users grouped by forum as a dict."""
    with closing(_db()) as conn:
        rows = conn.execute(
            "SELECT username, forum, COUNT(*) AS post_count "
            "FROM posts GROUP BY username, forum ORDER BY post_count DESC"
        ).fetchall()
    grouped = {}
    for r in rows:
        grouped.setdefault(r["forum"], []).append({
            "username": r["username"],
            "post_count": r["post_count"],
        })
    return grouped


@app.route("/api/users")
def api_users():
    """Return all users grouped by platform."""
    try:
        return jsonify(_query_all_users_grouped())
    except Exception as e:
        return _server_error("api_users", e)


def _query_users_for_forum(forum):
    """Return users for a specific forum as a list of dicts."""
    with closing(_db()) as conn:
        rows = conn.execute(
            "SELECT username, forum, COUNT(*) AS post_count "
            "FROM posts WHERE forum = ? GROUP BY username, forum "
            "ORDER BY post_count DESC", (forum,)
        ).fetchall()
    return [{"username": r["username"], "post_count": r["post_count"]} for r in rows]


@app.route("/api/users/<path:forum>")
def api_users_forum(forum):
    """Return users for a specific platform."""
    try:
        if not _is_valid_identifier(forum, 255):
            return jsonify({"error": "Invalid forum identifier."}), 400
        return jsonify(_query_users_for_forum(forum))
    except Exception as e:
        return _server_error("api_users_forum", e)


@app.route("/api/user/<username>/<path:forum>/posts")
def api_user_posts(username, forum):
    """Return up to 50 posts for a given user on a given forum."""
    try:
        if not _is_valid_identifier(username, 128) or not _is_valid_identifier(forum, 255):
            return jsonify({"error": "Invalid username or forum identifier."}), 400
        posts = get_user_posts(username, forum)
        if not posts:
            return jsonify({"error": "Not found."}), 404
        return jsonify(posts[:50])
    except Exception as e:
        return _server_error("api_user_posts", e)


def _ensure_fingerprint_exists(username, forum):
    """Build and cache a fingerprint for a user if one is not already stored."""
    from Model.fingerprint_store import update_fingerprint
    if get_fingerprint(username, forum) is None:
        update_fingerprint(username, forum)
    return get_fingerprint(username, forum)


def _predict_pair_score(fp_a_arr, fp_b_arr):
    """Compute similarity score and verdict for two fingerprint arrays using server-governed canonical thresholds."""
    from Model.features import CANONICAL_FEATURE_DIM

    if fp_a_arr.shape != fp_b_arr.shape:
        raise ValueError(f"Fingerprint shape mismatch: {fp_a_arr.shape} vs {fp_b_arr.shape}")

    model = _get_model()
    expected_dim = getattr(getattr(model, 'classifier', None), 'n_features_in_', 
                           getattr(getattr(model, 'scaler', None), 'n_features_in_', CANONICAL_FEATURE_DIM))
    if fp_a_arr.shape != (expected_dim,):
        raise ValueError(f"Fingerprint dimension mismatch: expected ({expected_dim},), got {fp_a_arr.shape}")

    diff = np.abs(fp_a_arr - fp_b_arr).reshape(1, -1)
    diff = sanitize_finite_array(diff, context="_predict_pair_score")

    if model is not None and model.is_trained:
        score, verdict, meta = model.predict_pair_twd(diff, MATCH_THRESHOLD, MISMATCH_THRESHOLD, COSINE_TIE_BREAKER)
    else:
        score = float(cosine_similarity(fp_a_arr.reshape(1, -1), fp_b_arr.reshape(1, -1))[0][0])
        verdict = VERDICT_UNCERTAIN
        if score >= MATCH_THRESHOLD:
            verdict = VERDICT_MATCH
        elif score <= MISMATCH_THRESHOLD:
            verdict = VERDICT_MISMATCH
        meta = {"reason": "Fallback cosine similarity"}

    meta["thresholds"] = {
        "match": MATCH_THRESHOLD,
        "mismatch": MISMATCH_THRESHOLD,
        "tie_breaker": COSINE_TIE_BREAKER,
    }
    return score, verdict, meta


def _query_post_count(username, forum):
    """Return the post count for a single user+forum pair."""
    with closing(_db()) as conn:
        cnt = conn.execute(
            "SELECT COUNT(*) FROM posts WHERE username=? AND forum=?",
            (username, forum)
        ).fetchone()[0]
    return cnt


@app.route("/api/compare/pair", methods=["POST"])
def api_compare_pair():
    """Compare two users and return a similarity score + verdict.
    
    Forensic integrity note: Classification thresholds are server-governed and immutable
    to prevent client-side verdict manipulation.
    """
    try:
        data = request.get_json(force=True)
        user_a, forum_a = data["user_a"], data["forum_a"]
        user_b, forum_b = data["user_b"], data["forum_b"]

        fp_a = _ensure_fingerprint_exists(user_a, forum_a)
        fp_b = _ensure_fingerprint_exists(user_b, forum_b)

        score, verdict, meta = _predict_pair_score(
            fp_a["fingerprint"], fp_b["fingerprint"]
        )

        return jsonify({
            "score": score,
            "verdict": verdict,
            "user_a": {"username": user_a, "forum": forum_a, "post_count": _query_post_count(user_a, forum_a)},
            "user_b": {"username": user_b, "forum": forum_b, "post_count": _query_post_count(user_b, forum_b)},
            "meta": meta,
        })
    except (ValueError, KeyError, TypeError) as e:
        return jsonify({"error": f"Invalid request parameters: {e}"}), 400
    except Exception as e:
        return _server_error("api_compare_pair", e)


MAX_SYNC_FP_BUILD = 100


def _ensure_all_fingerprints_exist():
    """Synchronize missing fingerprints with a single LEFT JOIN and cached model components."""
    from Model.fingerprint_store import update_fingerprint, _load_lda_components_from_model

    with _fingerprint_sync_lock:
        with closing(_db()) as conn:
            rows = conn.execute(
                "SELECT p.username, p.forum "
                "FROM posts p "
                "LEFT JOIN fingerprints f ON p.username = f.username AND p.forum = f.forum "
                "WHERE f.id IS NULL AND p.username != 'unknown' AND p.content IS NOT NULL "
                "GROUP BY p.username, p.forum"
            ).fetchall()

        candidates = [
            (r["username"], r["forum"])
            for r in rows
            if (r["username"], r["forum"]) not in _skipped_short_users
        ]
        if not candidates:
            return

        model = None
        try:
            model = _get_model()
        except Exception as exc:
            log_failure("Cached model unavailable during fingerprint sync", exc, level=logging.DEBUG, logger_name="dashboard")
        if model is not None:
            lda_model, lda_vec, board_vocab = model.lda_model, model.lda_vectorizer, model.board_vocab
        else:
            lda_model, lda_vec, board_vocab = _load_lda_components_from_model()

        for uname, frm in candidates[:MAX_SYNC_FP_BUILD]:
            if update_fingerprint(uname, frm, lda_model, lda_vec, board_vocab) is None:
                _skipped_short_users.add((uname, frm))


@app.route("/api/compare/all", methods=["POST"])
def api_compare_all():
    """Find the most similar users to a given user across all platforms."""
    try:
        data = request.get_json(force=True)
        username = data["username"]
        forum = data["forum"]
        top_n = int(data.get("top_n", 10))

        if _ensure_fingerprint_exists(username, forum) is None:
            return jsonify({"error": "User not found or lacks substantial posts to build fingerprint."}), 404

        _ensure_all_fingerprints_exist()

        matches = find_similar_users(username, forum, top_n=top_n)
        return jsonify(matches)
    except (ValueError, KeyError, TypeError) as e:
        return jsonify({"error": f"Invalid request parameters: {e}"}), 400
    except Exception as e:
        return _server_error("api_compare_all", e)


DEFAULT_GRAPH_MAX_NODES = 250
HARD_GRAPH_MAX_NODES = 500
MAX_GRAPH_EDGES = 2000
GRAPH_CHUNK_SIZE = 4096


def _build_graph_nodes(all_fps):
    """Build node list and vector list from fingerprints."""
    nodes = []
    vectors = []
    node_lookup = []
    for fp in all_fps:
        uname = fp["username"]
        forum = fp["forum"]
        post_count = fp["post_count"]
        node_id = f"{uname}|{forum}"
        node_lookup.append(node_id)
        vectors.append(np.array(fp["fingerprint"]))
        nodes.append({
            "id": node_id,
            "label": f"{uname}\n({post_count} posts)",
            "group": forum,
            "value": post_count,
            "title": f"User: {uname}<br>Forum: {forum}<br>Posts: {post_count}"
        })
    return nodes, vectors, node_lookup


def _validate_graph_vectors(vectors, expected_dim):
    """Ensure every vector matches expected_dim before stacking."""
    for idx, v in enumerate(vectors):
        if v.shape != (expected_dim,):
            raise ValueError(f"Vector {idx} dimension mismatch: expected ({expected_dim},), got {v.shape}")


def _build_graph_edges_with_model(vectors, node_lookup, model, threshold,
                                  chunk_size=GRAPH_CHUNK_SIZE, max_edges=MAX_GRAPH_EDGES):
    """Compute edges using chunked vectorized batching to avoid O(n^2) memory blowup."""
    expected_dim = getattr(getattr(model, 'classifier', None), 'n_features_in_', 
                           getattr(getattr(model, 'scaler', None), 'n_features_in_', CANONICAL_FEATURE_DIM))
    _validate_graph_vectors(vectors, expected_dim)

    V = np.asarray(vectors, dtype=np.float64)
    edges = []
    pair_iter = itertools.combinations(range(len(vectors)), 2)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=".*X does not have valid feature names.*")
        while True:
            chunk = list(itertools.islice(pair_iter, chunk_size))
            if not chunk:
                break
            idx_a = np.fromiter((p[0] for p in chunk), dtype=np.intp, count=len(chunk))
            idx_b = np.fromiter((p[1] for p in chunk), dtype=np.intp, count=len(chunk))
            X_chunk = sanitize_finite_array(np.abs(V[idx_a] - V[idx_b]), context="_build_graph_edges_model")
            X_scaled = model.scaler.transform(X_chunk)
            probs = model.classifier.predict_proba(X_scaled)[:, 1]
            for pos in np.where(probs >= threshold)[0]:
                i, j = chunk[int(pos)]
                score = float(probs[pos])
                edges.append({
                    "from": node_lookup[i],
                    "to": node_lookup[j],
                    "value": score,
                    "title": f"Similarity: {score:.3f}",
                })

    if max_edges and len(edges) > max_edges:
        edges.sort(key=lambda e: e["value"], reverse=True)
        edges = edges[:max_edges]
    return edges


def _build_graph_edges_cosine(vectors, node_lookup, threshold,
                              chunk_size=256, max_edges=MAX_GRAPH_EDGES):
    """Compute edges using chunked matrix cosine similarity to avoid O(n^2) slow loops."""
    _validate_graph_vectors(vectors, CANONICAL_FEATURE_DIM)

    V = sanitize_finite_array(np.asarray(vectors, dtype=np.float64), context="_build_graph_edges_cosine")
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    V_norm = V / norms

    n = len(V)
    edges = []
    cols = np.arange(n)

    for i_start in range(0, n, chunk_size):
        i_end = min(i_start + chunk_size, n)
        sim_block = V_norm[i_start:i_end] @ V_norm.T
        rows = np.arange(i_start, i_end)
        mask = (rows[:, None] < cols[None, :]) & (sim_block >= threshold)
        rel_r, abs_c = np.where(mask)
        for rr, cc in zip(rel_r, abs_c):
            score = float(sim_block[rr, cc])
            edges.append({
                "from": node_lookup[i_start + int(rr)],
                "to": node_lookup[int(cc)],
                "value": score,
                "title": f"Similarity: {score:.3f}",
            })

    if max_edges and len(edges) > max_edges:
        edges.sort(key=lambda e: e["value"], reverse=True)
        edges = edges[:max_edges]
    return edges


@app.route("/api/compare/graph")
def api_compare_graph():
    """Build a bounded similarity graph of top fingerprinted users."""
    try:
        threshold = float(request.args.get("threshold", DEFAULT_CANDIDATE_THRESHOLD))
        requested_max = int(request.args.get("max_nodes", DEFAULT_GRAPH_MAX_NODES))
        max_nodes = max(1, min(requested_max, HARD_GRAPH_MAX_NODES))

        all_fps = get_all_fingerprints()
        if not all_fps:
            return jsonify({"nodes": [], "edges": []})

        total_users = len(all_fps)
        truncated = total_users > max_nodes
        if truncated:
            all_fps = sorted(all_fps, key=lambda fp: fp.get("post_count", 0), reverse=True)[:max_nodes]

        nodes, vectors, node_lookup = _build_graph_nodes(all_fps)

        edges = []
        if len(vectors) > 1:
            model = _get_model()
            if model is not None and model.is_trained:
                edges = _build_graph_edges_with_model(vectors, node_lookup, model, threshold)
            else:
                edges = _build_graph_edges_cosine(vectors, node_lookup, threshold)

        return jsonify({
            "nodes": nodes,
            "edges": edges,
            "total_users": total_users,
            "truncated": truncated,
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return _server_error("api_compare_graph", e)


@app.route("/api/model/status")
def api_model_status():
    """Return the current model status."""
    try:
        model = _get_model()
        is_trained = model is not None and model.is_trained
        _, is_stale, last_trained = _check_model_staleness()
        return jsonify({
            "is_trained": is_trained,
            "is_stale": is_stale,
            "last_trained": last_trained,
            "training_in_progress": _snapshot_train_state()["running"],
        })
    except Exception as e:
        return _server_error("api_model_status", e)


@app.route("/api/model/importances")
def api_model_importances():
    """Return the top 30 feature importances from the trained classifier."""
    try:
        model = _get_model()
        if model is None or not model.is_trained:
            return jsonify({"error": "Model is not trained"}), 400
        importances = model.classifier.feature_importances_
        n_lda = getattr(model.lda_model, "n_components", CANONICAL_LDA_TOPICS)
        names = get_feature_names(lda_topics=n_lda, board_vocab=model.board_vocab)
        if len(names) != len(importances):
            raise ValueError(
                f"Feature name count ({len(names)}) does not match model importances ({len(importances)})"
            )
        paired = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)
        top = [{"feature": n, "importance": float(v)} for n, v in paired[:30]]
        return jsonify(top)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return _server_error("api_model_importances", e)


def _train_model_task(lda_topics_val):
    """Background thread: prepare data, train, save, and rebuild fingerprints."""
    global _cached_model
    try:
        _update_train_state(running=True, status="preparing data", error=None, metrics=None)
        X, y, pair_info, lda_model, lda_vec, board_vocab = create_training_data_from_file(lda_topics=lda_topics_val)

        _update_train_state(status="training model")
        new_model = AuthorSimilarityModel()
        feature_names = get_feature_names(lda_topics=lda_topics_val, board_vocab=board_vocab)
        metrics = new_model.train(X, y, feature_names, lda_model, lda_vec, board_vocab)

        _update_train_state(status="saving model")
        new_model.save()

        with _model_lock:
            _cached_model = new_model

        _update_train_state(status="rebuilding fingerprints")
        from Model.fingerprint_store import rebuild_all_fingerprints
        rebuild_all_fingerprints()
        with _fingerprint_sync_lock:
            _skipped_short_users.clear()

        _update_train_state(running=False, status="complete", metrics=metrics)
    except Exception as exc:
        print(f"[ERROR] _train_model_task: {exc}", file=sys.stderr)
        traceback.print_exc()
        _update_train_state(running=False, status="failed", error="Training failed. Check server console logs.")


@app.route("/api/model/train", methods=["POST"])
def api_model_train():
    """Start model training in a background thread atomically."""
    try:
        data = request.get_json(force=True, silent=True) or {}
        lda_topics = int(data.get("lda_topics", 20))
        with _train_lock:
            if train_state["running"]:
                return jsonify({"error": "Training already in progress"}), 409
            train_state.update({"running": True, "status": "starting", "error": None, "metrics": None})
            t = threading.Thread(target=_train_model_task, args=(lda_topics,), daemon=True)
            t.start()
        return jsonify({"message": "Training started"})
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid training parameter: {e}"}), 400
    except Exception as e:
        return _server_error("api_model_train", e)


@app.route("/api/model/train/status")
def api_model_train_status():
    """Return a consistent snapshot of the current training state."""
    return jsonify(_snapshot_train_state())


def _log_crawl(msg):
    """Thread-safely add a message to the crawl log (keeps last 5 entries)."""
    with _crawl_lock:
        crawl_state["log"].append(msg)
        if len(crawl_state["log"]) > 5:
            crawl_state["log"] = crawl_state["log"][-5:]


class TeeWriter:
    """Writes to both real stdout and a capture buffer, forwarding log lines safely."""
    def __init__(self, real, capture):
        self.real = real
        self.capture = capture

    def write(self, text):
        self.real.write(text)
        self.capture.write(text)
        for line in text.strip().split('\n'):
            line = line.strip()
            if line:
                _log_crawl(line)

    def flush(self):
        self.real.flush()

    def isatty(self):
        return False


def _on_captcha_pending():
    """Explicit callback triggered when Playwright waits on HITL CAPTCHA."""
    _update_crawl_state(status="captcha_pending")


def _on_posts_found(count):
    """Explicit callback triggered by crawlers when new posts are collected."""
    with _crawl_lock:
        crawl_state["posts_found"] = crawl_state.get("posts_found", 0) + int(count)


def _save_crawl_results(results):
    """Normalize, save to DB, append to JSON, and clear skipped short-user cache."""
    normalized, _ = persist_crawl_results(results, json_filepath=str(PROJECT_ROOT / "crawl_data.json"))
    with _fingerprint_sync_lock:
        _skipped_short_users.clear()
    return normalized


def _run_crawl_with_progress(crawl_func, *args, **kwargs):
    """Run a crawl function with explicit lifecycle callbacks and stdout log mirror."""
    bf.DASHBOARD_MODE = True

    try:
        bf.ensure_clean_browser()
    except Exception as exc:
        log_failure("Pre-crawl browser cleanup failed", exc, level=logging.DEBUG, logger_name="dashboard")

    real_stdout = sys.stdout
    capture = io.StringIO()
    sys.stdout = TeeWriter(real_stdout, capture)
    try:
        with crawl_callbacks(
            on_captcha_pending=_on_captcha_pending,
            on_posts_found=_on_posts_found,
        ):
            results = crawl_func(*args, **kwargs)
    finally:
        sys.stdout = real_stdout
        bf.DASHBOARD_MODE = False

    return results


def _run_background_crawl(crawl_func, *args, **kwargs):
    """Shared background thread for all crawl types."""
    global _crawl_thread
    results = []
    try:
        results = _run_crawl_with_progress(crawl_func, *args, **kwargs) or []

        if results:
            results = _save_crawl_results(results)

        if bf.is_stop_requested() or not _snapshot_crawl_state().get("running", True):
            _update_crawl_state(
                running=False, status="stopped", error=None,
                result={"posts_found": len(results)},
            )
            return

        _update_crawl_state(
            running=False,
            status="complete",
            error=None,
            result={"posts_found": len(results)},
        )
    except bf.CrawlAbortedByUser:
        _update_crawl_state(
            running=False, status="stopped", error=None,
            result={"posts_found": len(results)},
        )
    except Exception as exc:
        log_failure("Background crawl failed", exc, level=logging.ERROR, logger_name="dashboard")
        traceback.print_exc()
        if bf.is_stop_requested():
            _update_crawl_state(
                running=False, status="stopped", error=None,
                result={"posts_found": len(results)},
            )
        else:
            _update_crawl_state(
                running=False, status="failed", error="Crawl failed. Check server console logs."
            )
    finally:
        # Teardown Playwright resources strictly on the owning worker thread
        try:
            bf.close_browser()
        except Exception as exc:
            log_failure("Failed closing browser on worker exit", exc, level=logging.DEBUG, logger_name="dashboard")
        with _crawl_lock:
            _crawl_thread = None


def _start_crawl_thread(status_msg, crawl_func, *args, **kwargs):
    """Atomically initialize state and start background crawl thread, or return False if busy."""
    global _crawl_thread
    with _crawl_lock:
        if crawl_state["running"] or (_crawl_thread is not None and _crawl_thread.is_alive()):
            return False
        bf.clear_stop_request()
        crawl_state.update({
            "running": True, "status": status_msg, "result": None,
            "error": None, "posts_found": 0, "log": [],
            "started_at": datetime.now().isoformat(),
        })
        _crawl_thread = threading.Thread(
            target=_run_background_crawl, args=(crawl_func,) + args, kwargs=kwargs, daemon=True
        )
        _crawl_thread.start()
        return True


@app.route("/api/tor/status")
def api_tor_status():
    """Return current Tor connectivity status and exit IP."""
    try:
        force = request.args.get("force", "false").lower() == "true"
        status = bf.get_tor_details(force=force)
        return jsonify(status)
    except Exception as e:
        print(f"[ERROR] api_tor_status: {e}", file=sys.stderr)
        traceback.print_exc()
        return jsonify({"tor_active": False, "is_tor": False, "error": "Tor status check failed. Check server console logs."}), 500


@app.route("/api/crawl/general", methods=["POST"])
def api_crawl_general():
    """Start a general forum crawl in a background thread."""
    try:
        data = request.get_json(force=True)
        url = data["url"]
        max_pages = data.get("max_pages", 5)
        max_threads = data.get("max_threads", 10)

        try:
            bf.ensure_tor_safety(url)
        except bf.TorOPSECError as oe:
            return jsonify({"error": str(oe)}), 403

        if is_twitter_url(url):
            tw_username = extract_username_from_url(url)
            if not tw_username:
                return jsonify({"error": "Could not extract username from Twitter URL."}), 400
            if not _start_crawl_thread(
                f"crawling twitter @{tw_username}", crawl_twitter_user,
                url, username=tw_username, max_tweets=max_pages * 50,
            ):
                return jsonify({"error": "Crawl already in progress"}), 409
            return jsonify({"message": f"Twitter crawl started for @{tw_username}"})

        if not _start_crawl_thread("crawling", crawl, url, max_pages, max_threads):
            return jsonify({"error": "Crawl already in progress"}), 409
        return jsonify({"message": "Crawl started"})
    except (ValueError, KeyError, TypeError) as e:
        return jsonify({"error": f"Invalid crawl request: {e}"}), 400
    except Exception as e:
        return _server_error("api_crawl_general", e)


@app.route("/api/crawl/user", methods=["POST"])
def api_crawl_user():
    """Start a user-specific crawl in a background thread."""
    try:
        data = request.get_json(force=True)
        url = data["url"]
        username = data["username"]
        max_pages = data.get("max_pages", 5)

        try:
            bf.ensure_tor_safety(url)
        except bf.TorOPSECError as oe:
            return jsonify({"error": str(oe)}), 403

        if is_twitter_url(url):
            tw_username = extract_username_from_url(url) or username
            if not _start_crawl_thread(
                f"crawling twitter @{tw_username}", crawl_twitter_user,
                url, username=tw_username, max_tweets=max_pages * 50,
            ):
                return jsonify({"error": "Crawl already in progress"}), 409
            return jsonify({"message": "User crawl started"})

        if not _start_crawl_thread("crawling user", crawl_user, url, username, max_pages):
            return jsonify({"error": "Crawl already in progress"}), 409
        return jsonify({"message": "User crawl started"})
    except (ValueError, KeyError, TypeError) as e:
        return jsonify({"error": f"Invalid user crawl request: {e}"}), 400
    except Exception as e:
        return _server_error("api_crawl_user", e)


def _compute_elapsed_seconds(started_at_iso):
    """Return elapsed seconds since started_at_iso, or None on parse failure."""
    try:
        started = datetime.fromisoformat(started_at_iso)
        return int((datetime.now() - started).total_seconds())
    except Exception as exc:
        log_failure("Failed parsing crawl started_at timestamp", exc, level=logging.DEBUG, logger_name="dashboard")
        return None


@app.route("/api/crawl/status")
def api_crawl_status():
    """Return a consistent snapshot of the current crawl state."""
    try:
        state = _snapshot_crawl_state()
        if state["running"] and state.get("started_at"):
            elapsed = _compute_elapsed_seconds(state["started_at"])
            if elapsed is not None:
                state["elapsed_seconds"] = elapsed
        return jsonify(state)
    except Exception as e:
        return _server_error("api_crawl_status", e)


@app.route("/api/crawl/captcha_action", methods=["POST"])
def api_crawl_captcha_action():
    """Handle HITL CAPTCHA action from dashboard."""
    try:
        state = _snapshot_crawl_state()
        if not state.get("running") or state.get("status") != "captcha_pending":
            return jsonify({"error": "No CAPTCHA challenge is currently awaiting user input."}), 409

        data = request.get_json(force=True)
        if not isinstance(data, dict):
            raise ValueError("JSON object expected")
        action = (data.get("action") or "").strip().lower()
        if action not in bf.VALID_CAPTCHA_ACTIONS:
            return jsonify({"error": f"Invalid action. Allowed: {sorted(bf.VALID_CAPTCHA_ACTIONS)}"}), 400

        if not bf.submit_captcha_action(action, force=False):
            return jsonify({"error": "CAPTCHA response already queued."}), 409

        _update_crawl_state(status="stopping..." if action == "exit" else "crawling")
        return jsonify({"success": True})
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid CAPTCHA action payload: {e}"}), 400
    except Exception as e:
        return _server_error("api_crawl_captcha_action", e)


@app.route("/api/crawl/stop", methods=["POST"])
def api_crawl_stop():
    """Signal active crawl to stop cooperatively without cross-thread Playwright teardown."""
    try:
        bf.request_stop()
        with _crawl_lock:
            thread_alive = _crawl_thread is not None and _crawl_thread.is_alive()
            if thread_alive:
                crawl_state["status"] = "stopping..."
            else:
                crawl_state["running"] = False
                crawl_state["status"] = "stopped"
        return jsonify({"message": "Crawl stop signaled" if thread_alive else "Crawl stopped"})
    except Exception as e:
        return _server_error("api_crawl_stop", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trowler MMFIC Web Dashboard")
    add_verbosity_args(parser)
    cli_args = parser.parse_args()
    configure_logging(verbose=cli_args.verbose, debug=cli_args.debug)

    from app.database import init_db
    init_db()
    init_fingerprint_tables()

    print("=== Checking Tor Connection ===")
    tor_ok = bf.check_tor()
    if bf.USE_TOR and not tor_ok:
        print("[!] WARNING: Tor is enabled but unreachable.")
        print("    Hidden services (.onion) and Tor-enforced crawls will be blocked.\n")

    print("=" * 60)
    print("  Trowler MMFIC Dashboard  |  http://127.0.0.1:5000")
    print("  [SECURITY] NEVER EXPOSE BEYOND LOCALHOST (127.0.0.1).")
    print("             Keep bound to loopback even behind SSH tunnels.")
    if _TOKEN_FROM_ENV:
        print("  [AUTH]     X-API-Token loaded from DASHBOARD_API_TOKEN.")
    else:
        print("  [AUTH]     No DASHBOARD_API_TOKEN set in .env -> Ephemeral token active:")
        print(f"             X-API-Token: {_API_TOKEN}")
        print("             (Set DASHBOARD_API_TOKEN in .env for a persistent token.)")
    print("=" * 60)

    app.run(debug=False, host="127.0.0.1", port=5000)
