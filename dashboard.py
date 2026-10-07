"""
dashboard.py - Flask backend for the MMFIC web dashboard.

Serves as the API backend for a single-page application that provides
identity-comparison analytics, crawl management, and ML model training.
"""

import sys
import io
import json
import re
import sqlite3
import threading
import itertools
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
from flask import Flask, render_template, jsonify, request
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Project root & path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.database import DB_PATH, save_posts, get_user_posts
from app.parser.date_utils import normalize_timestamps
from app.crawler.crawler import crawl
from app.crawler.user_crawler import crawl_user
from app.crawler.twitter_adapter import is_twitter_url, crawl_twitter_user, extract_username_from_url
from Model.fingerprint_store import (
    get_fingerprint, store_fingerprint,
    get_all_fingerprints, find_similar_users, init_fingerprint_tables,
)
from Model.model import AuthorSimilarityModel
from Model.data_prep import create_training_data_from_file
from Model.features import get_feature_names
import app.crawler.browser_fetcher as bf
from app.crawler.crawl_utils import append_to_json_file

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------------------------
# Global state dicts for background task tracking
# ---------------------------------------------------------------------------
crawl_state = {
    "running": False, "status": "idle", "result": None, "error": None,
    "posts_found": 0, "started_at": None, "log": [],
}
train_state = {"running": False, "status": "idle", "error": None, "metrics": None}
_crawl_thread = None  # Reference to the active crawl thread

# ---------------------------------------------------------------------------
# Thread-safe lazy-loaded ML model cache
# ---------------------------------------------------------------------------
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
        conn = _db()
        result = conn.execute("SELECT MAX(crawled_at) FROM posts").fetchone()[0]
        conn.close()
        return result
    except Exception:
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


# ===================================================================
# Page Route
# ===================================================================
@app.route("/")
def index():
    """Serve the main SPA page."""
    return render_template("index.html")


# ===================================================================
# Stats & Users
# ===================================================================
def _query_db_stats():
    """Query platforms, profiles, posts count and max crawled_at from DB."""
    conn = _db()
    platforms = conn.execute("SELECT COUNT(DISTINCT forum) FROM posts").fetchone()[0]
    profiles = conn.execute("SELECT COUNT(DISTINCT username || forum) FROM posts").fetchone()[0]
    posts = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
    max_crawled = conn.execute("SELECT MAX(crawled_at) FROM posts").fetchone()[0]
    conn.close()
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
        return jsonify({"error": str(e)}), 500


def _query_all_users_grouped():
    """Return all users grouped by forum as a dict."""
    conn = _db()
    rows = conn.execute(
        "SELECT username, forum, COUNT(*) AS post_count "
        "FROM posts GROUP BY username, forum ORDER BY post_count DESC"
    ).fetchall()
    conn.close()
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
        return jsonify({"error": str(e)}), 500


def _query_users_for_forum(forum):
    """Return users for a specific forum as a list of dicts."""
    conn = _db()
    rows = conn.execute(
        "SELECT username, forum, COUNT(*) AS post_count "
        "FROM posts WHERE forum = ? GROUP BY username, forum "
        "ORDER BY post_count DESC", (forum,)
    ).fetchall()
    conn.close()
    return [{"username": r["username"], "post_count": r["post_count"]} for r in rows]


@app.route("/api/users/<path:forum>")
def api_users_forum(forum):
    """Return users for a specific platform."""
    try:
        return jsonify(_query_users_for_forum(forum))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/user/<username>/<path:forum>/posts")
def api_user_posts(username, forum):
    """Return up to 50 posts for a given user on a given forum."""
    try:
        posts = get_user_posts(username, forum)
        return jsonify(posts[:50])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===================================================================
# Identity Comparison
# ===================================================================
def _ensure_fingerprint_exists(username, forum):
    """Build and cache a fingerprint for a user if one is not already stored."""
    from Model.fingerprint_store import update_fingerprint
    if get_fingerprint(username, forum) is None:
        update_fingerprint(username, forum)
    return get_fingerprint(username, forum)


def _predict_pair_score(fp_a_arr, fp_b_arr, match_threshold, mismatch_threshold, cosine_tie_breaker):
    """Compute similarity score and verdict for two fingerprint arrays."""
    # Compute diff vector (absolute difference, reshaped to 2D)
    diff = np.abs(fp_a_arr - fp_b_arr).reshape(1, -1)
    diff = np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)

    model = _get_model()
    if model is not None and model.is_trained:
        return model.predict_pair_twd(diff, match_threshold, mismatch_threshold, cosine_tie_breaker)

    # Predict with model or cosine fallback
    score = float(cosine_similarity(fp_a_arr.reshape(1, -1), fp_b_arr.reshape(1, -1))[0][0])
    verdict = "UNCERTAIN"
    if score >= match_threshold:
        verdict = "MATCH"
    elif score <= mismatch_threshold:
        verdict = "MISMATCH"
    return score, verdict, {"reason": "Fallback cosine similarity"}


def _query_post_count(username, forum):
    """Return the post count for a single user+forum pair."""
    conn = _db()
    cnt = conn.execute(
        "SELECT COUNT(*) FROM posts WHERE username=? AND forum=?",
        (username, forum)
    ).fetchone()[0]
    conn.close()
    return cnt


@app.route("/api/compare/pair", methods=["POST"])
def api_compare_pair():
    """Compare two users and return a similarity score + verdict."""
    try:
        data = request.get_json(force=True)
        user_a, forum_a = data["user_a"], data["forum_a"]
        user_b, forum_b = data["user_b"], data["forum_b"]

        # Build fingerprints if missing
        fp_a = _ensure_fingerprint_exists(user_a, forum_a)
        fp_b = _ensure_fingerprint_exists(user_b, forum_b)

        # Get threshold params (or defaults)
        match_threshold = data.get("match_threshold", 0.8)
        mismatch_threshold = data.get("mismatch_threshold", 0.3)
        cosine_tie_breaker = data.get("cosine_tie_breaker", 0.85)

        score, verdict, meta = _predict_pair_score(
            fp_a["fingerprint"], fp_b["fingerprint"],
            match_threshold, mismatch_threshold, cosine_tie_breaker
        )

        return jsonify({
            "score": score,
            "verdict": verdict,
            "user_a": {"username": user_a, "forum": forum_a, "post_count": _query_post_count(user_a, forum_a)},
            "user_b": {"username": user_b, "forum": forum_b, "post_count": _query_post_count(user_b, forum_b)},
            "meta": meta,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _ensure_all_fingerprints_exist():
    """Build fingerprints for every user in the DB that is missing one."""
    from Model.fingerprint_store import update_fingerprint
    conn = _db()
    rows = conn.execute("SELECT DISTINCT username, forum FROM posts").fetchall()
    conn.close()
    for r in rows:
        if get_fingerprint(r["username"], r["forum"]) is None:
            update_fingerprint(r["username"], r["forum"])


@app.route("/api/compare/all", methods=["POST"])
def api_compare_all():
    """Find the most similar users to a given user across all platforms."""
    try:
        data = request.get_json(force=True)
        username = data["username"]
        forum = data["forum"]
        top_n = data.get("top_n", 10)

        # Ensure fingerprint exists for the query user
        if get_fingerprint(username, forum) is None:
            store_fingerprint(username, forum)

        # Ensure all users have fingerprints
        _ensure_all_fingerprints_exist()

        matches = find_similar_users(username, forum, top_n=top_n)
        return jsonify(matches)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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


def _build_graph_edges_with_model(vectors, node_lookup, model, threshold):
    """Compute edges using the trained model in one batch."""
    indices = list(itertools.combinations(range(len(vectors)), 2))
    diffs = [np.abs(vectors[i] - vectors[j]) for i, j in indices]
    X_batch = np.nan_to_num(np.array(diffs), nan=0.0, posinf=0.0, neginf=0.0)
    # Predict in one giant batch instead of N^2 separate calls
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=".*X does not have valid feature names.*")
        X_scaled = model.scaler.transform(X_batch)
        probs = model.classifier.predict_proba(X_scaled)[:, 1]
    return [
        {"from": node_lookup[i], "to": node_lookup[j],
         "value": float(probs[idx]), "title": f"Similarity: {float(probs[idx]):.3f}"}
        for idx, (i, j) in enumerate(indices)
        if float(probs[idx]) >= threshold
    ]


def _build_graph_edges_cosine(vectors, node_lookup, threshold):
    """Compute edges using cosine similarity as fallback."""
    edges = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            score = float(cosine_similarity(vectors[i].reshape(1, -1), vectors[j].reshape(1, -1))[0][0])
            if score >= threshold:
                edges.append({
                    "from": node_lookup[i],
                    "to": node_lookup[j],
                    "value": score,
                    "title": f"Similarity: {score:.3f}"
                })
    return edges


@app.route("/api/compare/graph")
def api_compare_graph():
    """Build a similarity graph of all fingerprinted users."""
    try:
        threshold = float(request.args.get("threshold", 0.3))
        all_fps = get_all_fingerprints()
        if not all_fps:
            return jsonify({"nodes": [], "edges": []})

        # all_fps is a list of dicts: [{"username": ..., "forum": ..., "fingerprint": ..., "post_count": ...}, ...]
        nodes, vectors, node_lookup = _build_graph_nodes(all_fps)

        edges = []
        if len(vectors) > 1:
            # Batch compute differences to save massive CPU/Memory overhead
            model = _get_model()
            if model is not None and model.is_trained:
                edges = _build_graph_edges_with_model(vectors, node_lookup, model, threshold)
            else:
                edges = _build_graph_edges_cosine(vectors, node_lookup, threshold)

        return jsonify({"nodes": nodes, "edges": edges})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===================================================================
# Model
# ===================================================================
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
            "training_in_progress": train_state["running"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/model/importances")
def api_model_importances():
    """Return the top 30 feature importances from the trained classifier."""
    try:
        model = _get_model()
        if model is None or not model.is_trained:
            return jsonify({"error": "Model is not trained"}), 400
        importances = model.classifier.feature_importances_
        names = get_feature_names()
        paired = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)
        top = [{"feature": n, "importance": float(v)} for n, v in paired[:30]]
        return jsonify(top)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _train_model_task(lda_topics_val):
    """Background thread: prepare data, train, save, and rebuild fingerprints."""
    global _cached_model
    try:
        train_state.update({"running": True, "status": "preparing data", "error": None, "metrics": None})
        X, y, pair_info, lda_model, lda_vec, board_vocab = create_training_data_from_file(lda_topics=lda_topics_val)

        train_state["status"] = "training model"
        new_model = AuthorSimilarityModel()
        feature_names = get_feature_names(lda_topics=lda_topics_val, board_vocab=board_vocab)
        metrics = new_model.train(X, y, feature_names, lda_model, lda_vec, board_vocab)

        train_state["status"] = "saving model"
        new_model.save()

        with _model_lock:
            _cached_model = new_model

        train_state["status"] = "rebuilding fingerprints"
        from Model.fingerprint_store import rebuild_all_fingerprints
        rebuild_all_fingerprints()

        train_state.update({"running": False, "status": "complete", "metrics": metrics})
    except Exception as exc:
        train_state.update({"running": False, "status": "failed", "error": str(exc)})


@app.route("/api/model/train", methods=["POST"])
def api_model_train():
    """Start model training in a background thread."""
    try:
        if train_state["running"]:
            return jsonify({"error": "Training already in progress"}), 409
        data = request.get_json(force=True, silent=True) or {}
        lda_topics = int(data.get("lda_topics", 20))
        t = threading.Thread(target=_train_model_task, args=(lda_topics,), daemon=True)
        t.start()
        return jsonify({"message": "Training started"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/model/train/status")
def api_model_train_status():
    """Return the current training state."""
    return jsonify(train_state)


# ===================================================================
# Crawling
# ===================================================================
def _log_crawl(msg):
    """Add a message to the crawl log (keeps last 5 entries)."""
    crawl_state["log"].append(msg)
    if len(crawl_state["log"]) > 5:
        crawl_state["log"] = crawl_state["log"][-5:]


class TeeWriter:
    """Writes to both real stdout and a capture buffer."""
    def __init__(self, real, capture):
        self.real = real
        self.capture = capture
    def write(self, text):
        self.real.write(text)
        self.capture.write(text)
        # Parse progress from crawler output
        for line in text.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            if 'posts' in line.lower() or 'thread' in line.lower() or 'phase' in line.lower():
                _log_crawl(line)

            # Check for dashboard captcha event magic string
            if "[DASHBOARD_CAPTCHA_PENDING]" in line:
                crawl_state["status"] = "captcha_pending"

            # Count posts from summary lines like "[+] Profile crawl: 5 posts"
            if 'posts' in line.lower():
                m = re.search(r'(\d+)\s+(?:posts?|new)', line, re.I)
                if m:
                    crawl_state["posts_found"] = crawl_state.get("posts_found", 0) + int(m.group(1))
    def flush(self):
        self.real.flush()
    def isatty(self):
        return False


def _save_crawl_results(results):
    """Normalize, save to DB, and append to JSON."""
    results = normalize_timestamps(results)
    save_posts(results)
    out_path = str(PROJECT_ROOT / "crawl_data.json")
    append_to_json_file(results, out_path)
    return results


def _run_crawl_with_progress(crawl_func, *args, **kwargs):
    """Run a crawl function while capturing stdout to update progress."""
    bf.DASHBOARD_MODE = True

    # Clean up any stale browser state from previous sessions
    try:
        bf.ensure_clean_browser()
    except Exception:
        pass

    # Redirect stdout to capture crawler print messages
    real_stdout = sys.stdout
    capture = io.StringIO()
    sys.stdout = TeeWriter(real_stdout, capture)
    try:
        results = crawl_func(*args, **kwargs)
    finally:
        sys.stdout = real_stdout
        bf.DASHBOARD_MODE = False

    return results


def _reset_crawl_state(status):
    """Reset crawl_state for a fresh crawl run."""
    crawl_state.update({
        "running": True, "status": status, "result": None,
        "error": None, "posts_found": 0, "log": [],
        "started_at": datetime.now().isoformat(),
    })


def _run_background_crawl(crawl_func, *args, **kwargs):
    """Shared background thread for all crawl types."""
    global _crawl_thread
    try:
        results = _run_crawl_with_progress(crawl_func, *args, **kwargs)

        if not crawl_state["running"]: # Stop was requested
            crawl_state.update({"status": "stopped", "result": {"posts_found": len(results)}})
            return
            
        if results:
            results = normalize_timestamps(results)
            save_posts(results)
            append_to_json_file(results, str(PROJECT_ROOT / "crawl_data.json"))
            
        crawl_state.update({
            "running": False,
            "status": "complete",
            "result": {"posts_found": len(results) if results else 0},
        })
    except Exception as exc:
        import traceback
        traceback.print_exc()
        crawl_state.update({"running": False, "status": "failed", "error": str(exc)})
    finally:
        _crawl_thread = None


def _start_crawl_thread(status_msg, crawl_func, *args, **kwargs):
    """Helper to initialize state and start the background crawl thread."""
    global _crawl_thread
    _reset_crawl_state(status_msg)
    _crawl_thread = threading.Thread(
        target=_run_background_crawl, args=(crawl_func,) + args, kwargs=kwargs, daemon=True
    )
    _crawl_thread.start()


@app.route("/api/crawl/general", methods=["POST"])
def api_crawl_general():
    """Start a general forum crawl in a background thread."""
    try:
        if crawl_state["running"]:
            return jsonify({"error": "Crawl already in progress"}), 409

        data = request.get_json(force=True)
        url = data["url"]
        max_pages = data.get("max_pages", 5)
        max_threads = data.get("max_threads", 10)

        # Twitter/X detection: route to dedicated adapter
        if is_twitter_url(url):
            tw_username = extract_username_from_url(url)
            if not tw_username:
                return jsonify({"error": "Could not extract username from Twitter URL."}), 400
            _start_crawl_thread(f"crawling twitter @{tw_username}", crawl_twitter_user, url, username=tw_username, max_tweets=max_pages * 50)
            return jsonify({"message": f"Twitter crawl started for @{tw_username}"})

        _start_crawl_thread("crawling", crawl, url, max_pages, max_threads)
        return jsonify({"message": "Crawl started"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/crawl/user", methods=["POST"])
def api_crawl_user():
    """Start a user-specific crawl in a background thread."""
    try:
        if crawl_state["running"]:
            return jsonify({"error": "Crawl already in progress"}), 409

        data = request.get_json(force=True)
        url = data["url"]
        username = data["username"]
        max_pages = data.get("max_pages", 5)

        # Twitter/X detection: route to dedicated adapter
        if is_twitter_url(url):
            tw_username = extract_username_from_url(url) or username
            _start_crawl_thread(f"crawling twitter @{tw_username}", crawl_twitter_user, url, username=tw_username, max_tweets=max_pages * 50)
            return jsonify({"message": "User crawl started"})

        _start_crawl_thread("crawling user", crawl_user, url, username, max_pages)
        return jsonify({"message": "User crawl started"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _compute_elapsed_seconds(started_at_iso):
    """Return elapsed seconds since started_at_iso, or None on parse failure."""
    try:
        started = datetime.fromisoformat(started_at_iso)
        return int((datetime.now() - started).total_seconds())
    except Exception:
        return None


@app.route("/api/crawl/status")
def api_crawl_status():
    """Return the current crawl state."""
    try:
        state = dict(crawl_state)
        # Include elapsed time if crawl is running
        if state["running"] and state.get("started_at"):
            elapsed = _compute_elapsed_seconds(state["started_at"])
            if elapsed is not None:
                state["elapsed_seconds"] = elapsed
        return jsonify(state)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/crawl/captcha_action", methods=["POST"])
def api_crawl_captcha_action():
    """Handle HITL CAPTCHA action from dashboard."""
    try:
        data = request.get_json(force=True)
        action = data.get("action", "resume")

        # Unblock the crawler thread
        bf.dashboard_captcha_action = action
        bf.dashboard_captcha_event.set()

        # The stdout parser should naturally revert it to crawling when it outputs again,
        # but we can set it immediately for snappy UI
        crawl_state["status"] = "crawling"

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/crawl/stop", methods=["POST"])
def api_crawl_stop():
    """Signal the crawl to stop and kill the browser to abort network requests."""
    try:
        crawl_state.update({"running": False, "status": "stopping..."})
        # Force-close the browser to abort the crawl thread
        try:
            bf.close_browser()
        except Exception:
            pass
        crawl_state["status"] = "stopped"
        return jsonify({"message": "Crawl stopped"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===================================================================
# Entry Point
# ===================================================================
if __name__ == "__main__":
    from app.database import init_db
    init_db()
    init_fingerprint_tables()

    print("=" * 50)
    print("  Trowler MMFIC Dashboard")
    print("  http://127.0.0.1:5000")
    print("=" * 50)

    app.run(debug=False, host="127.0.0.1", port=5000)
