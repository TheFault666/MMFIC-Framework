"""
Data preparation for training the author similarity model.

Converts raw crawl data into labeled training pairs:
  - Positive pair: two sets of posts from the SAME user  (label=1)
  - Negative pair: two sets of posts from DIFFERENT users (label=0)

The key insight: we split each user's posts into two halves.
Same-user halves become positive pairs, cross-user halves become negative pairs.
This lets us train WITHOUT manual labeling.
"""

import json
import os
import random
import sqlite3
import numpy as np
from collections import Counter, defaultdict
from urllib.parse import urlparse
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from .features import extract_user_features


def _extract_domain_from_source(source):
    if source.startswith("http"):
        return urlparse(source).netloc
    return source


def _parse_entry_to_profile(entry, profiles):
    user = entry.get("username", "unknown")
    if user == "unknown":
        return
    post = entry.get("post", "")
    if len(post.split()) < 5:
        return
    source = entry.get("source", "manual")
    domain = _extract_domain_from_source(source)
    profile_key = f"{user}@{domain}"
    profiles[profile_key]["posts"].append(post)
    profiles[profile_key]["timestamps"].append(entry.get("timestamp", "N/A"))
    profiles[profile_key]["sources"].append(source)
    profiles[profile_key]["username"] = user
    profiles[profile_key]["forum"] = domain


def load_crawl_data(filepath="crawl_data.json"):
    """Load crawl data and group posts by username@domain.
    
    Uses a composite key (username + source domain) so that
    'admin' on Forum A and 'admin' on Forum B are treated as
    SEPARATE users during training. The model's job is to later
    detect if they're actually the same person.
    
    Returns:
        dict: {"user@domain": {"posts": [...], "timestamps": [...], "username": ..., "forum": ...}}
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    profiles = defaultdict(lambda: {"posts": [], "timestamps": [], "sources": [], "username": "", "forum": ""})
    
    for entry in data:
        _parse_entry_to_profile(entry, profiles)
    
    filtered = {
        key: profile_data for key, profile_data in profiles.items()
        if len(profile_data["posts"]) >= 3
    }
    
    return filtered


def _split_posts(posts, timestamps=None, sources=None):
    """Split a user's posts into two halves for self-pairing.
    
    Shuffles first to avoid temporal bias (e.g. user changes style over time).
    """
    timestamps = timestamps or ["N/A"] * len(posts)
    sources = sources or ["manual"] * len(posts)
    
    combined = list(zip(posts, timestamps, sources))
    random.shuffle(combined)
    
    mid = len(combined) // 2
    half_a = combined[:mid]
    half_b = combined[mid:]
    
    posts_a = [p for p, t, s in half_a]
    posts_b = [p for p, t, s in half_b]
    ts_a = [t for p, t, s in half_a]
    ts_b = [t for p, t, s in half_b]
    src_a = [s for p, t, s in half_a]
    src_b = [s for p, t, s in half_b]
    
    return (posts_a, ts_a, src_a), (posts_b, ts_b, src_b)


def _extract_board_from_source(src):
    """Extract the first meaningful path segment from a source URL."""
    parts = [p for p in urlparse(src).path.split('/') if p]
    return parts[0] if parts else None


def _fit_lda_and_board_vocab(eligible_users, profiles, lda_topics, seed):
    """Fit LDA topic model and build top-10 board vocabulary from all eligible posts."""
    all_posts = []
    all_boards = []
    for user in eligible_users:
        all_posts.append(" ".join(profiles[user]["posts"]))
        for src in profiles[user]["sources"]:
            if src and src != "manual":
                board = _extract_board_from_source(src)
                if board:
                    all_boards.append(board)
                    
    lda_vectorizer = CountVectorizer(max_features=5000, stop_words="english")
    X_text = lda_vectorizer.fit_transform(all_posts)
    lda_model = LatentDirichletAllocation(n_components=lda_topics, random_state=seed)
    lda_model.fit(X_text)
    
    board_vocab = [b for b, c in Counter(all_boards).most_common(10)]
    
    return lda_model, lda_vectorizer, board_vocab


def _compute_user_split_features(eligible_users, profiles, min_posts_per_half,
                                  lda_model, lda_vectorizer, board_vocab):
    """Pre-compute split features for each eligible user.
    
    Returns a dict of {user: (feat_a, feat_b)} for users whose halves
    meet the minimum post threshold.
    """
    user_halves = {}
    for user in eligible_users:
        posts = profiles[user]["posts"]
        timestamps = profiles[user]["timestamps"]
        sources = profiles[user].get("sources", [])
        
        (posts_a, ts_a, src_a), (posts_b, ts_b, src_b) = _split_posts(posts, timestamps, sources)
        
        if len(posts_a) < min_posts_per_half or len(posts_b) < min_posts_per_half:
            continue
        
        feat_a = extract_user_features(posts_a, ts_a, src_a, lda_model, lda_vectorizer, board_vocab)
        feat_b = extract_user_features(posts_b, ts_b, src_b, lda_model, lda_vectorizer, board_vocab)
        user_halves[user] = (feat_a, feat_b)
    
    return user_halves


def _build_positive_pairs(user_halves):
    """Generate positive pairs (same user, different post halves).
    
    Returns (X_pairs, y_labels, pair_info) for same-person samples.
    """
    X_pairs, y_labels, pair_info = [], [], []
    for user, (feat_a, feat_b) in user_halves.items():
        diff = np.abs(feat_a - feat_b)
        X_pairs.append(diff)
        y_labels.append(1)
        pair_info.append((user, user))
    return X_pairs, y_labels, pair_info


def _build_negative_pairs(user_halves, n_positive, neg_ratio):
    """Generate negative pairs (different users) up to neg_ratio × n_positive.
    
    Returns (X_pairs, y_labels, pair_info) for different-person samples.
    """
    users_with_halves = list(user_halves.keys())
    X_pairs, y_labels, pair_info = [], [], []
    
    target_negatives = n_positive * neg_ratio
    max_attempts = target_negatives * 5
    n_negative = 0
    attempts = 0
    
    while n_negative < target_negatives and attempts < max_attempts:
        attempts += 1
        user_x, user_y = random.sample(users_with_halves, 2)
        
        feat_x = user_halves[user_x][0]
        feat_y = user_halves[user_y][0]
        
        diff = np.abs(feat_x - feat_y)
        X_pairs.append(diff)
        y_labels.append(0)
        pair_info.append((user_x, user_y))
        n_negative += 1
    
    return X_pairs, y_labels, pair_info


def prepare_training_pairs(profiles, neg_ratio=3, min_posts_per_half=1, seed=42, lda_topics=20):
    """Generate labeled training pairs from user profiles.
    
    Strategy (no manual labeling needed):
      1. For each user with enough posts, split posts into two halves
      2. Positive pair: half_A features vs half_B features (same person)
      3. Negative pairs: half_A of user_X vs half_A of user_Y (diff people)
    
    Args:
        profiles: dict from load_crawl_data()
        neg_ratio: how many negative pairs per positive pair
        min_posts_per_half: minimum posts required in each half
        seed: random seed for reproducibility
        lda_topics: number of topics for LDA modeling
    
    Returns:
        X: numpy array of shape (n_pairs, n_features) — absolute feature differences
        y: numpy array of shape (n_pairs,) — 1=same person, 0=different
        pair_info: list of (user_a, user_b) tuples for debugging
        lda_model: fitted LatentDirichletAllocation model
        lda_vectorizer: fitted CountVectorizer
        board_vocab: list of top board paths
    """
    random.seed(seed)
    np.random.seed(seed)
    
    eligible_users = [
        user for user, data in profiles.items()
        if len(data["posts"]) >= min_posts_per_half * 2
    ]
    
    if len(eligible_users) < 2:
        raise ValueError(
            f"Need at least 2 users with {min_posts_per_half * 2}+ posts. "
            f"Got {len(eligible_users)}. Crawl more data first."
        )
    
    lda_model, lda_vectorizer, board_vocab = _fit_lda_and_board_vocab(
        eligible_users, profiles, lda_topics, seed
    )
    
    user_halves = _compute_user_split_features(
        eligible_users, profiles, min_posts_per_half,
        lda_model, lda_vectorizer, board_vocab
    )
    
    pos_X, pos_y, pos_info = _build_positive_pairs(user_halves)
    neg_X, neg_y, neg_info = _build_negative_pairs(user_halves, len(pos_X), neg_ratio)
    
    X_pairs = pos_X + neg_X
    y_labels = pos_y + neg_y
    pair_info = pos_info + neg_info
    
    X = np.array(X_pairs, dtype=np.float64)
    y = np.array(y_labels, dtype=np.int32)
    
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X, y, pair_info, lda_model, lda_vectorizer, board_vocab


def prepare_prediction_pair(posts_a, posts_b, timestamps_a=None, timestamps_b=None, 
                            sources_a=None, sources_b=None, lda_model=None, 
                            lda_vectorizer=None, board_vocab=None):
    """Prepare a single pair for prediction (no label).
    
    Args:
        posts_a: list of posts from user A
        posts_b: list of posts from user B
    
    Returns:
        numpy array of shape (1, n_features) — ready for model.predict()
    """
    feat_a = extract_user_features(posts_a, timestamps_a, sources_a, lda_model, lda_vectorizer, board_vocab)
    feat_b = extract_user_features(posts_b, timestamps_b, sources_b, lda_model, lda_vectorizer, board_vocab)
    
    diff = np.abs(feat_a - feat_b)
    diff = np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
    
    return diff.reshape(1, -1)


def _query_posts_from_db(db_path):
    """Read all qualifying posts from the posts table.
    
    Returns a list of sqlite3.Row objects with username, forum,
    content, timestamp, url columns.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute("""
        SELECT username, forum, content, timestamp, url
        FROM posts
        WHERE username != 'unknown' AND content IS NOT NULL
    """)
    
    rows = c.fetchall()
    conn.close()
    return rows


def load_crawl_data_from_db():
    """Load crawl data from the SQLite database and group by username@forum.
    
    This is an alternative to load_crawl_data() that reads from the posts table
    directly, ensuring ALL crawled data (including user-specific crawls and
    older MMFIC sessions) is available for training.
    
    Returns:
        dict: {"user@domain": {"posts": [...], "timestamps": [...], "username": ..., "forum": ...}}
    """
    from pathlib import Path
    
    db_path = Path(__file__).resolve().parent.parent / "Forum_profiles.db"
    if not db_path.exists():
        return {}
    
    profiles = defaultdict(lambda: {"posts": [], "timestamps": [], "sources": [], "username": "", "forum": ""})
    
    for row in _query_posts_from_db(db_path):
        user = row["username"]
        forum = row["forum"]
        content = row["content"]
        timestamp = row["timestamp"] or "N/A"
        
        if len(content.split()) < 5:
            continue
        
        source = row["url"] or "manual"
        
        profile_key = f"{user}@{forum}"
        profiles[profile_key]["posts"].append(content)
        profiles[profile_key]["timestamps"].append(timestamp)
        profiles[profile_key]["sources"].append(source)
        profiles[profile_key]["username"] = user
        profiles[profile_key]["forum"] = forum
    
    filtered = {
        key: data for key, data in profiles.items()
        if len(data["posts"]) >= 3
    }
    
    return filtered


def _merge_json_into_profiles(filepath, db_profiles):
    """Merge JSON crawl data into an existing profiles dict (deduplicates by content)."""
    if not os.path.exists(filepath):
        return
    json_profiles = load_crawl_data(filepath)
    for key, data in json_profiles.items():
        if key not in db_profiles:
            db_profiles[key] = data
        else:
            existing_posts = set(db_profiles[key]["posts"])
            for post, ts, src in zip(data["posts"], data["timestamps"], data["sources"]):
                if post not in existing_posts:
                    db_profiles[key]["posts"].append(post)
                    db_profiles[key]["timestamps"].append(ts)
                    db_profiles[key]["sources"].append(src)
                    existing_posts.add(post)


def create_training_data_from_file(filepath="crawl_data.json", neg_ratio=3, seed=42, lda_topics=20):
    """Convenience: load data + generate pairs in one call.
    
    Falls back to database if the JSON file doesn't exist or has
    insufficient data (< 2 eligible users).
    
    Returns: X, y, pair_info, lda_model, lda_vectorizer, board_vocab
    """
    if os.path.exists(filepath):
        json_profiles = load_crawl_data(filepath)
        eligible = [u for u, d in json_profiles.items() if len(d["posts"]) >= 2]
        if len(eligible) >= 2:
            return prepare_training_pairs(json_profiles, neg_ratio=neg_ratio, seed=seed, lda_topics=lda_topics)
    
    db_profiles = load_crawl_data_from_db()
    _merge_json_into_profiles(filepath, db_profiles)
    return prepare_training_pairs(db_profiles, neg_ratio=neg_ratio, seed=seed, lda_topics=lda_topics)
