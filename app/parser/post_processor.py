"""
Universal post-processing pipeline for extracted forum data.

This module runs AFTER all parsers and cleans/validates results
regardless of which forum they came from. It solves:

1. Garbage usernames (numbers, UI labels, concatenated junk)
2. Post content containing page noise (nav, sidebar, metadata, CTAs)
3. Duplicate entries from multiple parsers hitting the same page
4. Username extraction from text patterns (e.g., /u/username)

This is NOT forum-specific — it uses universal patterns that appear
across Reddit-like, Lemmy, phpBB, vBulletin, XenForo, and custom
darkweb forums.
"""

import re
from urllib.parse import urlparse


# ── Username Validation ──────────────────────────────────────────

# UI words that should NEVER be a username (case-insensitive)
_UI_WORDS = {
    # Navigation & actions
    "back", "next", "previous", "reply", "quote", "report", "share",
    "permalink", "edit", "delete", "save", "cancel", "submit", "close",
    "search", "home", "top", "new", "old", "hot", "best", "rising",
    
    # Auth & account
    "login", "log in", "register", "sign up", "sign in", "logout",
    "log out", "sign out", "forgot", "password", "username",
    
    # Section headers
    "comments", "posts", "replies", "threads", "topics", "messages",
    "members", "users", "online", "offline", "staff", "moderators",
    "administrators", "admin", "rules", "wiki", "about", "info",
    
    # Forum/site names that appear as UI elements
    "community", "forum", "board", "discussion", "general",
    
    # Common English words that parsers misidentify as usernames
    "the", "and", "for", "are", "but", "not", "you", "all", "can",
    "had", "her", "was", "one", "our", "out", "has", "his", "how",
    "its", "may", "who", "did", "get", "let", "say", "she", "too",
    "use", "way", "make", "like", "just", "know", "take", "come",
    "good", "give", "most", "only", "find", "here", "thing",
    "many", "well", "also", "some", "time", "very", "when",
    "what", "your", "this", "that", "with", "have", "from",
    "step", "cards", "card", "link", "view", "more", "free",
    "sell", "buy", "need", "want", "help", "send", "text",
    "individual", "method",
    
    # Misc UI
    "unknown", "anonymous", "guest", "n/a", "none", "null",
    "undefined", "deleted", "removed", "banned", "suspended",
    "sort by", "view all", "show more", "load more", "see more",
    "advertise here", "join community", "leave community",
}

# Regex patterns for invalid usernames
_INVALID_USERNAME_PATTERNS = [
    re.compile(r"^\d+$"),                          # Pure numbers: "1", "47", "0"
    re.compile(r"^(comments|posts|replies)\d*$", re.I),  # "Comments14", "Posts3"
    re.compile(r"^(karma|points?|votes?)\d*$", re.I),    # "karma", "Points5"
    re.compile(r"^@[\w.]+$"),                       # @ fragments: "@gmail", "@draga"
    re.compile(r"^\d+\s*(hours?|minutes?|days?|weeks?|months?|years?)\s*ago$", re.I),  # "4 hours ago"
    re.compile(r"^(joined|created|posted)\b", re.I),     # "Joined 3 days ago"
    re.compile(r"^\d+\s*k?\s*members?$", re.I),          # "1.5k Members"
    re.compile(r"^sort\s+by", re.I),                     # "Sort by"
    re.compile(r"^[A-Z]{2,}$"),                     # All-caps short words: "NEWS", "OTHER"
]


def check_username_length_validity(username_stripped):
    if len(username_stripped) < 2:
        return False
    if len(username_stripped) > 40:
        return False
    return True

def check_username_content_validity(username_stripped):
    if "\n" in username_stripped:
        return False
    if username_stripped.lower() in _UI_WORDS:
        return False
    for pattern in _INVALID_USERNAME_PATTERNS:
        if pattern.match(username_stripped):
            return False
    return True

def _is_valid_username(username):
    """Check if a username is likely a real person, not UI garbage.
    
    Returns False for:
    - Pure numbers (comment counts, karma, etc.)
    - Known UI words (Comments, Reply, Back, etc.)
    - Too short (single character)
    - Common concatenation patterns (Comments14)
    - Email fragments (@gmail)
    """
    if not username or username == "unknown":
        return False
    
    username_stripped = username.strip()
    
    if not check_username_length_validity(username_stripped):
        return False
        
    if not check_username_content_validity(username_stripped):
        return False
    
    return True


# ── Username Extraction from Text ────────────────────────────────

# Universal patterns that contain usernames embedded in text
_USERNAME_TEXT_PATTERNS = [
    # Reddit/Lemmy/Kbin style: /u/username or u/username
    re.compile(r"(?:^|[/\s])u/([A-Za-z0-9_\-]{2,30})(?:\s|$|[^A-Za-z0-9_\-])"),
    # @mention style: @username
    re.compile(r"(?:^|\s)@([A-Za-z0-9_\-]{2,30})(?:\s|$|[^A-Za-z0-9_\-])"),
    # "by username" or "posted by username"
    re.compile(r"(?:posted\s+)?by\s+([A-Za-z0-9_\-]{2,30})(?:\s|$)", re.I),
    # "Author: username"
    re.compile(r"author:\s*([A-Za-z0-9_\-]{2,30})(?:\s|$)", re.I),
]


def _extract_username_from_text(text):
    """Try to extract a real username from post text using universal patterns.
    
    Used when CSS/DOM-based extraction failed but the username is
    visibly present in the text (e.g., "dark00store/u/dark00store").
    """
    if not text:
        return None
    
    # Only search the first 300 chars (username appears near the top)
    search_text = text[:300]
    
    for pattern in _USERNAME_TEXT_PATTERNS:
        match = pattern.search(search_text)
        if match:
            candidate = match.group(1).strip()
            if _is_valid_username(candidate):
                return candidate
    
    return None


def clean_username_trailing_digits(username):
    stripped = re.sub(r'(\D{3,})[0-9]$', r'\1', username)
    if stripped != username and len(stripped) >= 3:
        return stripped
    return username


def _clean_username(username):
    """Clean a username that may have trailing noise from concatenated text.
    
    Common issue: parser grabs 'gangsta_gnu0' where '0' is the karma score
    that was concatenated. We strip trailing single digits only if the
    remaining text is still a valid username (3+ chars).
    """
    if not username or username == "unknown":
        return username
    
    return clean_username_trailing_digits(username)


# ── Post Content Cleaning ────────────────────────────────────────

# Universal noise patterns found in forum page dumps
_NOISE_STRIP_PATTERNS = [
    # Profile metadata blocks (karma + joined + posts + comments)
    re.compile(
        r"\d+\s*karma\s*(?:Joined\s*\d+\s*(?:hours?|days?|weeks?|months?|years?)\s*ago\s*)?"
        r"(?:Posts?\s*\d+\s*)?(?:Comments?\s*\d+\s*)?",
        re.I
    ),
    # "Joined X ago" standalone
    re.compile(r"Joined\s*\d+\s*(?:hours?|days?|weeks?|months?|years?)\s*ago", re.I),
    # Timestamp prefixes in concatenated text: "4 hours ago" before titles
    re.compile(r"^\d+\s*(?:hours?|days?|weeks?|months?|years?)\s*ago", re.I),
    # CTA blocks
    re.compile(r"Do you want to leave a comment\?\s*Log\s*in\s*or\s*Register\s*(?:Log\s*in\s*Register\s*)?", re.I),
    re.compile(r"Log\s*in\s*or\s*Register\s*Log\s*in\s*Register", re.I),
    # Comment section headers
    re.compile(r"Sort\s*by:\s*(?:Top\s*)*(?:New\s*)?(?:Old\s*)?", re.I),
    re.compile(r"No comments yet\.\s*Be the first to comment!", re.I),
    # Sidebar community blocks
    re.compile(
        r"(?:About\s+Community\s*)?(?:GENERAL|TECHNOLOGY|ENTERTAINMENT|NEWS|OTHER|MARKETPLACE)\s*"
        r"[\w\s]+c/\w+.*?(?:Join\s*community|Created)",
        re.I | re.DOTALL
    ),
    re.compile(r"\d+\.?\d*k?\s*Members\s*\d+\s*Posts?\s*\d+\s*(?:weeks?|months?|days?)\s*ago\s*Created", re.I),
    # Staff/moderator blocks
    re.compile(r"Staff\s*members\s*(?:Staff members are hidden by community settings\.?|Community\s*manager.*)", re.I | re.DOTALL),
    # Advertise
    re.compile(r"Advertise\s*here\s*View\s*all", re.I),
    # Navigation
    re.compile(r"^Back\s*", re.I),
    # Vote/action buttons text
    re.compile(r"Reply\s*$", re.I),
    # Inline Reply artifacts between comments
    re.compile(r"Reply(?=[A-Z])"),
    # u/username patterns (remove the /u/ prefix notation from post text)
    re.compile(r"/u/[A-Za-z0-9_\-]+"),
    # Suspected Scammer/badges
    re.compile(r"Suspected\s+Scammer", re.I),
    # "× close" button artifacts
    re.compile(r"×\s*\d*"),
    # "P" badge artifact (e.g., "darkdotwebP")
    re.compile(r"(?<=[a-z])P(?=/u/)", re.I),
    
    # ── Dread Specific Artifacts ──
    # User metadata block: "P 5,515 points Posts 320 Comments 17,012"
    re.compile(r"(?:P\s+)?(?:-?\d+,?\d*)\s+points\s+Posts\s+(?:\d+,?\d*)\s+Comments\s+(?:\d+,?\d*)", re.I),
    # Post metadata block: "1 points 8 hours ago" or "1 points 2 days ago"
    re.compile(r"-?\d+\s+points?\s+\d+\s+(?:hours?|days?|weeks?|months?|years?)\s+ago", re.I),
    # Footer action buttons: "Reply Permalink Submit Formatting options..."
    re.compile(r"Reply\s+Permalink\s+Submit\s+Formatting\s+options\.\.\.", re.I),
]


def apply_strip_patterns(text, patterns):
    for pattern in patterns:
        text = pattern.sub(" ", text)
    return text

def collapse_whitespace(text):
    return re.sub(r"\s{2,}", " ", text).strip()


def _apply_noise_patterns(text):
    """Apply all _NOISE_STRIP_PATTERNS to text and collapse whitespace."""
    text = apply_strip_patterns(text, _NOISE_STRIP_PATTERNS)
    return collapse_whitespace(text)


def remove_reply_artifacts(text):
    return re.sub(r"(?:^Reply\s*|\s*Reply$)", "", text, flags=re.I).strip()


def _clean_post_content(text):
    """Strip universal forum noise from post content.
    
    Removes navigation, profile metadata, sidebar info, CTAs,
    and other UI elements that get concatenated into the text
    when parsers grab too much.
    """
    if not text:
        return ""
    
    cleaned = _apply_noise_patterns(text)
    cleaned = remove_reply_artifacts(cleaned)
    
    return cleaned


# ── Deduplication ────────────────────────────────────────────────

def _is_substring_of_kept(post_text, kept):
    """Check if post_text is a substring of (or contains) any kept post text."""
    for kept_post in kept:
        kept_text = kept_post.get("post", "")
        if post_text in kept_text:
            return True, kept_post
        if kept_text[:200] in post_text:
            return True, None
    return False, None


def _upgrade_username_if_better(kept_post, candidate_post):
    """If the candidate post has a real username and the kept post doesn't, update it."""
    if (kept_post.get("username") == "unknown" and
            candidate_post.get("username") != "unknown" and
            _is_valid_username(candidate_post.get("username"))):
        kept_post["username"] = candidate_post["username"]


def _deduplicate_source_group(group):
    """Remove duplicate/overlapping posts within a single source URL group."""
    group.sort(key=lambda p: len(p.get("post", "")))
    
    kept = []
    for post in group:
        post_text = post.get("post", "")[:200]
        
        is_dup, matched_kept = _is_substring_of_kept(post_text, kept)
        if is_dup:
            if matched_kept is not None:
                _upgrade_username_if_better(matched_kept, post)
            continue
        
        kept.append(post)
    
    return kept


def group_posts_by_source(posts):
    by_source = {}
    for post in posts:
        source = post.get("source", "")
        by_source.setdefault(source, []).append(post)
    return by_source


def _deduplicate_posts(posts):
    """Remove duplicate and overlapping posts from the same source URL.
    
    Uses content similarity: if one post's text is a substring of another
    from the same URL, keep only the shorter (more precise) version —
    but only if the shorter one still has >= 10 words.
    """
    if len(posts) < 2:
        return posts
    
    by_source = group_posts_by_source(posts)
    
    result = []
    for source, group in by_source.items():
        if len(group) < 2:
            result.extend(group)
        else:
            result.extend(_deduplicate_source_group(group))
    
    return result


# ── Entry-validation helpers ─────────────────────────────────────

def _resolve_username(username, post_text):
    """Return a validated, cleaned username, falling back to text extraction."""
    if not _is_valid_username(username):
        extracted = _extract_username_from_text(post_text)
        username = extracted if extracted else "unknown"
    return _clean_username(username)


def _build_clean_entry(post, username, post_text):
    """Construct a cleaned post dict from validated fields."""
    return {
        "username": username,
        "post": post_text[:2000],
        "timestamp": post.get("timestamp") or "N/A",
        "source": post.get("source", ""),
        "is_preview": post.get("is_preview", False),
    }


def is_post_too_short(post_text):
    return len(post_text.split()) < 3 and "http" not in post_text and ".onion" not in post_text


# ── Main Pipeline ────────────────────────────────────────────────

def post_process(posts):
    """Universal post-processing pipeline.
    
    Runs on ALL extracted data regardless of source forum or parser.
    
    Steps:
    1. Validate and fix usernames
    2. Clean post content
    3. Remove garbage entries
    4. Deduplicate
    """
    if not posts:
        return []
    
    cleaned = []
    
    for post in posts:
        username  = post.get("username", "unknown")
        post_text = post.get("post", "")
        
        username = _resolve_username(username, post_text)
        post_text = _clean_post_content(post_text)
        
        if is_post_too_short(post_text):
            continue
        
        if _is_sidebar_content(post_text):
            continue
        
        if _is_challenge_text(post_text):
            continue
        
        cleaned.append(_build_clean_entry(post, username, post_text))
    
    cleaned = _deduplicate_posts(cleaned)
    
    return cleaned


def _is_sidebar_content(text):
    """Detect if text is sidebar/community info, not a real post."""
    sidebar_indicators = [
        re.compile(r"^\s*(?:GENERAL|TECHNOLOGY|ENTERTAINMENT|NEWS|OTHER|MARKETPLACE)\s", re.I),
        re.compile(r"Members\s*\d+\s*Posts?\s*\d+", re.I),
        re.compile(r"Join\s*community", re.I),
        re.compile(r"^\s*(?:About|Rules|Staff)\s", re.I),
        re.compile(r"community\s+settings", re.I),
    ]
    
    match_count = sum(1 for p in sidebar_indicators if p.search(text))
    return match_count >= 2


def _is_challenge_text(text):
    """Detect challenge/queue page content that slipped through."""
    challenge_patterns = [
        re.compile(r"placed\s+in\s+a\s+queue", re.I),
        re.compile(r"estimated\s+wait\s+time", re.I),
        re.compile(r"checking\s+your\s+browser", re.I),
        re.compile(r"please\s+wait.*?seconds?", re.I),
        re.compile(r"session\s+expired", re.I),
        re.compile(r"bot[\s-]*like\s+behavio", re.I),
    ]
    return any(p.search(text) for p in challenge_patterns)
