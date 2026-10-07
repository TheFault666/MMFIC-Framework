"""
Twitter/X Adapter for MMFIC Crawler.

Extracts tweets from a user's public timeline using Playwright DOM scraping
with Twitter's stable data-testid selectors. Outputs posts in the standard
MMFIC format: {"username": ..., "content": ..., "timestamp": ..., "url": ..., "forum": "x.com"}

This module reuses the existing browser_fetcher infrastructure (browser init,
stealth, HITL login) so that Twitter's mandatory login wall is handled
transparently via the same challenge detection pipeline.
"""

import random
from urllib.parse import urlparse


# ── URL Detection ────────────────────────────────────────────────

_TWITTER_HOSTS = {
    'twitter.com', 'www.twitter.com',
    'x.com', 'www.x.com',
    'mobile.twitter.com', 'mobile.x.com',
}


def is_twitter_url(url):
    """Check if a URL points to Twitter/X."""
    try:
        parsed = urlparse(url if '://' in url else f'https://{url}')
        return parsed.netloc.lower() in _TWITTER_HOSTS
    except Exception:
        return False


def extract_username_from_url(url):
    """Extract the target username from a Twitter/X URL.
    
    Handles:
        https://x.com/username
        https://x.com/username/status/12345
        https://twitter.com/username/with_replies
    
    Returns None for non-profile URLs like /search, /explore, /home.
    """
    try:
        parsed = urlparse(url if '://' in url else f'https://{url}')
        path = parsed.path.strip('/')
        if not path:
            return None

        first_segment = path.split('/')[0].lower()

        # These are Twitter app routes, not usernames
        reserved = {
            'search', 'explore', 'home', 'i', 'settings',
            'messages', 'notifications', 'compose', 'hashtag',
            'login', 'signup', 'tos', 'privacy',
        }
        if first_segment in reserved:
            return None

        return first_segment
    except Exception:
        return None


# ── Tweet Field Extractors ───────────────────────────────────────

def _extract_tweet_text(article):
    """Return the tweet's text content, or None if absent."""
    text_el = article.query_selector('[data-testid="tweetText"]')
    if not text_el:
        return None
    return text_el.inner_text().strip() or None


def _extract_tweet_timestamp(article):
    """Return the ISO timestamp string from a tweet article, or 'N/A'."""
    # Twitter renders timestamps as <time datetime="2024-01-15T10:30:00.000Z">
    time_el = article.query_selector('time[datetime]')
    if time_el:
        return time_el.get_attribute('datetime') or "N/A"
    return "N/A"


def _extract_tweet_url(article):
    """Return the absolute permalink of a tweet article, or empty string."""
    # Links to individual tweets contain /status/ in the href
    link_el = article.query_selector('a[href*="/status/"]')
    if not link_el:
        return ""
    href = link_el.get_attribute('href') or ""
    if href.startswith('/'):
        return f"https://x.com{href}"
    return href


# ── Retweet Detection ────────────────────────────────────────────

def _is_retweet(article, target_username, tweet_url):
    """Return True if the article is a retweet rather than an original post.
    
    Check if this tweet was authored by the target user.
    Retweets show a different author in the User-Name testid.
    """
    # Method 1: Check for "retweeted" social context banner
    social_ctx = article.query_selector('[data-testid="socialContext"]')
    if social_ctx:
        ctx_text = social_ctx.inner_text().lower()
        if 'reposted' in ctx_text or 'retweeted' in ctx_text:
            return True

    # Method 2: Check author link in the tweet header
    if tweet_url:
        # The permalink contains the author's username: /author/status/123
        url_parts = tweet_url.split('/')
        for i, part in enumerate(url_parts):
            if part == 'status' and i > 0:
                tweet_author = url_parts[i - 1].lower()
                return tweet_author != target_username

    return False


# ── Tweet Extraction ─────────────────────────────────────────────

def _extract_tweets_from_page(page, target_username):
    """Extract all visible tweet data from the current page state.
    
    Uses Twitter's data-testid attributes which are stable across UI updates
    because they are part of Twitter's own internal test infrastructure.
    
    Args:
        page: Playwright page object
        target_username: lowercase username to filter (skip retweets)
    
    Returns:
        list of dicts in MMFIC post format
    """
    tweets = []

    # Twitter wraps each tweet in an <article> with data-testid="tweet"
    articles = page.query_selector_all('article[data-testid="tweet"]')

    for article in articles:
        try:
            # ── Extract tweet text ───────────────────────────
            text = _extract_tweet_text(article)
            if not text:
                continue

            # ── Extract timestamp ────────────────────────────
            timestamp = _extract_tweet_timestamp(article)

            # ── Extract tweet permalink ──────────────────────
            tweet_url = _extract_tweet_url(article)

            # ── Detect retweets (skip them) ──────────────────
            if _is_retweet(article, target_username, tweet_url):
                continue

            tweets.append({
                "username": target_username,
                "content": text,
                "timestamp": timestamp,
                "url": tweet_url,
                "forum": "x.com",
            })

        except Exception:
            # Skip malformed tweet elements silently
            continue

    return tweets


# ── Main Crawl Function ──────────────────────────────────────────

def crawl_twitter_user(url, username=None, max_tweets=200):
    """Crawl a Twitter/X user's timeline and extract their tweets.
    
    Uses the existing Playwright browser from browser_fetcher. If a login
    wall is encountered, the hybrid challenge detection will trigger the
    Human-in-the-Loop visible browser window automatically.
    
    Args:
        url: Twitter profile URL (e.g., "https://x.com/username")
        username: Optional override for the target username.
                  If None, extracted from the URL.
        max_tweets: Maximum number of tweets to collect (default 200).
    
    Returns:
        list of dicts in MMFIC standard format:
        [{"username", "content", "timestamp", "url", "forum": "x.com"}, ...]
    """
    # Import browser infrastructure (lazy import to avoid circular deps)
    from app.crawler import browser_fetcher as bf

    if bf._page is None:
        bf.init_browser()

    page = bf._page

    # Resolve target username
    target = (username or extract_username_from_url(url))
    if not target:
        print("[-] Could not determine target username from URL.")
        print("    Provide a profile URL like: https://x.com/username")
        return []

    target = target.lower()
    profile_url = f"https://x.com/{target}"

    print(f"[*] Twitter Adapter: Crawling @{target}")
    print(f"[*] Navigating to {profile_url}...")

    if not _navigate_to_profile(page, profile_url):
        return []

    # ── Handle login wall via existing HITL system ───────────
    page = _resolve_login_wall(bf, page, profile_url, target)

    if _is_account_unavailable(page, target):
        return []

    return _scroll_and_collect_tweets(bf, page, target, max_tweets)


# ── Navigation Helpers ───────────────────────────────────────────

def _navigate_to_profile(page, profile_url):
    """Navigate the Playwright page to the given profile URL. Returns False on failure."""
    try:
        page.goto(profile_url, wait_until="domcontentloaded", timeout=180000)
    except Exception as e:
        print(f"[-] Navigation failed: {e}")
        return False

    # Let the SPA hydrate and tweets load
    page.wait_for_timeout(random.randint(3000, 5000))
    return True


def _resolve_login_wall(bf, page, profile_url, target):
    """Trigger HITL login to ensure the user is authenticated to crawl Twitter/X."""
    html = page.content()
    current_url = page.url

    score, reasons = bf._score_challenge_signals(html, current_url=current_url, original_url=profile_url)
    if bf._is_challenge_page(score, html=html, url=current_url):
        print("\n" + "="*60)
        print("[!] X/Twitter severely restricts unauthenticated scraping.")
        print("[!] Please log in to a throwaway Twitter account now.")
        print("[!] Once logged in, your session will be saved for future crawls.")
        print("="*60 + "\n")
        
        bf.solve_captcha_interactive(profile_url)

        # After manual login, re-navigate to the target profile
        print(f"[*] Resuming: navigating back to @{target}...")
        page = bf._page  # Page reference may have changed after HITL
        page.goto(profile_url, wait_until="domcontentloaded", timeout=180000)
        page.wait_for_timeout(random.randint(3000, 5000))

    return page


def _is_account_unavailable(page, target):
    """Return True if Twitter reports the account does not exist or is suspended.
    
    Check for "This account doesn't exist" or suspended notices.
    """
    body_text = page.inner_text('body')
    unavailable = any(phrase in body_text for phrase in [
        "This account doesn't exist",
        "Account suspended",
        "Hmm...this page doesn't exist",
    ])
    if unavailable:
        print(f"[-] Twitter reports: account @{target} does not exist or is suspended.")
    return unavailable


# ── Scroll & Collect Loop ────────────────────────────────────────

def _is_crawl_stopped_by_user(bf):
    """Return True if the dashboard has flagged the crawl as stopped."""
    # Check if crawl was stopped by user (dashboard integration)
    if not bf.DASHBOARD_MODE:
        return False
    try:
        from dashboard import crawl_state
        return not crawl_state.get("running", True)
    except ImportError:
        return False


def _scroll_page(page):
    """Scroll the page down by a human-like random distance and wait."""
    scroll_distance = random.randint(800, 1500)
    page.evaluate(f"window.scrollBy(0, {scroll_distance})")
    page.wait_for_timeout(random.randint(1500, 3000))


def _scroll_and_collect_tweets(bf, page, target, max_tweets):
    """Scroll through a timeline, deduplicating and collecting tweets up to max_tweets."""
    all_posts = []
    seen_texts = set()
    consecutive_empty = 0
    max_empty_scrolls = 5
    scroll_count = 0
    max_scrolls = max_tweets  # Upper bound safety limit

    print(f"[*] Scrolling through @{target}'s timeline (target: {max_tweets} tweets)...")

    while len(all_posts) < max_tweets and scroll_count < max_scrolls:
        # Extract tweets from current viewport
        batch = _extract_tweets_from_page(page, target)

        new_count = 0
        for post in batch:
            # Deduplicate by content text
            text_key = post["content"][:100]  # First 100 chars as key
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                all_posts.append(post)
                new_count += 1

        if new_count > 0:
            consecutive_empty = 0
            print(f"  [+] {len(all_posts)} tweets collected...")
        else:
            consecutive_empty += 1

        # Stop if we've hit the bottom (no new tweets after several scrolls)
        if consecutive_empty >= max_empty_scrolls:
            print(f"  [*] No new tweets after {max_empty_scrolls} scrolls. Reached end of timeline.")
            break

        if _is_crawl_stopped_by_user(bf):
            print("[!] Crawl stopped by user.")
            break

        # Scroll down with human-like behavior
        _scroll_page(page)

        # Occasional longer pause to mimic reading behavior
        scroll_count += 1
        if scroll_count % 10 == 0:
            pause = random.uniform(2.0, 4.0)
            print(f"  [*] Brief pause ({pause:.1f}s) to avoid rate limiting...")
            page.wait_for_timeout(int(pause * 1000))

    print(f"[+] Twitter crawl complete: {len(all_posts)} tweets from @{target}")
    return all_posts
