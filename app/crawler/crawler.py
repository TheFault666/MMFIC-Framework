import time
import random
import re
from bs4 import BeautifulSoup
from app.crawler.browser_fetcher import fetch_page, close_browser, CrawlAbortedByUser, SessionExpired, new_identity
from app.crawler.link_extractor import extract_thread_links
from app.crawler.crawl_utils import content_hash
from urllib.parse import urlparse, urljoin
from app.parser.router import extract_thread_data
from app.parser.llm_parser import LLMQuotaExceeded
from app.database import get_crawled_urls

# Track consecutive failures for adaptive sleep
_consecutive_failures = 0

# Maximum number of identity renewals before giving up
MAX_IDENTITY_RENEWALS = 5


def crawl(start_url, max_index_pages=5, max_threads_per_page=10):
    """
    Two-level crawler with automatic session recovery.
    
    If the site detects bot behavior and kills the session, the crawler
    automatically gets a new Tor identity and restarts from where it
    left off (already-crawled pages are remembered).
    
    Args:
        start_url:           Starting forum URL
        max_index_pages:     How many index pages to crawl
        max_threads_per_page: Max threads to follow per index page
    """
    global _consecutive_failures
    _consecutive_failures = 0

    # These persist across identity renewals
    visited_index = set()

    # Load previously visited threads from database to support incremental crawling
    print("[*] Loading visited URLs from database...")
    visited_threads = get_crawled_urls()
    print(f"[+] Loaded {len(visited_threads)} already-crawled URLs.")

    allowed_domain = urlparse(start_url).netloc
    all_data = []
    seen_hashes = set()   # for content deduplication
    identity_renewals = 0

    while identity_renewals <= MAX_IDENTITY_RENEWALS:
        _print_resume_banner(identity_renewals, visited_index, visited_threads, all_data)
        _print_crawl_header(start_url, max_index_pages, max_threads_per_page, identity_renewals)

        try:
            quota_exceeded = _crawl_index_level(
                start_url, max_index_pages, max_threads_per_page,
                allowed_domain, visited_index, visited_threads,
                all_data, seen_hashes,
            )
            if not quota_exceeded:
                # Crawl completed normally — break the retry loop
                break

        except SessionExpired as e:
            identity_renewals += 1
            _handle_session_expired(e, identity_renewals, all_data)
            # Loop continues — visited_index and visited_threads are preserved

        except CrawlAbortedByUser as e:
            print(f"\n{'!'*60}")
            print(f"[STOP] {e}")
            print(f"[STOP] Stopping crawl gracefully. Saving {len(all_data)} posts collected so far.")
            print(f"{'!'*60}\n")
            break

    _print_crawl_summary(all_data, visited_index, visited_threads, seen_hashes, identity_renewals)

    # Clean up sessions
    close_browser()

    return all_data


# ── Level-1 Index Crawl ───────────────────────────────────────────

def _crawl_index_level(
    start_url, max_index_pages, max_threads_per_page,
    allowed_domain, visited_index, visited_threads,
    all_data, seen_hashes,
):
    """Iterate over index pages, following pagination. Returns True if quota exceeded."""
    index_queue = [start_url]

    # ── LEVEL 1: Crawl index pages ────────────────────────────────
    while index_queue and len(visited_index) < max_index_pages:
        url = index_queue.pop(0)

        if url in visited_index:
            continue

        if urlparse(url).netloc != allowed_domain:
            print(f"[-] Skipping out-of-scope index URL: {url}")
            continue

        print(f"\n[INDEX] Crawling: {url}")
        visited_index.add(url)

        html = fetch_page(url)
        if not html:
            _consecutive_failures += 1
            print(f"[-] Could not fetch: {url}")
            continue

        _consecutive_failures = 0  # Reset on success

        quota_exceeded = _crawl_thread_level(
            html, url, max_threads_per_page,
            allowed_domain, visited_threads,
            all_data, seen_hashes,
        )
        if quota_exceeded:
            return True

        # Find more index pages (pagination)
        for link in _find_pagination_links(html, url):
            if link not in visited_index:
                index_queue.append(link)

        _backoff_sleep(6, 12)  # longer sleep between index pages

    return False


# ── Level-2 Thread Crawl ──────────────────────────────────────────

def _crawl_thread_level(
    html, index_url, max_threads_per_page,
    allowed_domain, visited_threads,
    all_data, seen_hashes,
):
    """Follow thread links found on an index page. Returns True if quota exceeded."""
    # ── LEVEL 2: Find threads and crawl each one ──────────────
    thread_links = extract_thread_links(html, index_url)

    # Limit threads per page
    thread_links = thread_links[:max_threads_per_page]

    # Shuffle thread order — bots crawl in DOM order, humans click randomly
    random.shuffle(thread_links)

    print(f"→ Following {len(thread_links)} threads (randomized order)...")

    for thread_url in thread_links:
        quota_exceeded = _process_single_thread(
            thread_url, allowed_domain, visited_threads, all_data, seen_hashes,
        )
        if quota_exceeded:
            return True

    return False


def _process_single_thread(thread_url, allowed_domain, visited_threads, all_data, seen_hashes):
    """Fetch, parse, and deduplicate posts from one thread. Returns True if quota exceeded."""
    global _consecutive_failures

    if thread_url in visited_threads:
        return False

    if urlparse(thread_url).netloc != allowed_domain:
        print(f"  [-] Skipping out-of-scope thread URL: {thread_url}")
        return False

    visited_threads.add(thread_url)

    print(f"\n  [THREAD] Crawling: {thread_url}")

    thread_html = fetch_page(thread_url)
    if not thread_html:
        _consecutive_failures += 1
        print(f"  [-] Could not fetch thread")
        return False

    _consecutive_failures = 0  # Reset on success

    # Extract FULL posts from thread page
    try:
        thread_data = extract_thread_data(thread_html, thread_url)
        _deduplicate_and_collect(thread_data, all_data, seen_hashes)

    except LLMQuotaExceeded as e:
        print(f"\n{'!'*60}")
        print(f"[STOP] LLM API quota/auth error: {e}")
        print(f"[STOP] Stopping crawl. Saving {len(all_data)} posts collected so far.")
        print(f"{'!'*60}\n")
        return True

    except Exception as e:
        print(f"  [-] Error parsing thread {thread_url}: {e}")
        # Continue to next thread
        return False

    _backoff_sleep(4, 8)  # .onion sites need longer gaps
    return False


# ── Post Deduplication ────────────────────────────────────────────

def _deduplicate_and_collect(thread_data, all_data, seen_hashes):
    """Add new, unique posts from thread_data into all_data using content hashing."""
    new_posts = 0
    for post in thread_data:
        # Defensive check: ensure post content exists
        if not post.get("post"):
            continue

        post_hash = content_hash(post)
        if post_hash not in seen_hashes:
            seen_hashes.add(post_hash)
            all_data.append(post)
            new_posts += 1

    real_posts = [
        p for p in thread_data
        if p.get("username") and p["username"] != "unknown"
    ]

    print(f"  -> Thread extracted {len(thread_data)} posts "
          f"({len(real_posts)} with real usernames, "
          f"{new_posts} new after dedup)")


# ── Identity Renewal ──────────────────────────────────────────────

def _handle_session_expired(exc, identity_renewals, all_data):
    """Log the session expiry, acquire a new Tor identity, and wait for the circuit to settle."""
    global _consecutive_failures

    print(f"\n{'!'*60}")
    print(f"[!] {exc}")
    print(f"[!] Getting new identity... (attempt {identity_renewals}/{MAX_IDENTITY_RENEWALS})")
    print(f"[!] Data collected so far: {len(all_data)} posts (will be preserved)")
    print(f"{'!'*60}\n")

    # Get a completely new identity
    new_identity()
    _consecutive_failures = 0

    # Wait before restarting to let the new circuit settle
    wait_time = random.uniform(15, 30)
    print(f"[*] Waiting {wait_time:.0f}s before resuming crawl...")
    time.sleep(wait_time)


# ── Banner / Summary Printers ─────────────────────────────────────

def _print_resume_banner(identity_renewals, visited_index, visited_threads, all_data):
    """Print the resumption banner when restarting after an identity renewal."""
    if identity_renewals == 0:
        return
    print(f"\n{'='*60}")
    print(f"[*] RESUMING CRAWL (Identity renewal #{identity_renewals})")
    print(f"[*] Already crawled: {len(visited_index)} index pages, {len(visited_threads)} threads")
    print(f"[*] Data collected so far: {len(all_data)} posts")
    print(f"{'='*60}\n")


def _print_crawl_header(start_url, max_index_pages, max_threads_per_page, identity_renewals):
    """Print the crawl configuration header."""
    print(f"\n{'='*60}")
    print(f"Starting 2-level crawl: {start_url}")
    print(f"Max index pages: {max_index_pages} | Threads per page: {max_threads_per_page}")
    if identity_renewals > 0:
        print(f"Identity renewals so far: {identity_renewals}/{MAX_IDENTITY_RENEWALS}")
    print(f"{'='*60}\n")


def _print_crawl_summary(all_data, visited_index, visited_threads, seen_hashes, identity_renewals):
    """Print the final crawl statistics."""
    # ── SUMMARY ──────────────────────────────────────────────────
    real = [d for d in all_data if d.get("username", "unknown") != "unknown"]
    print(f"\n{'='*60}")
    print(f"Crawl complete.")
    print(f"  Total items:        {len(all_data)}")
    print(f"  With real username: {len(real)}")
    print(f"  Index pages visited:{len(visited_index)}")
    print(f"  Threads visited:    {len(visited_threads)}")
    print(f"  Identity renewals:  {identity_renewals}")
    print(f"  Duplicates removed: {len(seen_hashes) - len(all_data) if len(seen_hashes) > len(all_data) else 0}")
    print(f"{'='*60}\n")


# ── Adaptive Sleep ────────────────────────────────────────────────

def _backoff_sleep(min_s, max_s):
    """Sleep with adaptive delay that increases on consecutive failures.
    
    On success (0 failures): uses normal range [min_s, max_s]
    On repeated failures: multiplies by up to 3x, capped at 60s
    """
    global _consecutive_failures

    failure_multiplier = min(1 + (_consecutive_failures * 0.5), 3.0)
    adjusted_min = min(min_s * failure_multiplier, 60)
    adjusted_max = min(max_s * failure_multiplier, 60)

    delay = random.uniform(adjusted_min, adjusted_max)

    if failure_multiplier > 1.0:
        print(f"  → Sleeping {delay:.1f}s (adaptive: {failure_multiplier:.1f}x due to {_consecutive_failures} failures)")
    else:
        print(f"  → Sleeping {delay:.1f}s")
    time.sleep(delay)


# ── Pagination ────────────────────────────────────────────────────

def _find_pagination_links(html, base_url):
    """Find 'Next page' links on index pages."""
    soup = BeautifulSoup(html, "html.parser")
    seen = set()
    pagination_links = []
    base_domain = urlparse(base_url).netloc

    for a in soup.find_all("a", href=True):
        text = a.get_text(strip=True).lower()
        href = a["href"]

        is_next = (
            text in ["next", "next page", "»", ">", "older"] or
            re.search(r"page[=\-/]\d+", href, re.I) or
            re.search(r"p=\d+|offset=\d+|start=\d+", href)
        )

        if not is_next:
            continue

        href = urljoin(base_url, href)

        if urlparse(href).netloc == base_domain and href not in seen:
            seen.add(href)
            pagination_links.append(href)

    return pagination_links