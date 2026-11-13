import logging
import random
import re
from bs4 import BeautifulSoup
from app.crawler.browser_fetcher import (
    fetch_page, close_browser, CrawlAbortedByUser, SessionExpired,
    new_identity, ensure_tor_safety, TorOPSECError,
    stop_requested, is_stop_requested,
)
from app.crawler.link_extractor import extract_thread_links
from app.crawler.crawl_utils import content_hash
from app.diagnostics import FailureTracker, log_failure, notify_posts_found
from urllib.parse import urlparse, urljoin
from app.parser.router import extract_thread_data
from app.parser.llm_parser import LLMQuotaExceeded
from app.database import get_crawled_urls

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
    tracker = FailureTracker()
    failure_state = {"consecutive_failures": 0, "tracker": tracker}

    try:
        ensure_tor_safety(start_url)
    except TorOPSECError as e:
        print(f"\n[CRITICAL OPSEC ERROR] {e}")
        return []

    visited_index = set()

    print("[*] Loading visited URLs from database...")
    visited_threads = get_crawled_urls()
    print(f"[+] Loaded {len(visited_threads)} already-crawled URLs.")

    allowed_domain = urlparse(start_url).netloc
    all_data = []
    seen_hashes = set()
    identity_renewals = 0

    try:
        while identity_renewals <= MAX_IDENTITY_RENEWALS:
            if is_stop_requested():
                print("[STOP] Crawl stop requested before next cycle.")
                break

            _print_resume_banner(identity_renewals, visited_index, visited_threads, all_data)
            _print_crawl_header(start_url, max_index_pages, max_threads_per_page, identity_renewals)

            try:
                quota_exceeded = _crawl_index_level(
                    start_url, max_index_pages, max_threads_per_page,
                    allowed_domain, visited_index, visited_threads,
                    all_data, seen_hashes, failure_state,
                )
                if not quota_exceeded:
                    break

            except SessionExpired as e:
                identity_renewals += 1
                if identity_renewals > MAX_IDENTITY_RENEWALS:
                    print(f"[STOP] Maximum identity renewals ({MAX_IDENTITY_RENEWALS}) reached.")
                    break
                renewed = _handle_session_expired(e, identity_renewals, all_data, failure_state)
                if not renewed:
                    print("[STOP] Aborting crawl: New Tor identity could not be established.")
                    break

            except TorOPSECError as e:
                print(f"\n[CRITICAL OPSEC ERROR] {e}")
                break

            except CrawlAbortedByUser as e:
                print(f"\n{'!'*60}")
                print(f"[STOP] {e}")
                print(f"[STOP] Stopping crawl gracefully. Saving {len(all_data)} posts collected so far.")
                print(f"{'!'*60}\n")
                break
    finally:
        close_browser()

    _print_crawl_summary(all_data, visited_index, visited_threads, seen_hashes, identity_renewals, tracker=tracker)

    return all_data


def _crawl_index_level(
    start_url, max_index_pages, max_threads_per_page,
    allowed_domain, visited_index, visited_threads,
    all_data, seen_hashes, failure_state,
):
    """Iterate over index pages, following pagination. Returns True if quota exceeded."""
    index_queue = [start_url]

    while index_queue and len(visited_index) < max_index_pages:
        if is_stop_requested():
            raise CrawlAbortedByUser("Crawl stopped by user.")

        url = index_queue.pop(0)

        if url in visited_index:
            continue

        if urlparse(url).netloc != allowed_domain:
            print(f"[-] Skipping out-of-scope index URL: {url}")
            continue

        print(f"\n[INDEX] Crawling: {url}")
        visited_index.add(url)

        html = fetch_page(url)
        tracker = failure_state.get("tracker")
        if tracker is not None:
            tracker.record("index_pages", total=1, failed=0 if html else 1)
        if not html:
            failure_state["consecutive_failures"] += 1
            log_failure("Failed to fetch index page", level=logging.WARNING, logger_name="crawler", url=url)
            print(f"[-] Could not fetch: {url}")
            continue

        failure_state["consecutive_failures"] = 0

        quota_exceeded = _crawl_thread_level(
            html, url, max_threads_per_page,
            allowed_domain, visited_threads,
            all_data, seen_hashes, failure_state,
        )
        if quota_exceeded:
            return True

        for link in _find_pagination_links(html, url):
            if link not in visited_index:
                index_queue.append(link)

        _backoff_sleep(6, 12, failure_state)

    return False


def _crawl_thread_level(
    html, index_url, max_threads_per_page,
    allowed_domain, visited_threads,
    all_data, seen_hashes, failure_state,
):
    """Follow thread links found on an index page. Returns True if quota exceeded."""
    thread_links = extract_thread_links(html, index_url)

    thread_links = thread_links[:max_threads_per_page]

    # Shuffle thread order — bots crawl in DOM order, humans click randomly
    random.shuffle(thread_links)

    print(f"→ Following {len(thread_links)} threads (randomized order)...")

    for thread_url in thread_links:
        if is_stop_requested():
            raise CrawlAbortedByUser("Crawl stopped by user.")
        quota_exceeded = _process_single_thread(
            thread_url, allowed_domain, visited_threads, all_data, seen_hashes, failure_state,
        )
        if quota_exceeded:
            return True

    return False


def _process_single_thread(thread_url, allowed_domain, visited_threads, all_data, seen_hashes, failure_state):
    """Fetch, parse, and deduplicate posts from one thread. Returns True if quota exceeded."""
    if is_stop_requested():
        raise CrawlAbortedByUser("Crawl stopped by user.")

    if thread_url in visited_threads:
        return False

    if urlparse(thread_url).netloc != allowed_domain:
        print(f"  [-] Skipping out-of-scope thread URL: {thread_url}")
        return False

    visited_threads.add(thread_url)

    print(f"\n  [THREAD] Crawling: {thread_url}")

    thread_html = fetch_page(thread_url)
    tracker = failure_state.get("tracker")
    if tracker is not None:
        tracker.record("threads_fetch", total=1, failed=0 if thread_html else 1)
    if not thread_html:
        failure_state["consecutive_failures"] += 1
        log_failure("Failed to fetch thread page", level=logging.WARNING, logger_name="crawler", url=thread_url)
        print("  [-] Could not fetch thread")
        return False

    failure_state["consecutive_failures"] = 0

    try:
        thread_data = extract_thread_data(thread_html, thread_url)
        _deduplicate_and_collect(thread_data, all_data, seen_hashes)
        if tracker is not None:
            tracker.record("threads_parse", total=1, failed=0)

    except LLMQuotaExceeded as e:
        print(f"\n{'!'*60}")
        print(f"[STOP] LLM API quota/auth error: {e}")
        print(f"[STOP] Stopping crawl. Saving {len(all_data)} posts collected so far.")
        print(f"{'!'*60}\n")
        return True

    except Exception as e:
        if tracker is not None:
            tracker.record("threads_parse", total=1, failed=1)
        log_failure("Error parsing thread", e, level=logging.ERROR, logger_name="crawler", url=thread_url)
        return False

    _backoff_sleep(4, 8, failure_state)
    return False


def _deduplicate_and_collect(thread_data, all_data, seen_hashes):
    """Add new, unique posts from thread_data into all_data using content hashing."""
    new_posts = 0
    for post in thread_data:
        if not post.get("post"):
            continue

        post_hash = content_hash(post)
        if post_hash not in seen_hashes:
            seen_hashes.add(post_hash)
            all_data.append(post)
            new_posts += 1

    if new_posts > 0:
        notify_posts_found(new_posts)

    real_posts = [
        p for p in thread_data
        if p.get("username") and p["username"] != "unknown"
    ]

    print(f"  -> Thread extracted {len(thread_data)} posts "
          f"({len(real_posts)} with real usernames, "
          f"{new_posts} new after dedup)")


def _handle_session_expired(exc, identity_renewals, all_data, failure_state):
    """Log the session expiry, acquire a new Tor identity, and wait for the circuit to settle.
    
    Returns:
        bool: True if identity renewal was successful, False otherwise.
    """
    print(f"\n{'!'*60}")
    print(f"[!] {exc}")
    print(f"[!] Getting new identity... (attempt {identity_renewals}/{MAX_IDENTITY_RENEWALS})")
    print(f"[!] Data collected so far: {len(all_data)} posts (will be preserved)")
    print(f"{'!'*60}\n")

    success = new_identity()
    failure_state["consecutive_failures"] = 0
    if not success:
        print("[CRITICAL] Identity renewal failed: Tor circuit could not be verified.")
        return False

    wait_time = random.uniform(15, 30)
    print(f"[*] Waiting {wait_time:.0f}s before resuming crawl...")
    if stop_requested.wait(timeout=wait_time):
        raise CrawlAbortedByUser("Crawl stopped by user.")
    return True


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


def _print_crawl_summary(all_data, visited_index, visited_threads, seen_hashes, identity_renewals, tracker=None):
    """Print the final crawl statistics."""
    real = [d for d in all_data if d.get("username", "unknown") != "unknown"]
    print(f"\n{'='*60}")
    print("Crawl complete.")
    print(f"  Total items:        {len(all_data)}")
    print(f"  With real username: {len(real)}")
    print(f"  Index pages visited:{len(visited_index)}")
    print(f"  Threads visited:    {len(visited_threads)}")
    print(f"  Identity renewals:  {identity_renewals}")
    print(f"  Duplicates removed: {len(seen_hashes) - len(all_data) if len(seen_hashes) > len(all_data) else 0}")
    if tracker is not None:
        stats = tracker.snapshot()
        for stage, label in (
            ("index_pages", "Index fetch failures"),
            ("threads_fetch", "Thread fetch failures"),
            ("threads_parse", "Thread parse errors"),
        ):
            entry = stats.get(stage)
            if entry and entry["failed"] > 0:
                msg = f"{label}: {entry['failed']}/{entry['total']}"
                print(f"  [WARN] {msg}")
                log_failure(msg, level=logging.WARNING, logger_name="crawler")
    print(f"{'='*60}\n")


def _backoff_sleep(min_s, max_s, failure_state):
    """Sleep with adaptive delay that increases on consecutive failures (interruptible on stop).
    
    On success (0 failures): uses normal range [min_s, max_s]
    On repeated failures: multiplies by up to 3x, capped at 60s
    """
    failures = failure_state["consecutive_failures"]

    failure_multiplier = min(1 + (failures * 0.5), 3.0)
    adjusted_min = min(min_s * failure_multiplier, 60)
    adjusted_max = min(max_s * failure_multiplier, 60)

    delay = random.uniform(adjusted_min, adjusted_max)

    if failure_multiplier > 1.0:
        print(f"  → Sleeping {delay:.1f}s (adaptive: {failure_multiplier:.1f}x due to {failures} failures)")
    else:
        print(f"  → Sleeping {delay:.1f}s")
    if stop_requested.wait(timeout=delay):
        raise CrawlAbortedByUser("Crawl stopped by user.")


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