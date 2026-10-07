"""
User-targeted crawler — crawls ALL posts by a specific username from a forum.

Used for collecting training data for the author similarity model.
Given a forum URL and a target username, this module:
  1. Opens the user's profile page (direct URL or auto-detected)
  2. Extracts post content directly from the profile page (all belong to user)
  3. Follows thread links to get full post content + user's replies
  4. Follows profile page pagination for more content
  5. Falls back to scanning forum threads if profile page doesn't work

Key design decisions:
  - On a profile page (/u/username), ALL listed posts belong to that user.
    We extract content directly and tag it with the target username.
  - On thread pages linked from the profile, we parse ALL posts and keep
    only those where the username matches (parser or raw HTML extraction).
  - NEVER hop to a different domain.
"""

import time
import random
import re
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup

from app.crawler.browser_fetcher import (
    fetch_page, close_browser, CrawlAbortedByUser, SessionExpired, new_identity
)
from app.parser.router import extract_thread_data
from app.parser.llm_parser import LLMQuotaExceeded, discover_profile_schema
from app.parser.schema_parser import get_schema_for_url, get_profile_schema_for_url
from app.crawler.crawl_utils import content_hash

# Maximum pages to follow from a user's profile
MAX_PROFILE_PAGES = 50

# Maximum identity renewals
MAX_IDENTITY_RENEWALS = 3

# ── Profile URL patterns for different forum types ───────────────

_PROFILE_PATTERNS = [
    "/u/{username}",                          # Lemmy, Tenebris, Dread
    "/user/{username}",                       # Reddit-like
    "/profile/{username}",                    # Generic
    "/members/{username}",                    # XenForo, vBulletin
    "/member/{username}",                     # SMF
    "/@{username}",                           # Mastodon-like
    "/users/{username}",                      # Discourse
]

_USER_POSTS_PATTERNS = [
    "/u/{username}",                          # Lemmy shows posts on profile
    "/user/{username}/posts",                 # Reddit
    "/user/{username}/submitted",             # Reddit old
    "/user/{username}/comments",              # Reddit comments
    "/search?author={username}",              # phpBB
    "/users/{username}/activity",             # Discourse
]

# Regex patterns to extract username from a profile URL
_PROFILE_URL_REGEXES = [
    re.compile(r"/u/([^/?#]+)"),
    re.compile(r"/user/([^/?#]+?)(?:/(?:posts|submitted|comments|activity))?$"),
    re.compile(r"/profile/([^/?#]+)"),
    re.compile(r"/members?/([^/?#]+)"),
    re.compile(r"/users/([^/?#]+)"),
    re.compile(r"/@([^/?#]+)"),
]


def detect_profile_url(url):
    """Detect if a URL is a user profile link and extract the username.
    
    Returns:
        (base_url, username) if detected, (None, None) otherwise.
    """
    parsed = urlparse(url)
    path = parsed.path
    
    for pattern in _PROFILE_URL_REGEXES:
        match = pattern.search(path)
        if match:
            username = match.group(1).strip()
            return _get_base_url(url), username
    
    return None, None


def _get_base_url(url):
    """Helper to extract the base URL (scheme + netloc)."""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def crawl_user(forum_url, target_username=None, max_pages=MAX_PROFILE_PAGES):
    """
    Crawl all posts by a specific user on a forum.
    
    Accepts either:
      - A forum base URL + target_username
      - A direct profile URL (username auto-detected)
    """
    # Auto-detect if the URL is a direct profile link
    direct_profile_url = None
    detected_base, detected_user = detect_profile_url(forum_url)
    
    if detected_base and detected_user:
        print(f"[+] Detected profile URL!")
        print(f"    Username: {detected_user}")
        print(f"    Base URL: {detected_base}")
        direct_profile_url = forum_url
        forum_url = detected_base
        if not target_username:
            target_username = detected_user
    
    if not target_username:
        print("[!] No username provided and could not detect from URL.")
        return []
    
    # (Removed onion scheme fixing)
    
    base_url = _get_base_url(forum_url)
    parsed = urlparse(forum_url)
    allowed_domain = parsed.netloc
    
    print(f"\n{'='*60}")
    print(f"USER-TARGETED CRAWL")
    print(f"  Forum:    {forum_url}")
    print(f"  Username: {target_username}")
    if direct_profile_url:
        print(f"  Profile:  {direct_profile_url}")
    print(f"  Domain:   {allowed_domain} (strict)")
    print(f"  Max pages: {max_pages}")
    print(f"{'='*60}\n")
    
    all_data = []
    seen_hashes = set()
    visited_urls = set()
    identity_renewals = 0
    
    while identity_renewals <= MAX_IDENTITY_RENEWALS:
        try:
            all_data = _run_crawl_phases(
                base_url, target_username, max_pages, visited_urls, seen_hashes, allowed_domain, direct_profile_url, all_data
            )
            
            break
            
        except SessionExpired as e:
            identity_renewals += 1
            if identity_renewals > MAX_IDENTITY_RENEWALS:
                print(f"[STOP] Max identity renewals reached.")
                break
            print(f"\n[!] Session expired. New identity ({identity_renewals}/{MAX_IDENTITY_RENEWALS})...")
            print(f"    Data preserved: {len(all_data)} posts")
            new_identity()
            time.sleep(random.uniform(10, 20))
            
        except CrawlAbortedByUser as e:
            print(f"\n[STOP] {e}")
            break
        except LLMQuotaExceeded as e:
            print(f"\n[STOP] LLM quota exceeded: {e}")
            break
    
    print(f"\n{'='*60}")
    print(f"User-targeted crawl complete: {target_username}")
    print(f"  Total posts collected: {len(all_data)}")
    print(f"  Pages visited:        {len(visited_urls)}")
    print(f"{'='*60}\n")
    
    close_browser()
    return all_data


def _run_crawl_phases(base_url, target_username, max_pages, visited_urls, seen_hashes, allowed_domain, direct_profile_url, all_data):
    """Execute the three phases of user crawling."""
    # Phase 1: Crawl profile page — extract posts directly
    print(f"\n[Phase 1] Crawling {target_username}'s profile...")
    profile_data, actual_profile_url, profile_html = _crawl_user_profile(
        base_url, target_username, max_pages,
        visited_urls, seen_hashes, allowed_domain,
        direct_profile_url=direct_profile_url
    )
    if profile_data:
        all_data.extend(profile_data)
        print(f"[+] Profile crawl: {len(profile_data)} posts")
    
    # Phase 2: Follow thread links to get full content + replies
    print(f"\n[Phase 2] Following thread links for full content...")
    thread_data = _crawl_user_threads(
        base_url, target_username, max_pages,
        visited_urls, seen_hashes, allowed_domain,
        profile_url=actual_profile_url,
        profile_html=profile_html
    )
    if thread_data:
        all_data.extend(thread_data)
        print(f"[+] Thread crawl: {len(thread_data)} more posts")
    
    # Phase 3: Fallback — scan general forum threads
    if len(all_data) < 3:
        print(f"\n[Phase 3] Scanning forum threads for posts by {target_username}...")
        scan_data = _crawl_threads_for_user(
            base_url, target_username, min(max_pages, 20),
            visited_urls, seen_hashes, allowed_domain
        )
        if scan_data:
            all_data.extend(scan_data)
            print(f"[+] Thread scan: {len(scan_data)} more posts")
            
    return all_data


# ═══════════════════════════════════════════════════════════════════
# Phase 1: Extract posts directly from the profile page
# ═══════════════════════════════════════════════════════════════════

def _build_profile_candidate_urls(base_url, username):
    """Build the list of candidate profile URLs from all known patterns."""
    seen = set()
    urls = []
    for pattern in _USER_POSTS_PATTERNS + _PROFILE_PATTERNS:
        url = base_url + pattern.format(username=username)
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def _find_working_profile_url(candidate_urls, visited, allowed_domain, direct_profile_url=None):
    """Try each candidate URL and return the first that looks like a real profile page.
    
    Returns:
        tuple: (profile_url, profile_html) or (None, None)
    """
    for url in candidate_urls:
        if url in visited:
            continue
        if urlparse(url).netloc != allowed_domain:
            continue
        
        print(f"  [TRY] {url}")
        visited.add(url)
        
        html = fetch_page(url)
        if not html:
            continue
        
        # Bug Fix 2: Trust direct profile URLs
        if direct_profile_url and url == direct_profile_url:
            print(f"  [+] Using provided direct profile URL!")
            return url, html
            
        if _is_profile_page(html, urlparse(url).path.split("/")[-1]):
            print(f"  [+] Found profile page!")
            return url, html
    
    return None, None


def _crawl_user_profile(base_url, username, max_pages, visited, seen_hashes, allowed_domain, direct_profile_url=None):
    """Extract posts directly from the user's profile page.
    
    On a profile page like /u/username, EVERY listed post belongs to that user.
    We extract the title + content directly and tag them with the target username.
    No need to rely on the parser's username extraction (which is broken for Tenebris).
    
    Returns:
        tuple: (all_data, actual_profile_url, profile_html)
    """
    all_data = []
    
    # Build candidate URLs
    if direct_profile_url:
        # When a direct URL is given, try it first exclusively
        primary_candidates = [direct_profile_url]
    else:
        # Only try patterns when no direct URL was provided
        primary_candidates = _build_profile_candidate_urls(base_url, username)
    
    # Find the working profile page
    profile_url, profile_html = _find_working_profile_url(
        primary_candidates, visited, allowed_domain, direct_profile_url=direct_profile_url
    )
    
    # Fallback: if direct URL didn't work, try pattern URLs
    if direct_profile_url and not profile_url:
        print(f"  [*] Direct URL didn't work, trying pattern URLs...")
        fallback_candidates = _build_profile_candidate_urls(base_url, username)
        profile_url, profile_html = _find_working_profile_url(
            fallback_candidates, visited, allowed_domain
        )
    
    if not profile_url:
        print(f"  [-] Could not find profile page for {username}")
        # Bug Fix 1: Return tuple
        return all_data, None, None
    
    # Extract posts from profile page (with pagination)
    pages_crawled = 0
    current_html = profile_html
    current_url = profile_url
    
    while pages_crawled < max_pages:
        pages_crawled += 1
        
        # Extract posts from THIS profile page
        posts = _extract_posts_from_profile_html(current_html, current_url, username)
        
        new_posts = _add_new_posts(posts, seen_hashes, all_data)
        
        print(f"  [PROFILE PAGE {pages_crawled}] Extracted {new_posts} new posts")
        
        if new_posts == 0 and pages_crawled > 1:
            break
        
        # Follow pagination
        next_url = _find_next_page(current_html, current_url, allowed_domain)
        if not next_url or next_url in visited:
            break
        if urlparse(next_url).netloc != allowed_domain:
            break
        
        visited.add(next_url)
        print(f"  [NEXT PAGE] {next_url}")
        current_html = fetch_page(next_url)
        if not current_html:
            break
        current_url = next_url
        time.sleep(random.uniform(3, 6))
    
    # Bug Fix 1: Return HTML so Phase 2 doesn't have to re-fetch
    return all_data, profile_url, profile_html


def _extract_posts_from_profile_html(html, url, target_username):
    """Extract all posts from a profile page HTML.
    
    Every post on a profile page belongs to the target user.
    We extract titles + content by looking for common post container patterns
    and parsing the content out of each one.
    """
    soup = BeautifulSoup(html, "html.parser")
    posts = []
    seen_texts = set()
    
    # Check for known profile schema first
    profile_schema = get_profile_schema_for_url(url)
    if profile_schema:
        posts.extend(_extract_posts_from_selectors(
            soup, profile_schema.get("container_selector", ""), 
            url, target_username, seen_texts, schema=profile_schema
        ))
    
    # If no profile schema, try the general thread schema
    if not posts:
        schema = get_schema_for_url(url)
        if schema:
            posts.extend(_extract_posts_from_selectors(
                soup, schema.get("container_selector", ""), 
                url, target_username, seen_texts, schema=schema
            ))
    
    if not posts:
        # Strategy 1: Look for common container blocks
        containers = soup.select("div.message-container, div.post-container, article.post, div.postTop, div.post-inner")
        if not containers:
            # Broader search
            containers = soup.find_all(["div", "article"], class_=re.compile(
                r"message-container|post-container|post-listing|comment-node|postTop|post-inner|postbit", re.I
            ))
        
        for container in containers:
            post = _parse_single_post_container(container, url, target_username)
            if post and post["post"] not in seen_texts:
                seen_texts.add(post["post"])
                posts.append(post)
    
    # Strategy 2: Find all links that look like thread titles
    if not posts:
        posts = _extract_from_title_links(soup, url, target_username, seen_texts)
    
    # Strategy 3: Find all content blocks near the username
    if not posts:
        posts = _extract_from_text_blocks(soup, url, target_username, seen_texts)
        
    # AI Fallback: If all heuristics fail, ask Gemini to find the schema
    if not posts:
        print(f"  [!] Heuristics failed to find posts on profile page. Asking AI for help...")
        try:
            discovered_schema = discover_profile_schema(html, url, target_username)
            if discovered_schema:
                posts.extend(_extract_posts_from_selectors(
                    soup, discovered_schema.get("container_selector", ""), 
                    url, target_username, seen_texts, schema=discovered_schema
                ))
        except Exception as e:
            print(f"  [-] AI profile extraction failed: {e}")
    
    return posts


def _extract_posts_from_selectors(soup, container_sel, url, target_username, seen_texts, schema=None):
    """Helper to extract and deduplicate posts from given CSS selectors."""
    extracted = []
    if not container_sel:
        return extracted
    try:
        for c in soup.select(container_sel):
            post = _parse_single_post_container(c, url, target_username, schema=schema)
            if post and post["post"] not in seen_texts:
                seen_texts.add(post["post"])
                extracted.append(post)
    except Exception:
        pass
    return extracted


def _extract_timestamp(container):
    """Extract a timestamp from a post container."""
    time_el = container.find(["time", "span", "div"], class_=re.compile(r"date|time|ago|timestamp|published", re.I))
    if time_el:
        return time_el.get_text(strip=True)
    return "N/A"


def _strip_noise_elements(clone):
    """Remove noise elements (sidebars, user-info, karma, badges etc.) from a cloned soup."""
    for noise in clone.find_all(True, class_=re.compile(
        r"sidebar|user-info|karma|badge|flair|vote|score|"
        r"rules|staff|advertisement|nav|metadata|date|time|"
        r"community|author|poster", re.I
    )):
        noise.decompose()
    for a in clone.find_all("a", href=re.compile(r"/u/|/user/|/profile/")):
        a.decompose()


def _strip_ui_text(text):
    """Remove common UI button/link text from extracted content."""
    text = re.sub(r'(Log in|Register|Reply|Report|Share|Advertise here)(\s|$)', '', text, flags=re.I)
    return text.strip()


def _extract_post_title(clone):
    """Extract post title from a cloned container."""
    title_el = clone.find(["h1", "h2", "h3", "h4", "h5", "a"], class_=re.compile(r"title|heading", re.I))
    if not title_el:
        title_div = clone.find("div", class_=re.compile(r"title", re.I))
        if title_div:
            title_el = title_div.find("a") or title_div
    if title_el:
        return title_el.get_text(strip=True)
    return ""


def _extract_post_content_body(clone, schema=None):
    """Extract post body content from a cloned container."""
    if schema and schema.get("post_selector"):
        for sel in schema["post_selector"].split(","):
            try:
                el = clone.select_one(sel.strip())
                if el:
                    return el.get_text(strip=True)
            except Exception:
                pass
                
    content_el = clone.find(True, class_=re.compile(
        r"^content$|post-body|message-body|body-text|entry-content|post-content|postMessage|messageText", re.I
    ))
    if content_el:
        return content_el.get_text(strip=True)
    return ""


def _parse_single_post_container(container, url, target_username, schema=None):
    """Parse a single post container from a profile or thread page."""
    
    # Clone to strip noise before text extraction
    clone = BeautifulSoup(str(container), "html.parser")
    _strip_noise_elements(clone)
        
    title = _extract_post_title(clone)
    content = _extract_post_content_body(clone, schema)
    
    # Combine title + content
    full_text = ""
    if title and content:
        full_text = f"{title}\n{content}"
    elif title:
        full_text = title
    elif content:
        full_text = content
    
    if not full_text or len(full_text.split()) < 3:
        return None
    
    # Find timestamp
    timestamp = "N/A"
    if schema and schema.get("timestamp_selector"):
        for sel in schema["timestamp_selector"].split(","):
            try:
                el = container.select_one(sel.strip())
                if el:
                    timestamp = el.get_text(strip=True)
                    break
            except Exception:
                pass
                
    if timestamp == "N/A":
        timestamp = _extract_timestamp(container)
    
    return {
        "username": target_username,
        "post": full_text[:2000],
        "timestamp": timestamp,
        "source": url,
        "is_preview": True,
    }


def _extract_from_title_links(soup, url, target_username, seen_texts):
    """Extract posts from title links on a profile page."""
    posts = []
    
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        text = a.get_text(strip=True)
        
        # Must look like a thread link (added Dread /d/ pattern)
        if not re.search(r"/post/|/thread/|/topic/|/comments/|/d/[\w\-]+/", href):
            continue
        
        # Must have substantive text (a real title)
        if not text or len(text.split()) < 2 or len(text) < 5:
            continue
        
        if text.lower() in ("reply", "comment", "edit", "delete", "report", "share"):
            continue
        
        if text in seen_texts:
            continue
        seen_texts.add(text)
        
        # Look for nearby content
        content = ""
        parent = a.parent
        if parent:
            content_el = parent.find_next_sibling(True, class_=re.compile(r"content|body|text", re.I))
            if content_el:
                content = content_el.get_text(strip=True)
        
        full_text = f"{text}\n{content}" if content else text
        
        # Find timestamp
        timestamp = "N/A"
        for sibling in (a.parent or soup).find_all(True, class_=re.compile(r"date|time|ago", re.I)):
            ts = sibling.get_text(strip=True)
            if ts and len(ts) < 50:
                timestamp = ts
                break
        
        posts.append({
            "username": target_username,
            "post": full_text[:2000],
            "timestamp": timestamp,
            "source": url,
            "is_preview": True,
        })
    
    return posts


def _extract_from_text_blocks(soup, url, target_username, seen_texts):
    """Last resort: find text blocks on the page that look like post content."""
    posts = []
    username_lower = target_username.lower()
    
    for tag in soup.find_all(["script", "style", "noscript", "nav", "footer"]):
        tag.decompose()
    
    for el in soup.find_all(["div", "article", "section", "li"]):
        classes = " ".join(el.get("class", [])).lower()
        
        if not any(c in classes for c in ["post", "message", "content", "comment", "entry", "listing"]):
            continue
        
        text = el.get_text(strip=True)
        if not text or len(text.split()) < 5:
            continue
        if len(text) > 3000:
            continue
        
        if username_lower not in text.lower()[:300]:
            continue
        
        cleaned = _clean_profile_post_text(text, target_username)
        
        if cleaned and len(cleaned.split()) >= 3 and cleaned not in seen_texts:
            seen_texts.add(cleaned)
            posts.append({
                "username": target_username,
                "post": cleaned[:2000],
                "timestamp": "N/A",
                "source": url,
                "is_preview": True,
            })
    
    return posts


def _clean_profile_post_text(text, username):
    """Clean up raw profile page text by removing metadata noise."""
    text = re.sub(
        r'^.*?/u/' + re.escape(username) + r'\s*\d*\s*karma.*?(?:Posts?\s*\d+\s*Comments?\s*\d+\s*)',
        '', text, flags=re.I | re.DOTALL
    )
    text = re.sub(r'^Back\s*', '', text, flags=re.I)
    text = re.sub(
        r'(?:Advertise here|View all|About Community|Join community|Staff members|'
        r'Do you want to leave a comment\?|Log in\s*or\s*Register).*$',
        '', text, flags=re.I | re.DOTALL
    )
    text = re.sub(r'No comments yet\..*$', '', text, flags=re.I | re.DOTALL)
    text = re.sub(r'^\s*\d+\s*(?:hours?|minutes?|days?|weeks?|months?)\s*ago\s*', '', text, flags=re.I)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════
# Phase 2: Follow thread links to get full content + replies
# ═══════════════════════════════════════════════════════════════════

def _fetch_and_combine_thread_posts(thread_url, username):
    """Fetch a thread page and return all posts by the target user.
    
    Combines parser pipeline results with raw HTML extraction.
    """
    thread_html = fetch_page(thread_url)
    if not thread_html:
        return [], thread_html
    
    # Method A: Use the parser pipeline
    parser_posts = []
    try:
        parser_posts = extract_thread_data(thread_html, thread_url)
    except LLMQuotaExceeded:
        raise
    except Exception as e:
        print(f"    [-] Parser error: {e}")
    
    # Filter parser results by username
    matched = _filter_by_username(parser_posts, username)
    
    # Method B: Raw HTML extraction
    raw_posts = _extract_user_posts_from_thread_html(thread_html, thread_url, username)
    
    return matched + raw_posts, thread_html


def _add_new_posts(posts, seen_hashes, all_data):
    """Deduplicate posts by content hash and append new ones to all_data.
    
    Returns:
        int: count of newly added posts
    """
    new_count = 0
    for post in posts:
        ph = content_hash(post)
        if ph not in seen_hashes:
            seen_hashes.add(ph)
            all_data.append(post)
            new_count += 1
    return new_count


def _crawl_user_threads(base_url, username, max_pages, visited, seen_hashes, allowed_domain, profile_url=None, profile_html=None):
    """Visit threads linked from the profile page and extract the user's posts.
    
    Bug Fix 1: Takes profile_url and profile_html as args to avoid re-fetching.
    """
    all_data = []
    
    # If no cached HTML, we must fetch it
    if not profile_html:
        if not profile_url:
            profile_url = base_url + f"/u/{username}"
        profile_html = fetch_page(profile_url)
        if not profile_html:
            return all_data
    
    # Extract thread links from the profile HTML
    thread_links = _extract_thread_links_from_profile(profile_html, profile_url, allowed_domain)
    
    if not thread_links:
        print(f"  [-] No thread links found on profile page")
        return all_data
    
    print(f"  [*] Found {len(thread_links)} thread links to visit")
    
    threads_visited = 0
    for thread_url in thread_links:
        if threads_visited >= max_pages:
            break
        if thread_url in visited:
            continue
        if urlparse(thread_url).netloc != allowed_domain:
            continue
        
        visited.add(thread_url)
        threads_visited += 1
        
        print(f"  [THREAD {threads_visited}] {thread_url}")
        combined, _ = _fetch_and_combine_thread_posts(thread_url, username)
        if not combined:
            continue
        
        new_count = _add_new_posts(combined, seen_hashes, all_data)
        
        if new_count > 0:
            print(f"    [+] {new_count} posts by {username}")
        
        time.sleep(random.uniform(2, 5))
    
    return all_data


def _extract_user_posts_from_thread_html(html, url, target_username):
    """Extract posts by a specific user directly from thread HTML."""
    soup = BeautifulSoup(html, "html.parser")
    posts = []
    seen_texts = set()
    username_lower = target_username.lower()
    
    schema = get_schema_for_url(url)
    
    # Bug Fix 5: Find user links by both href AND text
    user_links = []
    for a in soup.find_all("a", href=True):
        href = a.get("href", "").lower()
        text = a.get_text(strip=True).lower()
        
        if (f"/u/{username_lower}" in href or 
            f"/user/{username_lower}" in href or 
            text == username_lower):
            user_links.append(a)
    
    if not user_links:
        return posts
    
    for user_link in user_links:
        container = _find_post_container(user_link)
        if not container:
            continue
        
        content = _extract_content_from_container(container, schema=schema)
        if not content or len(content.split()) < 3:
            continue
        
        if content in seen_texts:
            continue
        seen_texts.add(content)
        
        timestamp = _extract_timestamp(container)
        
        posts.append({
            "username": target_username,
            "post": content[:2000],
            "timestamp": timestamp,
            "source": url,
            "is_preview": False,
        })
    
    return posts


def _find_post_container(element):
    """Walk up from an element to find its enclosing post/message container."""
    parent = element.parent
    for _ in range(8):
        if parent is None or parent.name in ("body", "html", "[document]"):
            return None
        
        classes = " ".join(parent.get("class", [])).lower()
        tag = parent.name or ""
        
        # Bug Fix 6: Universal container finder (added posttop, post-inner, etc.)
        if any(c in classes for c in [
            "message-container", "post-container", "comment-node",
            "post", "message", "comment", "reply", "entry",
            "posttop", "post-inner", "postbit", "postrow", "comment-body"
        ]):
            return parent
        
        if parent.get("data-post-id") or parent.get("data-comment-id"):
            return parent
        
        if tag == "article":
            return parent
        
        parent = parent.parent
    
    return None


def _extract_content_from_container(container, schema=None):
    """Extract the actual post content from a container, skipping metadata."""
    
    clone = BeautifulSoup(str(container), "html.parser")
    _strip_noise_elements(clone)
        
    if schema and schema.get("post_selector"):
        for sel in schema["post_selector"].split(","):
            try:
                el = clone.select_one(sel.strip())
                if el:
                    text = el.get_text(strip=True)
                    if text and len(text.split()) >= 3:
                        return _strip_ui_text(text)
            except Exception:
                continue

    for selector in ["div.content", "div.post-body", "div.message-body", 
                     "div.body-text", "div.entry-content", "div.post-content",
                     ".content", ".body", ".text"]:
        try:
            el = clone.select_one(selector)
            if el:
                text = el.get_text(strip=True)
                if text and len(text.split()) >= 3:
                    return _strip_ui_text(text)
        except Exception:
            continue
    
    text = clone.get_text(strip=True)
    text = _strip_ui_text(text)
    text = re.sub(r'\d+\s*karma', '', text, flags=re.I)
    text = re.sub(r'Joined\s*\d+\s*\w+\s*ago', '', text, flags=re.I)
    
    return text.strip()


# ═══════════════════════════════════════════════════════════════════
# Phase 3: Fallback — scan general forum threads
# ═══════════════════════════════════════════════════════════════════

def _crawl_threads_for_user(forum_url, username, max_pages, visited, seen_hashes, allowed_domain):
    """Fallback: crawl forum threads and filter for posts by the target user."""
    from app.crawler.link_extractor import extract_thread_links
    
    all_data = []
    
    html = fetch_page(forum_url)
    if not html:
        return all_data
    
    thread_links = extract_thread_links(html, forum_url)
    random.shuffle(thread_links)
    thread_links = thread_links[:max_pages]
    
    print(f"  [*] Scanning {len(thread_links)} threads for posts by {username}...")
    
    for thread_url in thread_links:
        if thread_url in visited:
            continue
        if urlparse(thread_url).netloc != allowed_domain:
            continue
        
        visited.add(thread_url)
        combined, _ = _fetch_and_combine_thread_posts(thread_url, username)
        
        new_count = _add_new_posts(combined, seen_hashes, all_data)
        
        if new_count > 0:
            print(f"  [+] Found {new_count} posts by {username} in {thread_url}")
        
        time.sleep(random.uniform(3, 6))
    
    return all_data


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _resolve_href(href, page_url):
    """Resolve a raw href attribute to an absolute URL."""
    if href.startswith("/"):
        return f"{_get_base_url(page_url)}{href}"
    if not href.startswith("http"):
        return urljoin(page_url, href)
    return href


def _extract_thread_links_from_profile(html, profile_url, allowed_domain):
    """Extract links to threads/posts from a profile page."""
    soup = BeautifulSoup(html, "html.parser")
    links = []
    seen = set()
    
    # Bug Fix 4: Fix thread link regex for Dread + add universal patterns
    thread_patterns = [
        re.compile(r"/post/[a-zA-Z0-9]+"),
        re.compile(r"/thread/[a-zA-Z0-9]+"),
        re.compile(r"/topic/[a-zA-Z0-9]+"),
        re.compile(r"/d/[\w\-]+/[\w\-]+"), # Dread thread link
        re.compile(r"/d/[^/]+/[^/?#]+"),
        re.compile(r"/comments/[a-zA-Z0-9]+"),
        re.compile(r"/viewtopic\.php"),
        re.compile(r"/showthread\.php"),
    ]
    
    for a in soup.find_all("a", href=True):
        href = _resolve_href(a["href"], profile_url)
        
        if urlparse(href).netloc != allowed_domain:
            continue
        
        path = urlparse(href).path
        
        # Skip profile/user/navigation links (make sure not to skip valid threads)
        if re.match(r"^/(u|user|profile|members?|users|search|login|register|settings|c|communities)(/|$)", path, re.I):
            continue
        
        if any(p.search(path) for p in thread_patterns) and href not in seen:
            seen.add(href)
            links.append(href)
    
    return links


def _is_profile_page(html, username):
    """Check if the HTML looks like a user profile page."""
    if not html:
        return False
    
    # Bug Fix 2: Increase lookup area for huge pages
    lower = html[:25000].lower()
    username_lower = username.lower()
    
    if username_lower not in lower:
        return False
    
    # Bug Fix 2: Add Dread-specific and broader indicators
    indicators = [
        f"/u/{username_lower}", f"/user/{username_lower}", f"@{username_lower}",
        "profile", "user activity", "post history", "posts by",
        "submissions", "overview", "karma", "joined",
        "posts", "comments", "cake day", "score"
    ]
    if not any(ind in lower for ind in indicators):
        return False
    
    errors = ["not found", "404", "does not exist", "no such user", "page not found"]
    if any(err in lower for err in errors):
        return False
    
    return True


def _filter_by_username(posts, target_username):
    """Filter posts to ONLY those by the target username (strict)."""
    target_lower = target_username.lower().strip()
    result = []
    
    for post in posts:
        post_username = post.get("username", "unknown").lower().strip()
        
        if post_username == "unknown":
            continue
        
        if post_username == target_lower:
            result.append(post)
            continue
        
        if post_username.replace("-", "_") == target_lower.replace("-", "_"):
            result.append(post)
            continue
    
    return result


def _find_next_page(html, current_url, allowed_domain):
    """Find the next page link."""
    soup = BeautifulSoup(html, "html.parser")
    
    for a in soup.find_all("a", href=True):
        text = a.get_text(strip=True).lower()
        classes = " ".join(a.get("class", [])).lower()
        
        is_next = (
            text in ["next", "next page", ">>", ">", "older", "more"] or
            "next" in classes or
            re.search(r"page[=/\-]\d+", a["href"], re.I) or
            re.search(r"p=\d+|offset=\d+|start=\d+|after=", a["href"], re.I)
        )
        
        if is_next:
            href = _resolve_href(a["href"], current_url)
            if urlparse(href).netloc == allowed_domain:
                return href
    
    return None
