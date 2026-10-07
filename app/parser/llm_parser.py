from bs4 import BeautifulSoup
import json
import hashlib
import os
import re
import time
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv
import litellm

load_dotenv()

def _extract_json_from_text(text):
    """Robustly extract JSON from LLM output, ignoring conversational wrappers."""
    text = text.strip()
    # Try to find a JSON block (object or array)
    match = re.search(r'(\{.*?\}|\[.*?\])', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Fallback to direct parse
    try:
        return json.loads(text.replace("```json", "").replace("```", "").strip())
    except json.JSONDecodeError:
        raise ValueError(f"Could not extract valid JSON from LLM response. Snippet: {text[:100]}")

# ── Custom Exceptions ────────────────────────────────────────────

class LLMQuotaExceeded(Exception):
    """Raised when the LLM API returns a quota/rate limit error.
    
    This should propagate up to the crawler and STOP the entire crawl,
    not silently continue wasting time on pages that will all fail.
    """
    pass


class JSRenderedPage(Exception):
    """Page is JavaScript-rendered (e.g. Discourse, React SPA).
    
    requests.get() returns an empty HTML shell — no content to parse.
    """
    pass


# ── Configuration & Client ───────────────────────────────────────
LLM_MODEL = os.getenv("LLM_MODEL", "gemini/gemini-2.5-flash")
LLM_API_BASE = os.getenv("LLM_API_BASE", None)

# Map GEMINI_API_KEY explicitly if set (LiteLLM uses it for gemini/ models)
if os.getenv("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

# ── File paths (relative to this module, not cwd) ────────────────
_MODULE_DIR = Path(__file__).resolve().parent
KNOWN_FILE     = _MODULE_DIR / "known_schemas.json"
CACHE_FILE     = Path("schema_cache.json")       # project root
HASH_CACHE_FILE = Path("structural_cache.json")  # project root

# ── Schema cache: remembers each domain's structure ──────────────
_schema_cache = {}
_hash_cache   = {}

# Load caches from disk
for f_path, cache_obj in [(CACHE_FILE, _schema_cache), (HASH_CACHE_FILE, _hash_cache)]:
    if f_path.exists():
        with open(f_path) as f:
            cache_obj.update(json.load(f))

# Known schemas always override runtime cache
if KNOWN_FILE.exists():
    with open(KNOWN_FILE) as f:
        _schema_cache.update(json.load(f))


def _save_all_caches():
    with open(CACHE_FILE, "w") as f:
        json.dump(_schema_cache, f, indent=2)
    with open(HASH_CACHE_FILE, "w") as f:
        json.dump(_hash_cache, f, indent=2)


def _get_domain(url):
    return urlparse(url).netloc


# ── Shared HTML stripping ────────────────────────────────────────

def _strip_soup_noise(soup, extra_tags=None):
    """Remove script/style/meta noise tags from soup in-place."""
    noise_tags = ["script", "style", "noscript", "meta", "link"]
    if extra_tags:
        noise_tags.extend(extra_tags)
    for noise in soup(noise_tags):
        noise.decompose()


def get_structural_hash(html):
    """Hash the tag+class structure to identify same-template pages."""
    soup = BeautifulSoup(html, "html.parser")
    _strip_soup_noise(soup)
    structure = []
    for tag in soup.find_all(True)[:400]:
        classes = ".".join(sorted(tag.get("class", [])))
        structure.append(f"{tag.name}({classes})")
    return hashlib.sha256(">".join(structure).encode()).hexdigest()


def _clean_html_for_llm(html, max_chars=8000):
    """Strip scripts/styles/noise before sending HTML to the LLM.
    
    Increased from 6000 to 8000 chars to capture more post containers
    and give the LLM better context for schema discovery.
    """
    soup = BeautifulSoup(html, "html.parser")
    _strip_soup_noise(soup, extra_tags=["nav", "footer", "header", "svg", "img"])
    return str(soup)[:max_chars]


# ── Enhanced text extraction ─────────────────────────────────────

def get_text_from_element(elem):
    text = elem.get_text(strip=True)
    if text and len(text) < 200:
        return text
    
    if elem.name == "a" and elem.get("href"):
        username = _extract_username_from_href(elem["href"])
        if username:
            return username
            
    if elem.name == "time" and elem.get("datetime"):
        return elem["datetime"]
        
    if elem.get("title"):
        return elem["title"]
        
    return None


def _safe_get_text(container, selector):
    """Select an element and get its text, trying multiple comma-separated selectors.
    
    Enhanced to:
    - Try each comma-separated selector individually
    - Extract username from href attribute when text is empty/avatar
    - Extract timestamp from datetime/title attributes
    """
    if not selector:
        return None
    
    selectors = [s.strip() for s in selector.split(",")]
    
    for sel in selectors:
        try:
            elem = container.select_one(sel)
            if elem:
                text = get_text_from_element(elem)
                if text is not None:
                    return text
        except Exception:
            continue
    
    return None


def _extract_username_from_href(href):
    """Extract a username from common URL patterns in forum links.
    
    Patterns:
      /u/JohnDoe, /user/JohnDoe, /profile/JohnDoe,
      /members/JohnDoe, /@JohnDoe
    """
    patterns = [
        r"/u/([^/?#]+)",
        r"/user/([^/?#]+)",
        r"/profile/([^/?#]+)",
        r"/members?/([^/?#]+)",
        r"/@([^/?#]+)",
        r"/author/([^/?#]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, href)
        if match:
            username = match.group(1).strip()
            if username and len(username) < 50:
                return username
    return None


def extract_username_from_patterns(container):
    username_patterns = [
        "a.hover-card-link",
        "a[class*='name']",
        "span[class*='name']",
        "a[class*='user']",
        "span[class*='user']",
        "a[class*='author']",
        "span[class*='author']",
        "a[href*='/u/']",
        "a[href*='/user/']",
        "a[href*='/profile/']",
        "a[href*='/members/']",
    ]
    
    for pattern in username_patterns:
        try:
            elem = container.select_one(pattern)
            if elem:
                text = elem.get_text(strip=True)
                if text and len(text) < 50 and "\n" not in text and text.count(" ") < 3:
                    return text
                if elem.name == "a" and elem.get("href"):
                    username = _extract_username_from_href(elem["href"])
                    if username:
                        return username
        except Exception:
            continue
    return None

def extract_username_from_attributes(container):
    for attr in ["data-username", "data-author", "data-user", "data-name",
                 "data-display-name", "data-poster"]:
        val = container.get(attr)
        if val and len(val) < 50:
            return val
        child = container.find(attrs={attr: True})
        if child:
            return child[attr]
    return None

def _safe_get_username(container, selector):
    """Enhanced username extraction with multiple fallback strategies.
    
    1. Try each CSS selector
    2. Check href-based extraction for <a> tags
    3. Search sibling/parent elements for username patterns
    """
    result = _safe_get_text(container, selector)
    if result and result != "unknown" and len(result) < 50:
        return result
    
    pattern_result = extract_username_from_patterns(container)
    if pattern_result:
        return pattern_result
        
    attr_result = extract_username_from_attributes(container)
    if attr_result:
        return attr_result
    
    return "unknown"


def extract_timestamp_from_time_element(container):
    time_elem = container.find("time")
    if time_elem:
        return (time_elem.get("datetime") or 
                time_elem.get("title") or 
                time_elem.get_text(strip=True) or "N/A")
    return None

def extract_timestamp_from_attributes(container):
    for attr in ["data-timestamp", "data-time", "data-date", "data-created"]:
        val = container.get(attr)
        if val:
            return val
        child = container.find(attrs={attr: True})
        if child:
            return child[attr]
    return None

def extract_timestamp_from_class_patterns(container):
    date_patterns = [
        re.compile(r"time|date|timestamp|posted|when|ago", re.I)
    ]
    for pattern in date_patterns:
        elem = container.find(True, class_=pattern)
        if elem:
            text = elem.get_text(strip=True)
            if text and len(text) < 100:
                return text
            if elem.get("title"):
                return elem["title"]
    return None

def extract_timestamp_from_titles(container):
    for a_tag in container.find_all("a", title=True):
        title = a_tag["title"]
        if re.search(r"\d{4}|\d{1,2}:\d{2}|ago|hour|minute|day|week|month", title, re.I):
            return title
    return None

def _safe_get_timestamp(container, selector):
    """Enhanced timestamp extraction with attribute fallbacks.
    
    Checks: CSS selectors → datetime attr → title attr → data-* attrs
    """
    result = _safe_get_text(container, selector)
    if result and result != "N/A":
        return result
    
    time_result = extract_timestamp_from_time_element(container)
    if time_result is not None:
        return time_result
        
    attr_result = extract_timestamp_from_attributes(container)
    if attr_result is not None:
        return attr_result
        
    class_result = extract_timestamp_from_class_patterns(container)
    if class_result is not None:
        return class_result
        
    title_result = extract_timestamp_from_titles(container)
    if title_result is not None:
        return title_result
    
    return "N/A"


# ── Quota/error keywords that should STOP the crawl ─────────────
_FATAL_ERROR_KEYWORDS = [
    "quota",
    "rate limit",
    "resource exhausted",
    "resourceexhausted",
    "too many requests",
    "429",
    "billing",
    "payment required",
    "access denied",
    "api key not valid",
    "invalid api key",
    "permission denied",
]


def _is_fatal_llm_error(error):
    """Check if an LLM error is fatal (quota/auth) vs transient (network)."""
    error_str = str(error).lower()
    return any(keyword in error_str for keyword in _FATAL_ERROR_KEYWORDS)


def _is_quota_rate_limit(error_msg):
    """Check if a RateLimitError string is specifically a quota exhaustion."""
    return "quota" in error_msg.lower()


def _execute_llm_call(model, prompt):
    """Issue a single LLM completion call and return the stripped text response."""
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        api_base=LLM_API_BASE,
    )
    return response.choices[0].message.content.strip()


def check_llm_exceptions(e, attempt, max_retries):
    """Check if the exception is fatal (quota/auth) and raise LLMQuotaExceeded."""
    error_msg = str(e)
    
    if isinstance(e, litellm.exceptions.AuthenticationError):
        raise LLMQuotaExceeded(f"LLM API Auth error: {e}. Check your API keys in .env")
        
    if isinstance(e, litellm.exceptions.RateLimitError):
        if _is_quota_rate_limit(error_msg) or attempt == max_retries - 1:
            raise LLMQuotaExceeded(f"LLM API quota error: {error_msg}")
            
    if _is_fatal_llm_error(e):
        raise LLMQuotaExceeded(f"LLM API fatal error: {error_msg}. Stopping all LLM calls to avoid wasting resources.")


def _llm_call_with_retry(prompt, model=None, max_retries=2):
    """Call LLM via LiteLLM with retry logic. Raises LLMQuotaExceeded on fatal errors."""
    target_model = os.getenv("LLM_MODEL") or model or "gemini/gemini-2.5-flash"
    
    for attempt in range(max_retries):
        try:
            return _execute_llm_call(target_model, prompt)
        except Exception as e:
            print(f"  [-] LLM call attempt {attempt+1} failed: {e}")
            check_llm_exceptions(e, attempt, max_retries)
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
    
    return None


# ── JS-rendered page detection ───────────────────────────────────

def _has_spa_marker(html_lower):
    """Check if the raw HTML contains a known SPA framework marker string."""
    spa_markers = [
        'discourse',             # Discourse forum
        'ember',                 # Ember.js (used by Discourse)
        '__NEXT_DATA__',         # Next.js
        'react-root',            # React
        'ng-app',                # Angular
        'data-discourse',        # Discourse data attrs
        'preloaded.currentuser', # Discourse preload
    ]
    return any(marker in html_lower for marker in spa_markers)


def _visible_text_length(soup):
    """Return the character length of visible text after stripping noise tags."""
    _strip_soup_noise(soup, extra_tags=["svg"])
    return len(soup.get_text(strip=True))


def _is_js_rendered(html):
    """Detect if a page is a JS-rendered SPA with no server-side content.
    
    Signs of JS-rendered pages:
      - Very little visible text after stripping scripts
      - Contains framework markers (Discourse, React, Angular, Ember)
      - Has a lot of <script> tags relative to content tags
    """
    html_lower = html[:10000].lower()

    if _has_spa_marker(html_lower):
        soup = BeautifulSoup(html, "html.parser")
        if _visible_text_length(soup) < 500:
            return True
    
    soup2 = BeautifulSoup(html, "html.parser")
    script_count = len(soup2.find_all("script"))
    _strip_soup_noise(soup2, extra_tags=["svg"])
    visible_text = soup2.get_text(strip=True)
    
    if script_count > 5 and len(visible_text) < 300:
        return True
    
    return False


# ── Schema validation ────────────────────────────────────────────

def _is_real_value(val):
    """Return True if val is a non-null, non-empty string selector."""
    if val is None:
        return False
    if isinstance(val, str) and val.strip().lower() in ("null", "none", "", "n/a"):
        return False
    return True


def _is_valid_schema(schema):
    """Check if a schema has at least container + one data selector.
    
    An all-null schema is useless and should NOT be cached.
    """
    if not schema or not isinstance(schema, dict):
        return False
    
    if not _is_real_value(schema.get("container_selector")):
        return False
    
    has_data = (
        _is_real_value(schema.get("username_selector")) or
        _is_real_value(schema.get("post_selector"))
    )
    
    return has_data


# ── Schema Discovery (called ONCE per unknown domain) ────────────

def _parse_schema_json(raw, domain):
    """Parse a raw LLM response string into a schema dict, or return None."""
    try:
        schema = _extract_json_from_text(raw)
        print(f"[+] Schema discovered for {domain}: {schema}")
        return schema
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[-] Schema parse failed: {e}")
        return None


def _discover_schema(html, url):
    """Ask Gemini to find CSS selectors ONCE per domain.
    
    Enhanced prompt with darkweb-specific examples and multi-selector output.
    
    Raises LLMQuotaExceeded if API quota is exhausted.
    """
    domain = _get_domain(url)
    html_sample = _clean_html_for_llm(html, max_chars=8000)

    prompt = f"""You are a CSS selector expert analyzing HTML from a forum or discussion website (possibly a darkweb .onion forum).

Your job is to find the CSS selectors that identify these elements in EACH POST:
1. **Post author / username** — the person who wrote the post
2. **Post content / body text** — the actual message content
3. **Post timestamp / date** — when it was posted
4. **Repeating container** — the parent element that wraps each individual post

IMPORTANT RULES:
- For each field, provide MULTIPLE fallback selectors separated by commas (e.g., "a.hover-card-link, span.name, a.name")
- Usernames are often inside: <a class="hover-card-link">, <span class="name">, <a href="/u/...">, <a class="author">, elements with class containing "user", "author", "name", "poster"
- Timestamps are often inside: <time datetime="...">, <span class="date">, <div class="date">, elements with class containing "time", "date", "ago"
- Post content is often inside: <div class="content">, <div class="body">, <div class="message">, elements with class containing "post-body", "message-body", "content"
- Containers often have: class containing "post", "message", "comment", "reply", or data attributes like data-post-id

Look at the HTML carefully. Find the REPEATING structure that represents individual posts.

Return ONLY a valid JSON object (no explanation, no markdown fences):
{{
  "username_selector": "a.hover-card-link, span.name",
  "post_selector": "div.content, div.body",
  "timestamp_selector": "div.date, time, span.timestamp",
  "container_selector": "div.post"
}}

If you cannot find a clear selector for a field, use null.

RAW HTML:
{html_sample}
"""
    raw = _llm_call_with_retry(prompt)
    if not raw:
        return None
    
    return _parse_schema_json(raw, domain)


# ── Extraction using discovered schema ───────────────────────────

def _extract_post_from_container(container, username_sel, post_sel, time_sel, url):
    """Extract a single post dict from a container element, or return None."""
    username  = _safe_get_username(container, username_sel)
    post_text = _safe_get_text(container, post_sel) or ""
    timestamp = _safe_get_timestamp(container, time_sel)

    if len(post_text.split()) < 5:
        return None

    return {
        "username":  username,
        "post":      post_text[:2000],
        "timestamp": timestamp,
        "source":    url
    }


def _extract_posts_from_containers(containers, schema, url):
    username_sel  = schema.get("username_selector")
    post_sel      = schema.get("post_selector")
    time_sel      = schema.get("timestamp_selector")
    data = []
    for container in containers:
        try:
            post = _extract_post_from_container(container, username_sel, post_sel, time_sel, url)
            if post:
                data.append(post)
        except Exception:
            continue
    return data


def _extract_with_schema(html, url, schema):
    """Use discovered selectors to extract posts.
    
    Enhanced with:
    - Multi-selector username extraction with href fallback
    - Sibling/parent username search when container extraction fails
    - Attribute-based timestamp extraction
    """
    soup = BeautifulSoup(html, "html.parser")
    
    container_sel = schema.get("container_selector")

    if not container_sel:
        return []

    containers = soup.select(container_sel)
    print(f"[+] Found {len(containers)} post containers via schema")

    data = _extract_posts_from_containers(containers, schema, url)

    unknown_count = sum(1 for d in data if d["username"] == "unknown")
    if data and unknown_count == len(data):
        print("[!] All usernames are 'unknown'. Trying sibling/parent search...")
        data = _try_sibling_username_search(soup, data, container_sel)

    return data


def _try_sibling_username_search(soup, data, container_sel):
    """When containers have posts but no usernames, look for usernames
    in sibling or parent elements adjacent to the container.
    
    Common pattern: username is in a separate <div> above the post container,
    not inside it.
    """
    containers = soup.select(container_sel)
    
    for i, container in enumerate(containers):
        if i >= len(data):
            break
        if data[i]["username"] != "unknown":
            continue
        
        prev = container.find_previous_sibling()
        if prev:
            username = _search_element_for_username(prev)
            if username:
                data[i]["username"] = username
                continue
        
        parent = container.parent
        if parent:
            username = _search_element_for_username(parent, depth=1)
            if username:
                data[i]["username"] = username
    
    return data


def _search_element_for_username(element, depth=0):
    """Search an element for username-like content.
    
    Looks for short text in links, spans with name/user/author classes.
    """
    if not element:
        return None
    
    patterns = [
        "a[class*='user']", "a[class*='name']", "a[class*='author']",
        "span[class*='user']", "span[class*='name']", "span[class*='author']",
        "a.hover-card-link",
    ]
    
    for pattern in patterns:
        try:
            found = element.select_one(pattern)
            if found:
                text = found.get_text(strip=True)
                if text and len(text) < 50 and "\n" not in text:
                    return text
        except Exception:
            continue
    
    for attr in ["data-username", "data-author", "data-user", "data-name"]:
        found = element.find(attrs={attr: True})
        if found:
            return found[attr]
    
    if depth == 0:
        for a in element.find_all("a", href=True):
            username = _extract_username_from_href(a["href"])
            if username:
                return username
    
    return None


# ── Main entry point ─────────────────────────────────────────────

def _cache_new_schema(domain, schema, html):
    """Store a newly discovered schema in both domain and structural caches."""
    struct_hash = get_structural_hash(html)
    _schema_cache[domain] = schema
    _hash_cache[struct_hash] = schema
    _save_all_caches()


def parse_with_llm(html, url):
    """Entry point with Domain + Structural caching.
    
    This should ONLY be called when manual parsing has already failed.
    
    Raises:
        LLMQuotaExceeded: if API quota/auth error occurs — caller should stop crawl.
    """
    domain = _get_domain(url)
    
    if _is_js_rendered(html):
        print(f"[!] SKIPPING {domain} — JS-rendered SPA (Discourse/React/etc). "
              f"requests.get() cannot fetch content from these sites.")
        return []
    
    schema = None

    if domain in _schema_cache:
        print(f"[+] Domain schema match: {domain}")
        schema = _schema_cache[domain]
    
    if not schema:
        struct_hash = get_structural_hash(html)
        if struct_hash in _hash_cache:
            print(f"[+] Structural hash match: {struct_hash[:16]}...")
            schema = _hash_cache[struct_hash]
            _schema_cache[domain] = schema
    
    if not schema:
        print(f"[!] New site/template. Calling Gemini for schema discovery...")
        schema = _discover_schema(html, url)
        
        if schema and _is_valid_schema(schema):
            _cache_new_schema(domain, schema, html)
        else:
            print(f"[!] LLM returned null/empty schema for {domain} — "
                  f"skipping this page but will retry on others")
            return []

    data = _extract_with_schema(html, url, schema)

    if len(data) < 2:
        print("[!] Schema extraction weak, trying LLM text extraction...")
        data = _llm_text_extract(html, url)

    return data


def clean_soup_for_llm_text_extract(soup):
    for noise in soup(["script", "style", "noscript", "meta", "link", "nav", "footer", "header"]):
        noise.decompose()

def parse_llm_text_extract_response(raw, url):
    try:
        items = _extract_json_from_text(raw)
        return [
            {"username": i.get("username", "unknown"),
             "post": i.get("post", "")[:2000],
             "timestamp": i.get("timestamp", "N/A"),
             "source": url}
            for i in items if i.get("post") and len(i.get("post", "").split()) >= 5
        ]
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[-] LLM text extract parse failed: {e}")
        return []

def _llm_text_extract(html, url):
    """Last resort: ask Gemini to pull structured data from raw text.
    
    Raises LLMQuotaExceeded if API quota is exhausted.
    """
    soup = BeautifulSoup(html, "html.parser")
    clean_soup_for_llm_text_extract(soup)
    text = soup.get_text(separator="\n")[:5000]

    prompt = f"""Extract every username and their associated post/message from this forum page text.

IMPORTANT: Look carefully for usernames. They are typically short (1-30 characters), appear before or above a post, and may be formatted differently from the post content. Common patterns include:
- Names appearing on their own line before a message
- Names followed by timestamps
- Names that repeat throughout the page as different people post

Return ONLY a JSON list. No markdown. No explanation:
[
  {{"username": "actual_username", "post": "their message text", "timestamp": "date if visible"}}
]

If you find no clear usernames, return [].

TEXT:
{text}
"""

    raw = _llm_call_with_retry(prompt)
    if not raw:
        return []

    return parse_llm_text_extract_response(raw, url)


def _save_profile_schema(domain, schema):
    """Persist a newly discovered profile schema to the schema cache."""
    if domain not in _schema_cache:
        _schema_cache[domain] = {}
    _schema_cache[domain]["profile_schema"] = schema
    _save_all_caches()


def discover_profile_schema(html, url, target_username):
    """Ask Gemini to find CSS selectors for posts on a profile page.
    
    Caches the result in schema_cache.json under "profile_schema" so it
    only runs once per domain.
    """
    domain = _get_domain(url)
    
    if domain in _schema_cache and "profile_schema" in _schema_cache[domain]:
        return _schema_cache[domain]["profile_schema"]
        
    html_sample = _clean_html_for_llm(html, max_chars=8000)

    prompt = f"""You are a CSS selector expert analyzing HTML from a user's profile page on a forum or discussion website.

The goal is to find the CSS selectors that identify posts made by the user '{target_username}'.
On profile pages, often EVERY post listed belongs to the user, so you just need to find:
1. **Repeating container** — the parent element that wraps each individual post
2. **Post content / body text** — the actual message content
3. **Post timestamp / date** — when it was posted

IMPORTANT RULES:
- For each field, provide MULTIPLE fallback selectors separated by commas (e.g., "div.content, div.body")
- Look at the HTML carefully. Find the REPEATING structure that represents individual posts on a profile page.

Return ONLY a valid JSON object (no explanation, no markdown fences):
{{
  "container_selector": "div.post, article",
  "post_selector": "div.content, div.body",
  "timestamp_selector": "div.date, time, span.timestamp"
}}

If you cannot find a clear selector for a field, use null.

RAW HTML:
{html_sample}
"""

    raw = _llm_call_with_retry(prompt)
    if not raw:
        return None
        
    try:
        schema = _extract_json_from_text(raw)
        
        if not schema or not isinstance(schema, dict) or not schema.get("container_selector"):
            return None
            
        print(f"[+] Profile schema discovered for {domain}: {schema}")
        _save_profile_schema(domain, schema)
        return schema
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[-] Profile schema extraction parse failed: {e}")
        return None