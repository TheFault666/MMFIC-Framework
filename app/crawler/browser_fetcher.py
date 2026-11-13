"""
Playwright-based fetcher module.

Replaces the 'requests' based fetcher to overcome advanced anti-bot
protections (DDoS-Guard, EndGame, Cloudflare) that rely on TLS fingerprinting
and Javascript execution checks.

Features:
  - Runs a real headless Chromium browser
  - Routes traffic through Tor SOCKS5 proxy
  - Rotating user-agents per session
  - Human-like behavior simulation (scroll, mouse, viewport)
  - Human-in-the-loop: Temporarily opens a visible browser window when a 
    CAPTCHA or wait queue is detected, letting you solve it manually before 
    seamlessly resuming the scrape with the authenticated session.
"""

import logging
import os
import queue
import random
import re
import threading
import time
from urllib.parse import urlparse

import requests
from playwright.sync_api import sync_playwright

from app.diagnostics import log_failure, notify_captcha_pending
try:
    from playwright_stealth import Stealth
    _stealth_adapter = Stealth()

    def stealth_sync(page):
        """Apply playwright-stealth 2.x evasions to a synchronous page."""
        _stealth_adapter.apply_stealth_sync(page)
except ImportError:
    try:
        from playwright_stealth import stealth_sync
    except ImportError:
        from playwright_stealth import stealth as stealth_sync


USE_TOR = True

TOR_PROXY = "socks5://127.0.0.1:9150"

# socks5h resolves DNS through Tor instead of locally
_TOR_PROXY_REQUESTS = "socks5h://127.0.0.1:9150"
_requests_proxies = {
    "http": _TOR_PROXY_REQUESTS,
    "https": _TOR_PROXY_REQUESTS
} if USE_TOR else None

CHALLENGE_INDICATORS = [
    "captcha", "ddos-guard", "cloudflare", "challenge",
    "enable javascript", "please wait", "checking your browser",
    "endgame", "just a moment",
    "placed in a queue", "estimated wait time",
    "session expired", "bot-like behavio", "new session",
]

_LOGIN_URL_PATTERNS = [
    "/login", "/signin", "/sign-in", "/sign_in",
    "/log-in", "/log_in", "/auth", "/oauth",
    "/register", "/signup", "/sign-up", "/sign_up",
    "/accounts/login", "/flow/login", "/session/new",
    "/member.php?action=login", "action=login",
    "/index.php?action=login",
]


def _score_challenge_signals(html, current_url=None, original_url=None):
    """Score a page using multiple signals to detect login walls / challenges.
    
    Returns (score, reasons) where score >= 40 means challenge detected.
    
    Signal weights:
      +40  Password input field found AND page has minimal content
      +30  URL was redirected to a login/auth path
      +25  Password field found (even with some content around it)
      +15  Very low content density (< 200 words of visible text)
      +10  Static keyword match from CHALLENGE_INDICATORS
      +5   OAuth/SSO iframes detected (clearnet only)
    """
    if not html or len(html) < 100:
        return 100, ["empty_or_tiny_page"]
    
    lower_html = html.lower()
    reasons = []
    score = 0
    
    has_password_field = ('type="password"' in lower_html or 
                         "type='password'" in lower_html or
                         'type=password' in lower_html)
    
    stripped = re.sub(r'<(script|style|noscript)[^>]*>.*?</\1>', '', lower_html, flags=re.DOTALL)
    stripped = re.sub(r'<[^>]+>', ' ', stripped)
    stripped = re.sub(r'\s+', ' ', stripped).strip()
    word_count = len(stripped.split())
    
    low_content = word_count < 200
    very_low_content = word_count < 80
    
    url_redirected_to_login = False
    if current_url and original_url:
        curr_path = current_url.split('?')[0].split('#')[0].rstrip('/')
        orig_path = original_url.split('?')[0].split('#')[0].rstrip('/')
        
        if curr_path != orig_path:
            curr_lower = current_url.lower()
            for pattern in _LOGIN_URL_PATTERNS:
                if pattern in curr_lower:
                    url_redirected_to_login = True
                    break
    
    keyword_matched = False
    first_2k = lower_html[:2000]
    for indicator in CHALLENGE_INDICATORS:
        if indicator in first_2k:
            keyword_matched = True
            break
    
    is_onion = original_url and '.onion' in original_url
    has_oauth = False
    if not is_onion:
        oauth_markers = [
            'accounts.google.com', 'appleid.apple.com',
            'facebook.com/login', 'facebook.com/v', 
            'oauth', 'openid', 'sso',
        ]
        has_oauth = any(marker in lower_html for marker in oauth_markers)
    
    if has_password_field and very_low_content:
        score += 40
        reasons.append("password_field+very_low_content")
    elif has_password_field and low_content:
        score += 35
        reasons.append("password_field+low_content")
    elif has_password_field:
        score += 25
        reasons.append("password_field_present")
    
    if url_redirected_to_login:
        score += 30
        reasons.append("url_redirected_to_login")
    
    if low_content and not has_password_field:
        score += 15
        reasons.append("low_content_density")
    
    if keyword_matched:
        score += 30
        reasons.append("keyword_match_boosted")
    
    if has_oauth:
        score += 5
        reasons.append("oauth_sso_detected")
    
    return score, reasons


def _is_challenge_page(score, html=None, url=None):
    """Detect if the response is a CAPTCHA, login wall, or challenge page.
    
    Uses hybrid multi-signal scoring. If a challenge is suspected and HTML/URL
    are provided, it pipes the page to the LLM. If the LLM successfully extracts 
    valid target content (like posts), it gracefully bypasses the HITL prompt.
    """
    if score >= 40:
        if html and url:
            print(f"[*] Potential challenge detected (Score: {score}). Verifying with LLM...")
            try:
                from app.parser.llm_parser import _llm_text_extract
                llm_posts = _llm_text_extract(html, url)
                if llm_posts and len(llm_posts) > 0:
                    print(f"[+] LLM verified valid content ({len(llm_posts)} posts) despite challenge signals. Bypassing HITL!")
                    return False
            except Exception as e:
                print(f"[-] LLM verification skipped/failed: {e}")
        return True
    return False


STATE_FILE = "browser_state.json"

_playwright_context = None
_browser = None
_context = None
_page = None

CAPTCHA_TIMEOUT_SECONDS = int(os.getenv("CAPTCHA_TIMEOUT_SECONDS", 3600))
VALID_CAPTCHA_ACTIONS = {"resume", "new", "exit"}

DASHBOARD_MODE = False
dashboard_captcha_queue = queue.Queue(maxsize=1)


def _drain_captcha_queue():
    """Clear any stale or unconsumed HITL CAPTCHA actions."""
    while True:
        try:
            dashboard_captcha_queue.get_nowait()
        except queue.Empty:
            break


def submit_captcha_action(action, force=False):
    """Queue a validated HITL CAPTCHA action from the dashboard.

    Args:
        action: One of 'resume', 'new', 'exit'.
        force: If True (e.g., crawl stop), replace any pending item.

    Returns:
        bool: True if queued cleanly, False if queue already holds an action.
    """
    normalized = (action or "").strip().lower()
    if normalized not in VALID_CAPTCHA_ACTIONS:
        raise ValueError(f"Invalid CAPTCHA action '{action}'. Must be one of {sorted(VALID_CAPTCHA_ACTIONS)}.")
    if force:
        _drain_captcha_queue()
    try:
        dashboard_captcha_queue.put_nowait(normalized)
        return True
    except queue.Full:
        return False

_state_lock = threading.Lock()
stop_requested = threading.Event()
_consecutive_failures = 0


def request_stop():
    """Signal all crawler loops and sleeps to abort cooperatively."""
    stop_requested.set()
    try:
        submit_captcha_action("exit", force=True)
    except Exception:
        pass


def clear_stop_request():
    """Clear stop flag before starting a fresh crawl."""
    stop_requested.clear()


def is_stop_requested():
    """Return True if an active crawl stop has been signaled."""
    return stop_requested.is_set()


_last_tor_check_time = 0
_last_tor_check_result = False
_last_tor_ip = None


def check_tor(force=False):
    """Verify Tor connection is working with a 15-second cache to prevent API flooding."""
    global _last_tor_check_time, _last_tor_check_result, _last_tor_ip
    if not USE_TOR:
        print("[!] Tor disabled — running in clearnet mode")
        return False

    now = time.time()
    with _state_lock:
        if not force and (now - _last_tor_check_time < 15):
            return _last_tor_check_result

    try:
        r = requests.get(
            "https://check.torproject.org/api/ip",
            proxies=_requests_proxies,
            timeout=15
        )
        data = r.json()
        is_tor = bool(data.get("IsTor"))
        ip = data.get("IP")
        with _state_lock:
            _last_tor_ip = ip
            _last_tor_check_time = now
            _last_tor_check_result = is_tor
        if is_tor:
            print(f"[+] Tor OK | Exit IP: {ip}")
            return True
        print(f"[!] Connected but NOT through Tor | IP: {ip}")
        return False
    except Exception as e:
        print(f"[-] Tor connection failed: {e}")
        with _state_lock:
            _last_tor_check_time = now
            _last_tor_check_result = False
        return False


def get_tor_details(force=False):
    """Return dictionary with Tor connection status and current IP."""
    if not USE_TOR:
        return {"tor_active": False, "is_tor": False, "ip": None, "mode": "clearnet"}
    is_ok = check_tor(force=force)
    with _state_lock:
        ip = _last_tor_ip
    return {
        "tor_active": is_ok,
        "is_tor": is_ok,
        "ip": ip,
        "mode": "tor"
    }


def is_onion_url(url):
    """Check if a URL targets a .onion hidden service."""
    if not url:
        return False
    try:
        host = urlparse(url if "://" in url else f"http://{url}").netloc.lower()
        host = host.split(":")[0]
        return host.endswith(".onion")
    except Exception:
        return False


def ensure_tor_safety(target_url=None):
    """
    Centralized Target-Aware OPSEC Gate.
    
    1. If target_url is .onion: Tor is MANDATORY. Refuses to proceed if Tor
       is offline or disabled to avoid local DNS/IP leaks.
    2. If USE_TOR is enabled: Tor must be operational before any crawl starts.
    3. If target_url is clearnet and USE_TOR is disabled: logs an explicit warning.
    """
    if is_onion_url(target_url):
        if not USE_TOR:
            raise TorOPSECError(
                "CRITICAL OPSEC VIOLATION: Target is a .onion address but USE_TOR is False. "
                "Crawl blocked to prevent local DNS leak."
            )
        if not check_tor():
            raise TorOPSECError(
                "CRITICAL OPSEC VIOLATION: Target is a .onion address but Tor is offline or unreachable. "
                "Crawl blocked to prevent identity leak."
            )
        return True

    if USE_TOR:
        if not check_tor():
            raise TorOPSECError(
                "Tor is enabled in configuration but unreachable. "
                "Crawl blocked to prevent accidental cleartext fallback."
            )
        return True

    print("[!] OPSEC WARNING: Tor is disabled. Crawling this clearnet host will expose your real IP address.")
    return True


def renew_tor_circuit():
    """Request a new Tor circuit (new exit node = new IP).
    
    Useful when a site blocks your current exit node.
    Requires ControlPort (9051 or 9151) to be enabled in torrc.
    Returns True if circuit was renewed and verified, False otherwise.
    """
    if not USE_TOR:
        return False
    
    try:
        from stem import Signal
        from stem.control import Controller
        
        default_control = 9151 if '9150' in TOR_PROXY else 9051
        control_port = int(os.getenv('TOR_CONTROL_PORT', default_control))
        with Controller.from_port(port=control_port) as controller:
            controller.authenticate()
            controller.signal(Signal.NEWNYM)
            print("[+] New Tor circuit requested — waiting 10s for build...")
            time.sleep(10)

        if check_tor(force=True):
            print("[+] Verified fresh Tor circuit is operational.")
            return True
        else:
            print("[-] Tor circuit requested but connection verification failed.")
            return False
    except ImportError:
        print("[-] 'stem' library not installed — cannot renew circuit")
        return False
    except Exception as e:
        print(f"[-] Circuit renewal failed: {e}")
        return False


# All Tor Browser users share this UA family — rotating too wildly is suspicious.
TOR_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; rv:128.0) Gecko/20100101 Firefox/128.0",
    "Mozilla/5.0 (Windows NT 10.0; rv:115.0) Gecko/20100101 Firefox/115.0",
    "Mozilla/5.0 (Windows NT 10.0; rv:109.0) Gecko/20100101 Firefox/109.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:115.0) Gecko/20100101 Firefox/115.0",
]

CLEARNET_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:130.0) Gecko/20100101 Firefox/130.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 Edg/130.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.6 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
]

VIEWPORT_SIZES = [
    {"width": 1920, "height": 1080},
    {"width": 1366, "height": 768},
    {"width": 1536, "height": 864},
    {"width": 1440, "height": 900},
    {"width": 1280, "height": 720},
    {"width": 1600, "height": 900},
]


_STEALTH_INIT_SCRIPT = """
    Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
    Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
    Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
    window.chrome = { runtime: {} };
"""


def _get_random_ua():
    """Select a random user-agent based on whether Tor is enabled."""
    if USE_TOR:
        return random.choice(TOR_USER_AGENTS)
    return random.choice(CLEARNET_USER_AGENTS)


def _get_random_viewport():
    """Select a random viewport size from common screen resolutions."""
    return random.choice(VIEWPORT_SIZES)


def _build_launch_args():
    """Build Chromium launch arguments with optional Tor proxy.
    
    Extracted to eliminate 3 identical copies of launch-arg construction.
    """
    args = ["--disable-blink-features=AutomationControlled"]
    if USE_TOR:
        # CRITICAL: Set the proxy at the browser level AND force DNS through it.
        # Without --proxy-server + --host-resolver-rules, Chromium tries to resolve
        # .onion addresses via local DNS, which always fails because only Tor
        # can resolve .onion. This causes ERR_SOCKS_CONNECTION_FAILED.
        args.extend([
            f"--proxy-server={TOR_PROXY}",
            "--host-resolver-rules=MAP * ~NOTFOUND , EXCLUDE 127.0.0.1",
        ])
    return args


def _build_context_kwargs(ua, viewport=None, load_state=False):
    """Build browser context keyword arguments.
    
    Extracted to eliminate 3 identical copies of context-kwarg construction.
    
    Args:
        ua: user-agent string
        viewport: viewport dict or None
        load_state: if True, load session state from STATE_FILE
    """
    proxy = {"server": TOR_PROXY} if USE_TOR else None
    kwargs = {
        "proxy": proxy,
        "user_agent": ua,
        "ignore_https_errors": True,
        "locale": "en-US",
        "timezone_id": "America/New_York",
    }
    if viewport:
        kwargs["viewport"] = viewport
    if load_state and os.path.exists(STATE_FILE) and os.path.getsize(STATE_FILE) > 0:
        kwargs["storage_state"] = STATE_FILE
    return kwargs


def _apply_stealth(page):
    """Apply stealth settings and init script to a page."""
    stealth_sync(page)
    page.add_init_script(_STEALTH_INIT_SCRIPT)


_SESSION_EXPIRED_INDICATORS = [
    "session expired",
    "session has expired",
    "bot-like behaviour",
    "bot like behaviour",
    "suspicious activity",
    "automated access",
    "start with new identity",
    "new identity",
    "try again with a new session",
    "your session was terminated",
    "inactivity",
]


def _is_session_expired_page(html):
    """Detect if the page says the session was killed due to bot detection."""
    if not html:
        return False
    lower = html[:3000].lower()
    return any(indicator in lower for indicator in _SESSION_EXPIRED_INDICATORS)


def _bump_consecutive_failures():
    """Increment consecutive failure counter thread-safely and return new value."""
    global _consecutive_failures
    with _state_lock:
        _consecutive_failures += 1
        return _consecutive_failures


def _reset_consecutive_failures():
    """Reset consecutive failure counter thread-safely."""
    global _consecutive_failures
    with _state_lock:
        _consecutive_failures = 0


def _close_playwright_handles(page=None, context=None, browser=None, pw_ctx=None):
    """Best-effort close of Playwright objects without letting one failure skip the rest."""
    for label, obj, method in (
        ("page", page, "close"),
        ("context", context, "close"),
        ("browser", browser, "close"),
        ("playwright", pw_ctx, "stop"),
    ):
        if obj is None:
            continue
        try:
            getattr(obj, method)()
        except Exception as exc:
            log_failure(f"Error tearing down Playwright {label}", exc, level=logging.DEBUG, logger_name="browser")


def ensure_clean_browser():
    """Reset any stale Playwright state. Call before spawning a new crawl thread."""
    global _playwright_context, _browser, _context, _page
    _close_playwright_handles(page=_page, context=_context, browser=_browser, pw_ctx=_playwright_context)
    _playwright_context = None
    _browser = None
    _context = None
    _page = None
    _reset_consecutive_failures()
    _drain_captcha_queue()


def init_browser(renew_identity=False):
    """Initialize the headless Playwright browser with randomized fingerprint.
    
    Args:
        renew_identity: If True, get a new Tor circuit before launching.
    """
    global _playwright_context, _browser, _context, _page

    if _playwright_context is not None:
        return True

    if USE_TOR and renew_identity:
        print("\n[*] Requesting new Tor identity before browser launch...")
        if not renew_tor_circuit():
            print("[-] Warning: Tor circuit renewal failed or unverified.")
            return False

    print("\n[*] Initializing Playwright browser engine...")
    _playwright_context = sync_playwright().start()

    ua = _get_random_ua()
    viewport = _get_random_viewport()

    print(f"[*] User-Agent: {ua[:60]}...")
    print(f"[*] Viewport: {viewport['width']}x{viewport['height']}")
    if USE_TOR:
        print(f"[*] Proxy: {TOR_PROXY} (DNS via proxy)")

    _browser = _playwright_context.chromium.launch(
        headless=True,
        args=_build_launch_args()
    )

    if not renew_identity and os.path.exists(STATE_FILE) and os.path.getsize(STATE_FILE) > 0:
        print(f"[*] Loading authenticated session state from {STATE_FILE}")

    context_kwargs = _build_context_kwargs(ua, viewport, load_state=not renew_identity)
    _context = _browser.new_context(**context_kwargs)
    _page = _context.new_page()
    _apply_stealth(_page)


def close_browser():
    """Clean up Playwright resources."""
    global _playwright_context, _browser, _context, _page
    try:
        if _context:
            try:
                _context.storage_state(path=STATE_FILE)
                print(f"[*] Browser session state saved to {STATE_FILE}")
            except Exception as exc:
                log_failure("Failed saving browser state on close", exc, level=logging.DEBUG, logger_name="browser")
    finally:
        _close_playwright_handles(browser=_browser, pw_ctx=_playwright_context)
        _playwright_context = None
        _browser = None
        _context = None
        _page = None


class CrawlAbortedByUser(Exception):
    """Raised when the user chooses to abort the crawl during an interactive challenge."""


class SessionExpired(Exception):
    """Raised when the site detects bot behavior and invalidates the session.
    
    The crawler should catch this, get a new Tor identity, and restart
    the crawl loop while preserving already-visited URLs.
    """


class TorOPSECError(Exception):
    """Raised when an operation violates Tor OPSEC rules (e.g. .onion without active Tor)."""


def _simulate_human(page):
    """Simulate human browsing behavior to reduce bot fingerprinting.
    
    Actions (randomized):
      - Random scroll down 30-70% of page
      - Small mouse movements to random coordinates
      - Brief pauses between actions
    """
    viewport = page.viewport_size
    if not viewport:
        return
    
    width = viewport["width"]
    height = viewport["height"]
    
    time.sleep(random.uniform(0.5, 1.5))
    
    for _ in range(random.randint(1, 3)):
        x = random.randint(100, max(101, width - 100))
        y = random.randint(100, max(101, height - 100))
        page.mouse.move(x, y)
        time.sleep(random.uniform(0.1, 0.4))
    
    scroll_percent = random.uniform(0.3, 0.7)
    page.evaluate(f"window.scrollTo(0, document.body.scrollHeight * {scroll_percent})")
    time.sleep(random.uniform(0.5, 1.0))
    
    if random.random() < 0.3:
        page.evaluate("window.scrollBy(0, -200)")
        time.sleep(random.uniform(0.3, 0.6))


def new_identity():
    """Completely reset browser session and get a new Tor circuit.
    
    This is the nuclear option — used when a site detects bot behavior
    and invalidates the session. It:
      1. Closes the current browser completely
      2. Deletes the old session state file (cookies are burned)
      3. Requests a new Tor circuit (new exit IP) and verifies it
      4. Re-launches the browser with a fresh fingerprint
      
    Returns:
        bool: True if new identity was successfully established, False otherwise.
    """
    global _playwright_context, _browser, _context, _page
    
    print(f"\n{'='*60}")
    print("[*] STARTING NEW IDENTITY")
    print(f"{'='*60}")
    
    _close_playwright_handles(browser=_browser, pw_ctx=_playwright_context)
    
    _playwright_context = None
    _browser = None
    _context = None
    _page = None
    _drain_captcha_queue()
    
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
        print("[*] Old session state deleted.")
    
    if USE_TOR:
        if not renew_tor_circuit():
            print("[-] CRITICAL: Failed to establish and verify a new Tor circuit.")
            print(f"{'='*60}\n")
            return False

    init_browser(renew_identity=False)
    
    print(f"[*] New identity established.")
    print(f"{'='*60}\n")
    return True


def _check_session_status(page):
    """Check if the current page is a session-expired page.
    
    Returns:
        tuple: (current_html, is_session_dead)
    """
    current_html = ""
    if page:
        current_html = page.content()
    return current_html, _is_session_expired_page(current_html)


def _launch_visible_browser(url, ua):
    """Launch a visible browser window for manual CAPTCHA solving.
    
    Returns:
        tuple: (browser, context, page)
    """
    print("\n[*] Spawning visible browser window for manual bypass...")
    _browser_vis = _playwright_context.chromium.launch(
        headless=False,
        args=_build_launch_args()
    )
    
    context_kwargs = _build_context_kwargs(ua, load_state=True)
    _context_vis = _browser_vis.new_context(**context_kwargs)
    _page_vis = _context_vis.new_page()
    _apply_stealth(_page_vis)
    
    try:
        _page_vis.goto(url, timeout=180000)
    except Exception as e:
        print(f"[-] Error loading page in visible browser: {e}")
    
    return _browser_vis, _context_vis, _page_vis


def _get_user_captcha_choice():
    """Get the user's choice for how to handle a CAPTCHA/challenge.
    
    In dashboard mode, drains any stale signals, advertises the pending challenge,
    and waits with a bounded timeout on a single-slot queue so responses cannot
    be overwritten and abandoned sessions cannot deadlock the crawler thread.
    In CLI mode, prompts the user interactively.
    
    Returns:
        str: user choice ('exit', 'new', 'resume', or empty for continue)
    """
    if DASHBOARD_MODE:
        _drain_captcha_queue()
        notify_captcha_pending()
        print("[*] CAPTCHA pending — awaiting dashboard action...")
        try:
            action = dashboard_captcha_queue.get(timeout=CAPTCHA_TIMEOUT_SECONDS)
            return action if action in VALID_CAPTCHA_ACTIONS else "exit"
        except queue.Empty:
            print(f"[!] HITL challenge timed out after {CAPTCHA_TIMEOUT_SECONDS}s — aborting crawl safely.")
            return "exit"
    else:
        print(">>>")
        print(">>> Options:")
        print(">>>   - Press ENTER to continue scraping.")
        print(">>>   - Type 'new' and press ENTER to get a NEW IDENTITY and restart crawl.")
        print(">>>   - Type 'exit' and press ENTER to stop scraping and save data.")
        return input(">>> ").strip().lower()


def _resume_headless_browser():
    """Re-launch headless browser after manual CAPTCHA bypass.
    
    Reads the saved state from STATE_FILE and creates a fresh headless session.
    """
    global _browser, _context, _page
    
    ua = _get_random_ua()
    viewport = _get_random_viewport()
    
    _browser = _playwright_context.chromium.launch(
        headless=True,
        args=_build_launch_args()
    )
    
    context_kwargs = _build_context_kwargs(ua, viewport, load_state=True)
    _context = _browser.new_context(**context_kwargs)
    _page = _context.new_page()
    _apply_stealth(_page)


def solve_captcha_interactive(url):
    """
    Suspends the headless browser, opens a visible browser window, 
    waits for the user to manually solve the CAPTCHA/Queue, saves 
    the new authenticated session, and resumes headless mode.
    
    If the page is a session-expired/bot-detected page, offers
    automatic new identity renewal.
    """
    global _playwright_context, _browser, _context, _page

    if DASHBOARD_MODE:
        print("[!] CAPTCHA detected while in DASHBOARD_MODE.")
        print("[!] Setting state to pending and waiting for dashboard resolution...")

    _, is_session_dead = _check_session_status(_page)
    
    print(f"\n{'!'*60}")
    if is_session_dead:
        print(f"[!] SESSION EXPIRED / BOT DETECTION TRIGGERED")
    else:
        print(f"[!] CHALLENGE / CAPTCHA / LOGIN WALL DETECTED")
    print(f"URL: {url}")
    print(f"{'!'*60}")
    
    if _context:
        try:
            _context.storage_state(path=STATE_FILE)
        except Exception as exc:
            log_failure("Failed saving session state before HITL", exc, level=logging.DEBUG, logger_name="browser")
        finally:
            _close_playwright_handles(browser=_browser)
        
    ua = _get_random_ua()
    
    vis_browser, vis_context, vis_page = _launch_visible_browser(url, ua)
    
    print("\n>>> Please solve the CAPTCHA, Login, or refresh the session in the opened browser window.")
    print(">>> Wait until you see the actual forum content.")
    
    html = ""
    try:
        user_choice = _get_user_captcha_choice()
        
        if user_choice in ['exit', 'quit', 'q']:
            print("\n[*] User chose to abort the crawl. Cleaning up visible browser...")
            raise CrawlAbortedByUser("Crawl aborted by user during challenge.")
        
        if user_choice in ['new', 'identity', 'renew', 'n']:
            print("\n[*] User chose new identity. Closing visible browser...")
            try: vis_browser.close()
            except Exception: pass
            try: _playwright_context.stop()
            except Exception: pass
            _playwright_context = None
            _browser = None
            _context = None
            _page = None
            raise SessionExpired("User requested new identity after bot detection.")
        
        vis_context.storage_state(path=STATE_FILE)
        html = vis_page.content()
    finally:
        try:
            vis_browser.close()
        except Exception:
            pass
    
    print("[*] State saved. Resuming headless scraping...\n")
    
    _resume_headless_browser()
    
    return html


def _ensure_url_scheme(url):
    """Ensure URL is trimmed and has a proper protocol."""
    url = url.strip()
    if not url:
        return ""
    if "://" not in url:
        url = "http://" + url
    return url



def _handle_wait_challenges(page):
    """Detect automatic JS challenges (Cloudflare/Endgame/DDoS-Guard) and let them resolve."""
    try:
        html = page.content().lower()
        if any(k in html for k in ["ddos-guard", "cloudflare", "just a moment", "checking your browser", "endgame", "please wait", "attention required"]):
            print("[*] Anti-bot JS challenge detected! Waiting 10-15s for automatic resolution...")
            _simulate_human(page)
            page.wait_for_timeout(10000)
            
            html = page.content().lower()
            if any(k in html for k in ["ddos-guard", "cloudflare", "just a moment", "checking your browser", "endgame", "please wait"]):
                print("[*] Challenge still active. Looking for checkboxes or verify buttons...")
                for _ in range(3):
                    try:
                        checkbox = page.locator("input[type='checkbox']").first
                        if checkbox and checkbox.is_visible(timeout=1000):
                            checkbox.click(timeout=2000)
                            page.wait_for_timeout(3000)
                    except Exception as exc:
                        log_failure("Wait-challenge checkbox interaction failed", exc, level=logging.DEBUG, logger_name="browser")

                    try:
                        btn = page.locator("button:has-text('Verify'), button:has-text('Continue'), button[name='submit']").first
                        if btn and btn.is_visible(timeout=1000):
                            btn.click(timeout=2000)
                            page.wait_for_timeout(3000)
                    except Exception as exc:
                        log_failure("Wait-challenge button interaction failed", exc, level=logging.DEBUG, logger_name="browser")
                page.wait_for_timeout(3000)
    except Exception as e:
        print(f"[-] Auto-challenge resolution failed: {e}")


def fetch_page(url, retries=3):
    """
    Fetch a page using the Playwright browser.
    Automatically handles CAPTCHA interruptions.
    
    Enhanced with:
      - Cooperative stop_requested checks and interruptible backoff
      - Tor OPSEC gate checking before navigation
      - Human-like behavior simulation after page load
      - Tor circuit renewal on repeated failures
      - Adaptive retry delays based on consecutive failures
    """
    global _page
    if is_stop_requested():
        raise CrawlAbortedByUser("Crawl stopped by user.")

    url = _ensure_url_scheme(url)
    if not url:
        return None

    ensure_tor_safety(url)

    if _page is None:
        init_browser()
        
    for attempt in range(retries):
        failures = 0
        try:
            if is_stop_requested():
                raise CrawlAbortedByUser("Crawl stopped by user.")
            if _page is None:
                raise CrawlAbortedByUser("Browser was closed externally (e.g. by Dashboard).")
                
            print(f"[*] Navigating to {url} (Attempt {attempt+1})...")
            # 'networkidle' is too flaky on Tor.
            response = _page.goto(url, wait_until="domcontentloaded", timeout=180000)
            if is_stop_requested():
                raise CrawlAbortedByUser("Crawl stopped by user.")
            
            _page.wait_for_timeout(random.randint(1500, 3000))
            
            _simulate_human(_page)
            if is_stop_requested():
                raise CrawlAbortedByUser("Crawl stopped by user.")
            
            _handle_wait_challenges(_page)
            
            html = _page.content()
            
            current_url = _page.url
            
            score, reasons = _score_challenge_signals(html, current_url=current_url, original_url=url)
            if _is_challenge_page(score, html=html, url=current_url):
                print(f"[!] Challenge detected (score={score}, signals={reasons})")
                failures = _bump_consecutive_failures()
                return solve_captcha_interactive(url)
            
            if response and response.status in [403, 429, 500, 502, 503]:
                failures = _bump_consecutive_failures()
                print(f"[-] Status {response.status} on attempt {attempt+1}")
                
                if response.status == 403 and attempt < retries - 1 and not is_stop_requested():
                    print("[!] 403 Forbidden — trying new Tor circuit...")
                    renew_tor_circuit()
            else:
                _reset_consecutive_failures()
                return html
                
        except CrawlAbortedByUser:
            raise
        except SessionExpired:
            raise
        except Exception as e:
            if is_stop_requested():
                raise CrawlAbortedByUser("Crawl stopped by user.")
            failures = _bump_consecutive_failures()
            err_str = str(e).lower()
            if "net::err_proxy_connection_failed" in err_str:
                print(f"[-] Tor Proxy Connection Failed! Ensure Tor is running on port 9150.")
                return None
            if "net::err_socks_connection_failed" in err_str:
                print(f"[-] SOCKS connection failed. Check Tor connectivity.")
                return None
            if "invalid url" in err_str:
                print(f"[-] Critical: The URL provided is invalid: '{url}'")
                return None
            print(f"[-] Playwright attempt {attempt+1} failed: {e}")
            
        if is_stop_requested():
            raise CrawlAbortedByUser("Crawl stopped by user.")

        failure_multiplier = min(1 + (failures * 0.3), 4.0)
        base_delay = min(3 * (2 ** attempt), 30)
        delay = (base_delay + random.uniform(0, base_delay * 0.5)) * failure_multiplier
        print(f"  -> Retrying in {delay:.1f}s (failure multiplier: {failure_multiplier:.1f}x)")
        if stop_requested.wait(timeout=delay):
            raise CrawlAbortedByUser("Crawl stopped by user.")
        
    return None
