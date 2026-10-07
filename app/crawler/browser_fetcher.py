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

import re
import time
import random
import os
import threading
import requests
from playwright.sync_api import sync_playwright
try:
    from playwright_stealth import stealth_sync
except ImportError:
    from playwright_stealth import stealth as stealth_sync


# ── Tor Configuration (merged from fetcher.py) ──────────────────

USE_TOR = True    # change to True when Tor is configured

TOR_PROXY = "socks5://127.0.0.1:9150"

# socks5h variant for requests library (DNS resolution through Tor)
_TOR_PROXY_REQUESTS = "socks5h://127.0.0.1:9150"
_requests_proxies = {
    "http": _TOR_PROXY_REQUESTS,
    "https": _TOR_PROXY_REQUESTS
} if USE_TOR else None

# Challenge page indicators (static keyword fallback — lowest priority signal)
CHALLENGE_INDICATORS = [
    "captcha", "ddos-guard", "cloudflare", "challenge",
    "enable javascript", "please wait", "checking your browser",
    "endgame", "just a moment",  # common onion anti-DDoS services
    "placed in a queue", "estimated wait time",  # queue pages
    "session expired", "bot-like behavio", "new session",  # bot detection
]

# URL path fragments that indicate a login/auth redirect
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
    
    # ── Signal 1: DOM Structure — Password fields ────────────────
    # This is the most universal indicator. Every login page on Earth
    # (clearnet or .onion) uses <input type="password">.
    has_password_field = ('type="password"' in lower_html or 
                         "type='password'" in lower_html or
                         'type=password' in lower_html)
    
    # ── Signal 2: Content density — How much readable text? ──────
    # Strip all tags to estimate visible text word count.
    # Remove script/style blocks first
    stripped = re.sub(r'<(script|style|noscript)[^>]*>.*?</\1>', '', lower_html, flags=re.DOTALL)
    # Remove all remaining tags
    stripped = re.sub(r'<[^>]+>', ' ', stripped)
    # Collapse whitespace
    stripped = re.sub(r'\s+', ' ', stripped).strip()
    word_count = len(stripped.split())
    
    low_content = word_count < 200
    very_low_content = word_count < 80
    
    # ── Signal 3: URL redirect detection ─────────────────────────
    url_redirected_to_login = False
    if current_url and original_url:
        # Normalize for comparison (strip trailing slashes, fragments)
        curr_path = current_url.split('?')[0].split('#')[0].rstrip('/')
        orig_path = original_url.split('?')[0].split('#')[0].rstrip('/')
        
        if curr_path != orig_path:
            # URL changed — check if it landed on a login-like path
            curr_lower = current_url.lower()
            for pattern in _LOGIN_URL_PATTERNS:
                if pattern in curr_lower:
                    url_redirected_to_login = True
                    break
    
    # ── Signal 4: Static keyword match (existing fallback) ───────
    keyword_matched = False
    first_2k = lower_html[:2000]
    for indicator in CHALLENGE_INDICATORS:
        if indicator in first_2k:
            keyword_matched = True
            break
    
    # ── Signal 5: OAuth/SSO provider buttons (clearnet only) ─────
    is_onion = original_url and '.onion' in original_url
    has_oauth = False
    if not is_onion:
        oauth_markers = [
            'accounts.google.com', 'appleid.apple.com',
            'facebook.com/login', 'facebook.com/v', 
            'oauth', 'openid', 'sso',
        ]
        has_oauth = any(marker in lower_html for marker in oauth_markers)
    
    # ── Scoring ──────────────────────────────────────────────────
    
    # Password field + very little content = almost certainly a login wall
    if has_password_field and very_low_content:
        score += 40
        reasons.append("password_field+very_low_content")
    elif has_password_field and low_content:
        score += 35
        reasons.append("password_field+low_content")
    elif has_password_field:
        # Password field exists but page has substantial content too
        # (e.g., sidebar login on a forum that shows public threads)
        score += 25
        reasons.append("password_field_present")
    
    if url_redirected_to_login:
        score += 30
        reasons.append("url_redirected_to_login")
    
    if low_content and not has_password_field:
        # Low content without a password field — could be a DDoS wait page
        score += 15
        reasons.append("low_content_density")
    
    if keyword_matched:
        # Boosted score to ensure DDOS-Guard/Cloudflare trigger HITL if auto-solve fails
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

DASHBOARD_MODE = False
dashboard_captcha_event = threading.Event()
dashboard_captcha_action = None

# Track consecutive failures for adaptive behavior
_consecutive_failures = 0


# ── Tor Utilities (merged from fetcher.py) ───────────────────────

def check_tor():
    """Verify Tor connection is working."""
    if not USE_TOR:
        print("[!] Tor disabled — running in clearnet mode")
        return False

    try:
        r = requests.get(
            "https://check.torproject.org/api/ip",
            proxies=_requests_proxies,
            timeout=15
        )
        data = r.json()
        if data.get("IsTor"):
            print(f"[+] Tor OK | Exit IP: {data.get('IP')}")
            return True
        else:
            print(f"[!] Connected but NOT through Tor | IP: {data.get('IP')}")
            return False
    except Exception as e:
        print(f"[-] Tor connection failed: {e}")
        return False


def renew_tor_circuit():
    """Request a new Tor circuit (new exit node = new IP).
    
    Useful when a site blocks your current exit node.
    Requires ControlPort 9051 to be enabled in torrc.
    """
    if not USE_TOR:
        return False
    
    try:
        from stem import Signal
        from stem.control import Controller
        
        import os
        default_control = 9151 if '9150' in TOR_PROXY else 9051
        control_port = int(os.getenv('TOR_CONTROL_PORT', default_control))
        with Controller.from_port(port=control_port) as controller:
            controller.authenticate()
            controller.signal(Signal.NEWNYM)
            print("[+] New Tor circuit requested — waiting 10s for build...")
            time.sleep(10)  # circuits take time to build
            return True
    except ImportError:
        print("[-] 'stem' library not installed — cannot renew circuit")
        return False
    except Exception as e:
        print(f"[-] Circuit renewal failed: {e}")
        return False


# ── User-Agent Rotation Pool ─────────────────────────────────────

# For .onion sites: Tor Browser variants (same family, slight version variation)
# All Tor Browser users share this UA family — rotating too wildly is suspicious.
TOR_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; rv:128.0) Gecko/20100101 Firefox/128.0",
    "Mozilla/5.0 (Windows NT 10.0; rv:115.0) Gecko/20100101 Firefox/115.0",
    "Mozilla/5.0 (Windows NT 10.0; rv:109.0) Gecko/20100101 Firefox/109.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:115.0) Gecko/20100101 Firefox/115.0",
]

# For clearnet: realistic modern browser UAs
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

# Common viewport sizes to rotate between
VIEWPORT_SIZES = [
    {"width": 1920, "height": 1080},
    {"width": 1366, "height": 768},
    {"width": 1536, "height": 864},
    {"width": 1440, "height": 900},
    {"width": 1280, "height": 720},
    {"width": 1600, "height": 900},
]


# ── De-duplicated constants & helpers ────────────────────────────

# Stealth init script — extracted from 3 identical copies
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


# Indicators that the site detected bot behavior and killed the session
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


def ensure_clean_browser():
    """Reset any stale Playwright state. Call before spawning a new crawl thread."""
    global _playwright_context, _browser, _context, _page
    if _page:
        try: _page.close()
        except: pass
    if _context:
        try: _context.close()
        except: pass
    if _browser:
        try: _browser.close()
        except: pass
    if _playwright_context:
        try: _playwright_context.stop()
        except: pass
    _playwright_context = None
    _browser = None
    _context = None
    _page = None


def init_browser(renew_identity=False):
    """Initialize the headless Playwright browser with randomized fingerprint.
    
    Args:
        renew_identity: If True, get a new Tor circuit before launching.
    """
    global _playwright_context, _browser, _context, _page

    if _playwright_context is not None:
        return

    # Get a fresh Tor IP on startup or identity renewal
    if USE_TOR and renew_identity:
        print("\n[*] Requesting new Tor identity before browser launch...")
        renew_tor_circuit()

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

    # Only load old session state if NOT renewing identity
    if not renew_identity and os.path.exists(STATE_FILE) and os.path.getsize(STATE_FILE) > 0:
        print(f"[*] Loading authenticated session state from {STATE_FILE}")

    context_kwargs = _build_context_kwargs(ua, viewport, load_state=not renew_identity)
    _context = _browser.new_context(**context_kwargs)
    _page = _context.new_page()
    _apply_stealth(_page)


def close_browser():
    """Clean up Playwright resources."""
    global _playwright_context, _browser, _context, _page
    if _context:
        try:
            _context.storage_state(path=STATE_FILE)
            print(f"[*] Browser session state saved to {STATE_FILE}")
        except:
            pass

    if _browser:
        try: _browser.close()
        except: pass
    if _playwright_context:
        try: _playwright_context.stop()
        except: pass

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


# ── Human-Like Behavior Simulation ───────────────────────────────

def _simulate_human(page):
    """Simulate human browsing behavior to reduce bot fingerprinting.
    
    Actions (randomized):
      - Random scroll down 30-70% of page
      - Small mouse movements to random coordinates
      - Brief pauses between actions
    """
    # Get page dimensions
    viewport = page.viewport_size
    if not viewport:
        return
    
    width = viewport["width"]
    height = viewport["height"]
    
    # 1. Small random pause (human reaction time)
    time.sleep(random.uniform(0.5, 1.5))
    
    # 2. Mouse movement to a random spot (simulates looking around)
    for _ in range(random.randint(1, 3)):
        x = random.randint(100, max(101, width - 100))
        y = random.randint(100, max(101, height - 100))
        page.mouse.move(x, y)
        time.sleep(random.uniform(0.1, 0.4))
    
    # 3. Scroll down 30-70% of the page
    scroll_percent = random.uniform(0.3, 0.7)
    page.evaluate(f"window.scrollTo(0, document.body.scrollHeight * {scroll_percent})")
    time.sleep(random.uniform(0.5, 1.0))
    
    # 4. Occasionally scroll back up a bit (humans do this)
    if random.random() < 0.3:
        page.evaluate("window.scrollBy(0, -200)")
        time.sleep(random.uniform(0.3, 0.6))





def new_identity():
    """Completely reset browser session and get a new Tor circuit.
    
    This is the nuclear option — used when a site detects bot behavior
    and invalidates the session. It:
      1. Closes the current browser completely
      2. Deletes the old session state file (cookies are burned)
      3. Requests a new Tor circuit (new exit IP)
      4. Re-launches the browser with a fresh fingerprint
    """
    global _playwright_context, _browser, _context, _page
    
    print(f"\n{'='*60}")
    print("[*] STARTING NEW IDENTITY")
    print(f"{'='*60}")
    
    # Step 1: Kill the current browser entirely
    if _browser:
        try: _browser.close()
        except: pass
    if _playwright_context:
        try: _playwright_context.stop()
        except: pass
    
    _playwright_context = None
    _browser = None
    _context = None
    _page = None
    
    # Step 2: Delete old session state (cookies are compromised)
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
        print("[*] Old session state deleted.")
    
    # Step 3: Launch fresh browser with new Tor circuit
    init_browser(renew_identity=True)
    
    print(f"[*] New identity established.")
    print(f"{'='*60}\n")


# ── CAPTCHA Solving — broken down from monolithic function ───────

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
        headless=False,  # <--- VISIBLE MODE
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
    
    In dashboard mode, waits for the dashboard API event.
    In CLI mode, prompts the user interactively.
    
    Returns:
        str: user choice ('exit', 'new', or empty for continue)
    """
    if DASHBOARD_MODE:
        print(">>> [DASHBOARD_CAPTCHA_PENDING]")  # Magic string for dashboard.py to catch
        # Wait for the dashboard API to trigger the event
        dashboard_captcha_event.clear()
        dashboard_captcha_event.wait()
        return dashboard_captcha_action or 'exit'
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
        # Note: We do NOT raise CrawlAbortedByUser here anymore. We will pause below.
    
    # First, check if this is a session-expired page (not just a CAPTCHA)
    _, is_session_dead = _check_session_status(_page)
    
    print(f"\n{'!'*60}")
    if is_session_dead:
        print(f"[!] SESSION EXPIRED / BOT DETECTION TRIGGERED")
    else:
        print(f"[!] CHALLENGE / CAPTCHA / LOGIN WALL DETECTED")
    print(f"URL: {url}")
    print(f"{'!'*60}")
    
    # Save state from headless and close it
    if _context:
        _context.storage_state(path=STATE_FILE)
        _browser.close()
        
    ua = _get_random_ua()
    
    # Launch visible browser for manual bypass
    vis_browser, vis_context, vis_page = _launch_visible_browser(url, ua)
    
    print("\n>>> Please solve the CAPTCHA, Login, or refresh the session in the opened browser window.")
    print(">>> Wait until you see the actual forum content.")
    
    user_choice = _get_user_captcha_choice()
    
    if user_choice in ['exit', 'quit', 'q']:
        print("\n[*] User chose to abort the crawl. Cleaning up visible browser...")
        vis_browser.close()
        raise CrawlAbortedByUser("Crawl aborted by user during challenge.")
    
    if user_choice in ['new', 'identity', 'renew', 'n']:
        print("\n[*] User chose new identity. Closing visible browser...")
        vis_browser.close()
        try: _playwright_context.stop()
        except: pass
        _playwright_context = None
        _browser = None
        _context = None
        _page = None
        raise SessionExpired("User requested new identity after bot detection.")
    
    # User pressed ENTER — manual bypass completed, resume
    # Save the newly authenticated state (with the CAPTCHA clearance cookies)
    vis_context.storage_state(path=STATE_FILE)
    html = vis_page.content()
    
    # Close visible browser
    vis_browser.close()
    
    print("[*] State saved. Resuming headless scraping...\n")
    
    # Re-launch headless browser with the new state
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
            
            # Refresh HTML after waiting
            html = page.content().lower()
            if any(k in html for k in ["ddos-guard", "cloudflare", "just a moment", "checking your browser", "endgame", "please wait"]):
                print("[*] Challenge still active. Looking for checkboxes or verify buttons...")
                # Turnstile or similar checkboxes
                for _ in range(3):
                    try:
                        checkbox = page.locator("input[type='checkbox']").first
                        if checkbox and checkbox.is_visible(timeout=1000):
                            checkbox.click(timeout=2000)
                            page.wait_for_timeout(3000)
                    except:
                        pass
                        
                    try:
                        btn = page.locator("button:has-text('Verify'), button:has-text('Continue'), button[name='submit']").first
                        if btn and btn.is_visible(timeout=1000):
                            btn.click(timeout=2000)
                            page.wait_for_timeout(3000)
                    except:
                        pass
                page.wait_for_timeout(3000)
    except Exception as e:
        print(f"[-] Auto-challenge resolution failed: {e}")


def fetch_page(url, retries=3):
    """
    Fetch a page using the Playwright browser.
    Automatically handles CAPTCHA interruptions.
    
    Enhanced with:
      - Human-like behavior simulation after page load
      - Tor circuit renewal on repeated failures
      - Adaptive retry delays based on consecutive failures
    """
    global _page, _consecutive_failures
    if _page is None:
        init_browser()
        
    url = _ensure_url_scheme(url)
    if not url:
        return None
        
    for attempt in range(retries):
        try:
            if _page is None:
                raise CrawlAbortedByUser("Browser was closed externally (e.g. by Dashboard).")
                
            print(f"[*] Navigating to {url} (Attempt {attempt+1})...")
            # Wait until DOM content is loaded. 'networkidle' is too flaky on Tor.
            response = _page.goto(url, wait_until="domcontentloaded", timeout=180000)
            
            # Wait a tiny bit extra for DOM to settle
            _page.wait_for_timeout(random.randint(1500, 3000))
            
            # Simulate human browsing behavior
            _simulate_human(_page)
            
            # Wait for any JS challenges to resolve automatically
            _handle_wait_challenges(_page)
            
            html = _page.content()
            
            # Get current URL (may differ from requested if site redirected)
            current_url = _page.url
            
            score, reasons = _score_challenge_signals(html, current_url=current_url, original_url=url)
            if _is_challenge_page(score, html=html, url=current_url):
                print(f"[!] Challenge detected (score={score}, signals={reasons})")
                _consecutive_failures += 1
                return solve_captcha_interactive(url)
            
            # If we get an error code, fallback to retry
            if response and response.status in [403, 429, 500, 502, 503]:
                _consecutive_failures += 1
                print(f"[-] Status {response.status} on attempt {attempt+1}")
                
                # On 403, try renewing Tor circuit
                if response.status == 403 and attempt < retries - 1:
                    print("[!] 403 Forbidden — trying new Tor circuit...")
                    renew_tor_circuit()
            else:
                _consecutive_failures = 0  # Reset on success
                return html
                
        except CrawlAbortedByUser:
            # Propagate the abort signal up to the main crawler loop immediately
            raise
        except SessionExpired:
            # Propagate so the crawler can restart with a new identity
            raise
        except Exception as e:
            _consecutive_failures += 1
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
            
        # Adaptive retry delay: increases with consecutive failures
        failure_multiplier = min(1 + (_consecutive_failures * 0.3), 4.0)
        base_delay = min(3 * (2 ** attempt), 30)
        delay = (base_delay + random.uniform(0, base_delay * 0.5)) * failure_multiplier
        print(f"  -> Retrying in {delay:.1f}s (failure multiplier: {failure_multiplier:.1f}x)")
        time.sleep(delay)
        
    return None
