# MMFIC Beta Wiki & Known Issues

Welcome to the MMFIC (Multi-Modal Forum Identity Correlation) Wiki! 

Because the Trowler data-acquisition subsystem operates in highly adversarial, anti-bot environments (the Darkweb, Tor, and rate-limited social networks), it naturally encounters extreme edge cases. 

We are releasing this framework as a **Beta** alongside our academic research. Below is a list of known issues, architectural quirks, and areas where community contributions are highly welcome.

---

## 🛠️ Known Issues & Limitations

### 1. Tor Connection Timeouts (`ERR_TIMED_OUT`)
**The Problem:** Many `.onion` services (like Dread) are under constant heavy load or DDoS attacks. This makes the Tor network extremely slow, frequently causing Trowler's Playwright headless browser to timeout before `DOMContentLoaded` is reached.
**Current Workaround:** We have universally increased the Playwright navigation `timeout` to 180 seconds (3 minutes) in `browser_fetcher.py`.
**Help Wanted:** Building an asynchronous multi-tab pre-flight checker that validates the Tor circuit health before Trowler attempts a heavy navigation.

### 2. The Twitter / X Login Wall
**The Problem:** Twitter/X severely restricts unauthenticated scraping. While it is theoretically possible to extract 1 or 2 tweets from behind the "Log in or sign up" overlay using DOM selectors, attempting to scroll or navigate without a session instantly locks the crawler.
**Current Solution:** The Trowler Twitter Adapter (`twitter_adapter.py`) explicitly catches this scenario, suspends the crawler, and launches the Human-In-The-Loop (HITL) visible browser. It prompts you to log into a throwaway account. Once logged in, the session is saved to `browser_state.json` and reused natively for all future headless crawls.
**Help Wanted:** Implementing automated cookie injection from external auth providers to bypass the initial HITL step entirely.

### 3. LLM Fallback on Complex SPAs (React/Vue)
**The Problem:** Trowler uses an LLM (Gemini/Claude) as a fallback schema parser when standard heuristics fail. However, on massive Single Page Applications (like Twitter), the raw DOM text can exceed 15,000 characters of menu items, SVG paths, and hidden accessibility text. Because we limit the LLM input to 5,000 characters to save token costs, the LLM often never sees the actual posts.
**Current Solution:** For complex SPAs, we rely on custom adapters (like `twitter_adapter.py`) using precise CSS selectors (`data-testid="tweet"`). The LLM fallback is reserved exclusively for standard, static HTML forums (phpBB, XenForo, vBulletin, etc.).

### 4. Cross-Thread Playwright Crashes in Dashboard
**The Problem:** When running Trowler crawls in the background via the web dashboard, clicking the "Stop" button can cause cross-thread exceptions if the daemon thread is actively awaiting a Playwright navigation. 
**Current Solution:** We wrapped Playwright shutdown sequences (`close_browser()`) in `try/except` blocks to catch Cross-Thread violations and prevent the Flask server from crashing, but the background crawler thread may still leave zombie Chromium processes.
**Help Wanted:** Migrating `browser_fetcher.py` from `sync_playwright` to `async_playwright` to properly handle graceful termination signals across the Flask event loop.

---

## 🤝 Contributing
The MMFIC framework is licensed under the **AGPLv3**. If you build a commercial Threat Intelligence dashboard or OSINT service using this code, you are legally required to open-source your modifications.

**How to contribute:**
1. Fork the repository.
2. Check the issues above or the GitHub Issues tab.
3. Submit a Pull Request. If you are submitting an adapter for a new platform, please ensure it utilizes the `browser_fetcher.py` infrastructure rather than raw `requests` to maintain Tor routing integrity.
