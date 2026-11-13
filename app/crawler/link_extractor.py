from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import re


def is_thread_link(link, base_url):
    """
    Detect if a link is a forum thread (not navigation/login/etc).
    Checks common forum URL patterns.
    """

    thread_indicators = [
        "/thread/", "/threads/", "/topic/", "/topics/",
        "/viewtopic", "/showthread", "/post/", "/t/",
        "/discussion/", "/forum/", "?tid=", "?t=",
        "/comments/",
    ]

    skip_indicators = [
        "/login", "/register", "/logout", "/search",
        "/profile", "/user/", "/admin", "/mod",
        ".jpg", ".png", ".gif", ".pdf", ".zip",
        "javascript:", "mailto:", "#"
    ]

    link_lower = link.lower()

    for skip in skip_indicators:
        if skip in link_lower:
            return False

    for indicator in thread_indicators:
        if indicator in link_lower:
            return True

    parsed = urlparse(link)
    base_parsed = urlparse(base_url)

    if parsed.netloc != base_parsed.netloc:
        return False

    if re.search(r'/\d+', parsed.path):
        return True

    return False


def extract_thread_links(html, base_url):
    """
    From an index/listing page, extract links that look like threads.
    Returns list of full URLs.
    """
    soup = BeautifulSoup(html, "html.parser")
    thread_links = []
    base_domain = urlparse(base_url).netloc

    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"])

        if urlparse(href).netloc == base_domain:
            if is_thread_link(href, base_url):
                if href not in thread_links:
                    thread_links.append(href)

    print(f"[+] Found {len(thread_links)} thread links on {base_url}")
    return thread_links