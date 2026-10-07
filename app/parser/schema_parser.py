"""
Schema-based parser — uses known CSS selectors from known_schemas.json.

This parser is tried FIRST in the pipeline, before generic pattern matching.
If we already know the exact CSS selectors for a forum (from known_schemas.json
or schema_cache.json), there's no reason to guess with heuristics.

This is the most reliable parser for known forums.
"""

import json
import os
import re
from bs4 import BeautifulSoup
from urllib.parse import urlparse


# Load schemas once at import time
_SCHEMAS = {}
_SCHEMA_FILES = [
    os.path.join(os.path.dirname(__file__), "known_schemas.json"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "schema_cache.json"),
]

for schema_file in _SCHEMA_FILES:
    if os.path.exists(schema_file):
        try:
            with open(schema_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                _SCHEMAS.update(data)
        except Exception:
            pass

if _SCHEMAS:
    print(f"[+] Schema parser loaded {len(_SCHEMAS)} known forum schemas")


def get_schema_for_url(url):
    """Get the known schema for a URL's domain, if one exists."""
    domain = urlparse(url).netloc
    
    if domain in _SCHEMAS:
        return _SCHEMAS[domain]
    
    bare = domain.replace("www.", "")
    if bare in _SCHEMAS:
        return _SCHEMAS[bare]
    
    for key in _SCHEMAS:
        if key in domain or domain in key:
            return _SCHEMAS[key]
    
    return None


def get_profile_schema_for_url(url):
    """Get the known profile schema for a URL's domain, if one exists."""
    schema = get_schema_for_url(url)
    if schema and "profile_schema" in schema:
        return schema["profile_schema"]
    return None


def find_containers(soup, container_sel):
    if not container_sel:
        return []
        
    containers = _css_select_safe(soup, container_sel)
    
    if not containers:
        print(f"[-] Schema containers not found: {container_sel}")
        fallback_sel = container_sel.rsplit(".", 1)[0] if "." in container_sel else ""
        if fallback_sel:
            containers = _css_select_safe(soup, fallback_sel)
            if containers:
                print(f"[+] Fallback container matched: {fallback_sel} ({len(containers)} found)")
                
    return containers


def extract_post_from_container(container, username_sel, post_sel, timestamp_sel, url):
    username = _extract_with_selector(container, username_sel)
    post_text = _extract_with_selector(container, post_sel)
    timestamp = _extract_with_selector(container, timestamp_sel)
    
    if not username or username == "unknown":
        username = _extract_username_from_links(container)
    
    if not username or username == "unknown":
        username = _extract_username_from_sibling(container)
    
    if not post_text or len(post_text.split()) < 3:
        return None
    
    return {
        "username": username if username else "unknown",
        "post": post_text[:2000],
        "timestamp": timestamp if timestamp else "N/A",
        "source": url,
        "is_preview": False,
    }


def parse_with_schema(html, url):
    """Parse a thread page using known CSS selectors.
    
    Returns a list of posts, or empty list if no schema exists for this domain
    or if the selectors don't match the page.
    """
    schema = get_schema_for_url(url)
    if not schema:
        return []
    
    domain = urlparse(url).netloc
    print(f"[+] Using known schema for {domain}")
    
    soup = BeautifulSoup(html, "html.parser")
    
    container_sel = schema.get("container_selector", "")
    username_sel = schema.get("username_selector", "")
    post_sel = schema.get("post_selector", "")
    timestamp_sel = schema.get("timestamp_selector", "")
    
    containers = find_containers(soup, container_sel)
    
    if not containers:
        return []
    
    print(f"[+] Found {len(containers)} containers with schema selector")
    
    data = []
    for container in containers:
        post = extract_post_from_container(container, username_sel, post_sel, timestamp_sel, url)
        if post:
            data.append(post)
    
    if not data and containers:
        data = _extract_all_pairs(soup, username_sel, post_sel, timestamp_sel, url)
    
    if data:
        real = [d for d in data if d.get("username", "unknown") != "unknown"]
        print(f"[+] Schema parser extracted {len(data)} posts ({len(real)} with usernames)")
    
    return data


def _css_select_safe(soup, selector):
    """Run CSS select, handling multiple comma-separated selectors."""
    try:
        return soup.select(selector)
    except Exception:
        results = []
        for sel in selector.split(","):
            sel = sel.strip()
            if sel:
                try:
                    results.extend(soup.select(sel))
                except Exception:
                    pass
        return results


def _extract_with_selector(container, selector):
    """Extract text from the first matching element within a container."""
    if not selector:
        return ""
    
    for sel in selector.split(","):
        sel = sel.strip()
        if not sel:
            continue
        try:
            found = container.select_one(sel)
            if found:
                text = found.get_text(strip=True)
                if text:
                    return text
                
                img = found.find("img") if hasattr(found, "find") else None
                if img and img.get("alt"):
                    return img["alt"].strip()
                
                if found.name == "a" and found.get("href"):
                    href = found["href"]
                    import urllib.parse
                    parsed = urllib.parse.urlparse(href)
                    parts = [p for p in parsed.path.split("/") if p]
                    if parts and parts[-1].lower() not in ("profile", "user", "u", "members", "member"):
                        return parts[-1].strip()
        except Exception:
            continue
    
    return ""


def _extract_username_from_links(container):
    """Extract username from profile links within the container.
    
    Looks for <a> tags with href patterns like /u/username, /user/username, etc.
    This is the most reliable method for Reddit/Lemmy-like forums.
    """
    profile_patterns = [
        re.compile(r"/u/([^/?#]+)"),
        re.compile(r"/user/([^/?#]+)"),
        re.compile(r"/profile/([^/?#]+)"),
        re.compile(r"/members?/([^/?#]+)"),
    ]
    
    for a in container.find_all("a", href=True):
        href = a.get("href", "")
        for pattern in profile_patterns:
            match = pattern.search(href)
            if match:
                username = match.group(1).strip()
                if username and len(username) < 40:
                    return username
        
        classes = " ".join(a.get("class", []))
        if "hover-card" in classes or "user" in classes.lower() or "author" in classes.lower():
            text = a.get_text(strip=True)
            if text and len(text) < 40 and "\n" not in text:
                return text
    
    return ""


def _extract_username_from_sibling(container):
    """Check the previous sibling element for username info.
    
    Some forums have the user info in a sibling div, not inside the post container.
    """
    prev = container.find_previous_sibling()
    if not prev:
        return ""
    
    username = _extract_username_from_links(prev)
    if username:
        return username
    
    username_pattern = re.compile(r"username|user-name|author|poster|name", re.I)
    found = prev.find(True, class_=username_pattern)
    if found:
        text = found.get_text(strip=True)
        if text and len(text) < 40 and "\n" not in text:
            return text
    
    return ""


def _extract_all_pairs(soup, username_sel, post_sel, timestamp_sel, url):
    """Fallback: find all usernames and posts on the page independently."""
    data = []
    
    usernames = _css_select_safe(soup, username_sel) if username_sel else []
    posts = _css_select_safe(soup, post_sel) if post_sel else []
    timestamps = _css_select_safe(soup, timestamp_sel) if timestamp_sel else []
    
    for i, post_el in enumerate(posts):
        post_text = post_el.get_text(strip=True)
        if not post_text or len(post_text.split()) < 3:
            continue
        
        username = "unknown"
        if i < len(usernames):
            username = usernames[i].get_text(strip=True) or "unknown"
        
        timestamp = "N/A"
        if i < len(timestamps):
            timestamp = timestamps[i].get_text(strip=True) or "N/A"
        
        data.append({
            "username": username,
            "post": post_text[:2000],
            "timestamp": timestamp,
            "source": url,
            "is_preview": False,
        })
    
    return data
