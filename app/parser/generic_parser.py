from bs4 import BeautifulSoup
import re
import hashlib

# Noise patterns — skip text that matches these
_NOISE_PATTERNS = re.compile(
    r"(cookie|privacy|terms of service|copyright|all rights reserved|"
    r"powered by|wordpress|admin@|sign up|sign in|log in|register|"
    r"forgot password|navigation|breadcrumb|sidebar|advertisement|"
    r"subscribe|newsletter)",
    re.I
)

def remove_noise_tags(soup):
    for noise in soup.find_all(["nav", "header", "footer", "aside",
                                 "script", "style", "noscript"]):
        noise.decompose()

def is_leaf_element(tag):
    if tag.find(["p", "div", "table"]):
        return False
    return True

def is_quality_text(text):
    words = text.split()
    if len(words) < 30:
        return False
    if _NOISE_PATTERNS.search(text[:150]):
        return False
    return True

def calculate_content_hash(text):
    return hashlib.md5(text[:200].encode()).hexdigest()

def parse_generic(html, url):
    """Generic fallback parser with quality gates.
    
    Only extracts text blocks that could plausibly be forum posts:
      - Must have ≥30 words (not just 80 chars)
      - Must not match noise patterns
      - Deduplicated by content hash
    """
    soup = BeautifulSoup(html, "html.parser")
    
    remove_noise_tags(soup)
    
    data = []
    seen = set()

    for tag in soup.find_all(["p", "div"]):
        if not is_leaf_element(tag):
            continue
        
        text = tag.get_text(strip=True)
        if not is_quality_text(text):
            continue
        
        content_key = calculate_content_hash(text)
        if content_key in seen:
            continue
        seen.add(content_key)

        data.append({
            "username": "unknown",
            "post": text[:2000],
            "timestamp": "N/A",
            "source": url
        })

    return data
