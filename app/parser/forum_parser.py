from bs4 import BeautifulSoup, Tag
import re

# ── Noise patterns to exclude from content extraction ────────────
_NOISE_PATTERNS = re.compile(
    r"(cookie|privacy|terms of service|copyright|all rights reserved|"
    r"powered by|wordpress|admin@|sign up|sign in|log in|register|"
    r"forgot password|navigation|breadcrumb|sidebar|advertisement|"
    r"subscribe|newsletter|footer|header)",
    re.I
)

_NAV_TAGS = {"nav", "header", "footer", "aside", "noscript"}


# ── Index page parser (finds thread links, NO content extraction) ─
def try_find_index_posts(soup):
    selectors = [
        ("div", "post"),
        ("div", "thread"),
        ("div", "topic"),
        ("li", "thread"),
        ("tr", "thread"),
    ]
    for tag, cls in selectors:
        posts = soup.find_all(tag, class_=re.compile(cls, re.I))
        if posts:
            return posts
    return []


def parse_forum(html, url):
    """
    Parse a forum INDEX page.
    Returns thread preview data + marks page as index type.
    Only used for thread discovery — previews are lightweight.
    """
    soup = BeautifulSoup(html, "html.parser")
    data = []

    posts = try_find_index_posts(soup)

    for post in posts:
        try:
            username  = _find_username(post)
            content   = _find_content(post)
            timestamp = _find_timestamp(post)

            if content and len(content.split()) >= 5:
                data.append({
                    "username":  username,
                    "post":      content[:500],
                    "timestamp": timestamp,
                    "source":    url,
                    "is_preview": True
                })
        except Exception:
            continue

    return data


# ── Thread page parser (extracts full post bodies) ───────────────
def _extract_posts_from_containers(containers, url):
    data = []
    for container in containers:
        try:
            username  = _find_username(container)
            content   = _find_content(container)
            timestamp = _find_timestamp(container)

            if not content or len(content.split()) < 10:
                continue
            if _NOISE_PATTERNS.search(content[:100]):
                continue

            data.append({
                "username":  username,
                "post":      content[:2000],
                "timestamp": timestamp,
                "source":    url,
                "is_preview": False
            })
        except Exception:
            continue
    return data


def parse_thread_page(html, url):
    """
    Parse a THREAD page — a single discussion thread.
    Returns ALL posts in the thread with full content.
    
    Strategy:
      1. Score candidate container families by structural quality
      2. Extract username/content/timestamp from best containers
      3. Fallback to structural text-block sweep (cleaned)
    """
    soup = BeautifulSoup(html, "html.parser")
    
    _strip_noise_tags(soup)
    
    data = []

    post_containers = _find_post_containers_scored(soup)

    if post_containers:
        data = _extract_posts_from_containers(post_containers, url)

    if len(data) < 2:
        structural = _parse_thread_structural(soup, url)
        if len(structural) > len(data):
            data = structural

    return data


def _strip_noise_tags(soup):
    """Remove nav/footer/script/style elements from soup (in-place).
    
    NOTE: We do NOT strip <header> tags because many forum platforms
    (Lemmy, Kbin, etc.) place the username inside a <header> within
    each post container. Stripping them destroys username extraction.
    """
    for noise in soup.find_all(["nav", "footer", "aside",
                                 "script", "style", "noscript"]):
        noise.decompose()


def _score_container_family(containers):
    score = 0
    sample = containers[0]
    
    if _has_username_child(sample):
        score += 2
    
    if _has_content_child(sample):
        score += 2
    
    if sample.find("time") or sample.find(True, class_=re.compile(r"time|date", re.I)):
        score += 1
    
    if len(containers) >= 5:
        score += 1
        
    return score


def _find_post_containers_scored(soup):
    """Find post containers using a scoring system.
    
    Scores each candidate tag family by:
      - Has child with username-like class (+2)
      - Has child with content-like class (+2)
      - Has child with time/date element (+1)
      - Number of repeating instances (≥3 required)
    
    Returns containers from the highest-scoring family.
    """
    
    patterns = [
        ("div",     r"post(?!-header|-footer|-meta|-count|-icon)"),
        ("div",     r"message(?!-header|-footer|-meta)"),
        ("div",     r"comment(?!-header|-footer|-meta)"),
        ("div",     r"reply"),
        ("li",      r"post"),
        ("li",      r"comment"),
        ("div",     r"content-wrap"),
        ("div",     r"forum-post"),
        ("div",     r"thread-post"),
    ]
    
    best_containers = []
    best_score = 0
    
    for tag, pattern in patterns:
        containers = soup.find_all(tag, class_=re.compile(pattern, re.I))
        if len(containers) < 2:
            continue
        
        score = _score_container_family(containers)
        if score > best_score:
            best_score = score
            best_containers = containers
    
    for attr in ["data-post-id", "data-message-id", "data-comment-id"]:
        containers = soup.find_all(attrs={attr: True})
        if len(containers) >= 2:
            score = _score_container_family(containers)
            if score > best_score:
                best_score = score
                best_containers = containers
    
    return best_containers


def _has_username_child(container):
    """Check if container has a child element with username-like class."""
    username_pattern = re.compile(r"username|user-name|author|poster|handle|nick|member-name|hover-card-link|\bname\b", re.I)
    if container.find(True, class_=username_pattern) is not None:
        return True
    if container.find("a", href=re.compile(r"/u/|/user/|/profile/", re.I)):
        return True
    return False


def _has_content_child(container):
    """Check if container has a child element with content-like class."""
    content_pattern = re.compile(r"post-body|post-content|message-body|content|body-text|entry-content|bbcode|reply-body", re.I)
    return container.find(True, class_=content_pattern) is not None


def _try_username_selectors(tag):
    username_selectors = [
        ("*",    re.compile(r"username|user-name|author-name|poster-name|handle|nick", re.I)),
        ("a",    re.compile(r"hover-card-link|username|user-link|author|profile|member", re.I)),
        ("span", re.compile(r"username|user|author|name", re.I)),
        ("div",  re.compile(r"username|author-info", re.I)),
    ]
    
    for tag_name, cls_pattern in username_selectors:
        found = tag.find(tag_name, class_=cls_pattern)
        if found:
            text = found.get_text(strip=True)
            if text and len(text) < 40 and '\n' not in text and text.count(' ') < 3:
                return text
    return None


def _try_username_href(tag):
    href_patterns = [
        re.compile(r"/u/", re.I),
        re.compile(r"/user/", re.I),
        re.compile(r"/profile/", re.I),
        re.compile(r"/members?/", re.I),
    ]
    for pattern in href_patterns:
        found = tag.find("a", href=pattern)
        if found:
            text = found.get_text(strip=True)
            if text and len(text) < 40 and '\n' not in text and text.count(' ') < 3:
                return text
            href = found["href"]
            for url_pattern in [r"/u/([^/?#]+)", r"/user/([^/?#]+)", r"/profile/([^/?#]+)", r"/members?/([^/?#]+)"]:
                match = re.search(url_pattern, href)
                if match:
                    return match.group(1).strip()
    return None


def _try_username_attributes(tag):
    for attr in ["data-username", "data-author", "data-user", "data-name",
                 "data-display-name", "data-poster"]:
        found = tag.find(attrs={attr: True})
        if found:
            return found[attr]
    return None


def _try_username_headings(tag):
    for wrapper_class in [re.compile(r"user|author|poster|member", re.I)]:
        wrapper = tag.find(True, class_=wrapper_class)
        if wrapper:
            for heading in wrapper.find_all(["h4", "h5", "strong"]):
                text = heading.get_text(strip=True)
                if text and len(text) < 40 and '\n' not in text:
                    return text
    return None


def _find_username(tag):
    """Extract a username from a post block.
    
    Enhanced with:
      - hover-card-link and href-based username extraction
      - More data attributes (data-name, data-display-name, data-poster)
      - Must be short, single-line text
    """
    res = _try_username_selectors(tag)
    if res: return res
    
    res = _try_username_href(tag)
    if res: return res
    
    res = _try_username_attributes(tag)
    if res: return res
    
    res = _try_username_headings(tag)
    if res: return res
    
    return "unknown"


def _try_content_selectors(tag):
    content_selectors = [
        ("div",  re.compile(r"post-body|post-content|message-body|message-content", re.I)),
        ("div",  re.compile(r"bbcode|post-text|entry-content|reply-body|postcontent", re.I)),
        ("td",   re.compile(r"post-body|message|postcontent", re.I)),
        ("div",  re.compile(r"postbody|messagetext|body-text", re.I)),
    ]
    
    for tag_name, cls_pattern in content_selectors:
        found = tag.find(tag_name, class_=cls_pattern)
        if found:
            text = found.get_text(separator=" ", strip=True)
            if len(text.split()) >= 10 and not _NOISE_PATTERNS.search(text[:100]):
                return text
    return None


def _try_content_paragraphs(tag):
    candidates = []
    for elem in tag.find_all("p"):
        if _is_inside_noise(elem):
            continue
        text = elem.get_text(strip=True)
        if len(text.split()) >= 10 and not _NOISE_PATTERNS.search(text[:100]):
            candidates.append(text)
    
    if candidates:
        return max(candidates, key=len)
    return ""


def _find_content(tag):
    """Extract the main post body text.
    
    Tightened rules:
      - Must have ≥10 words
      - Must not match noise patterns
      - Fallback only looks at elements that are siblings of username elements
    """
    res = _try_content_selectors(tag)
    if res: return res
    
    return _try_content_paragraphs(tag)


def _is_inside_noise(elem):
    """Check if element is nested inside a nav/footer/sidebar."""
    for parent in elem.parents:
        if not isinstance(parent, Tag):
            continue
        parent_classes = " ".join(parent.get("class", []))
        if parent.name in _NAV_TAGS:
            return True
        if re.search(r"nav|footer|sidebar|menu|breadcrumb|header", parent_classes, re.I):
            return True
    return False


def _try_timestamp_selectors(tag):
    time_selectors = [
        ("time",  None),         # HTML5 time element
        ("span",  re.compile(r"time|date|timestamp|posted|when|ago", re.I)),
        ("div",   re.compile(r"time|date|timestamp", re.I)),
        ("abbr",  re.compile(r"time|date", re.I)),
        ("small", re.compile(r"time|date|posted", re.I)),
    ]
    
    for tag_name, cls_pattern in time_selectors:
        if cls_pattern:
            found = tag.find(tag_name, class_=cls_pattern)
        else:
            found = tag.find(tag_name)
        
        if found:
            dt = found.get("datetime") or found.get("title") or found.get_text(strip=True)
            if dt:
                return dt
    return None


def _try_timestamp_attributes(tag):
    for attr in ["data-timestamp", "data-time", "data-date", "data-created"]:
        found = tag.find(attrs={attr: True})
        if found:
            return found[attr]
    return None


def _try_timestamp_titles(tag):
    for a_tag in tag.find_all("a", title=True):
        title = a_tag["title"]
        if re.search(r"\d{4}|\d{1,2}:\d{2}|ago|hour|minute|day|week|month", title, re.I):
            return title
            
    for span in tag.find_all("span", title=True):
        title = span["title"]
        if re.search(r"\d{4}|\d{1,2}:\d{2}|ago|hour|minute|day|week|month", title, re.I):
            return title
    return None


def _find_timestamp(tag):
    """Extract timestamp from a post block.
    
    Enhanced with:
      - Data attributes (data-timestamp, data-time, data-date)
      - Title attributes on links (hover timestamps)
      - Broader class pattern matching
    """
    res = _try_timestamp_selectors(tag)
    if res: return res
    
    res = _try_timestamp_attributes(tag)
    if res: return res
    
    res = _try_timestamp_titles(tag)
    if res: return res
    
    return "N/A"


def is_valid_structural_block(elem):
    if elem.find(["p", "div", "table"]):
        return False
    
    if _is_inside_noise(elem):
        return False
        
    text = elem.get_text(strip=True)
    if len(text.split()) < 10:
        return False
    if _NOISE_PATTERNS.search(text[:100]):
        return False
        
    return True


def _parse_thread_structural(soup, url):
    """
    Last-resort thread parser: finds text blocks that look like posts.
    
    Tightened from original:
      - Skips blocks that match noise patterns
      - Requires ≥10 words per block
      - Caps at 20 blocks (not 30)
      - Skips elements inside nav/footer/sidebar
    """
    data = []
    seen_texts = set()
    
    for elem in soup.find_all(["p", "div", "td"]):
        if not is_valid_structural_block(elem):
            continue
        
        text = elem.get_text(strip=True)
        text_key = text[:100]
        if text_key in seen_texts:
            continue
        seen_texts.add(text_key)
        
        data.append({
            "username":   "unknown",
            "post":       text[:2000],
            "timestamp":  "N/A",
            "source":     url,
            "is_preview": False
        })
        
        if len(data) >= 20:
            break
    
    return data