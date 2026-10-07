"""
Proximity-based post extractor — the universal fallback parser.

Instead of relying on CSS selectors or container structures,
this parser works by:
  1. Finding ALL short text elements that look like usernames
  2. Finding ALL long text elements that look like post content
  3. Matching them by DOM proximity (nearest username to each post)

This handles the common case where the username is NOT inside the
same container as the post — e.g., it's in a sibling div, a parent
element, or a table cell next to the content cell.
"""

from bs4 import BeautifulSoup, Tag
import re

# ── Noise patterns to exclude ────────────────────────────────────
_NOISE_RE = re.compile(
    r"(cookie|privacy|terms of service|copyright|all rights reserved|"
    r"powered by|admin@|sign up|sign in|log in|register|"
    r"forgot password|navigation|breadcrumb|sidebar|advertisement|"
    r"subscribe|newsletter|jump to|report|quote|share|permalink|"
    r"reply with quote|post reply|new topic)",
    re.I
)

_NAV_TAGS = {"nav", "header", "footer", "aside", "noscript", "script", "style"}

# Patterns that indicate a username element
_USERNAME_CLASS_RE = re.compile(
    r"user|author|poster|member|nick|handle|name|identity|profile|"
    r"hover-card|display.?name|byline",
    re.I
)

# Patterns that indicate a post content element
_CONTENT_CLASS_RE = re.compile(
    r"post|message|content|body|text|comment|reply|entry|article|"
    r"bbcode|description",
    re.I
)

# Patterns that indicate a timestamp element
_TIME_CLASS_RE = re.compile(
    r"time|date|timestamp|posted|when|ago|created|published",
    re.I
)

# URL patterns that contain usernames
_PROFILE_HREF_RE = re.compile(
    r"/(?:u|user|profile|members?|author|people)/"
    r"([^/?#&]+)",
    re.I
)


def remove_noise_elements(soup):
    for noise in soup.find_all(list(_NAV_TAGS)):
        noise.decompose()

def extract_by_proximity(html, url):
    """Extract posts by finding username-content pairs via DOM proximity.
    
    Returns:
        list of dicts with username, post, timestamp, source
    """
    soup = BeautifulSoup(html, "html.parser")
    
    remove_noise_elements(soup)
    
    data = _extract_from_repeating_wrappers(soup, url)
    if len(data) >= 2:
        real = [d for d in data if d["username"] != "unknown"]
        if len(real) >= 2:
            return data
    
    proximity_data = _extract_by_dom_proximity(soup, url)
    if len(proximity_data) > len(data):
        data = proximity_data
    
    table_data = _extract_from_tables(soup, url)
    if len(table_data) > len(data):
        data = table_data
    
    return data


def group_children_by_signature(parent):
    child_groups = {}
    for child in parent.children:
        if not isinstance(child, Tag):
            continue
        if child.name in _NAV_TAGS:
            continue
        
        classes = child.get("class", [])
        sig = f"{child.name}:{classes[0] if classes else ''}"
        if sig not in child_groups:
            child_groups[sig] = []
        child_groups[sig].append(child)
    return child_groups


def score_child_group(children):
    if len(children) < 2:
        return 0
    
    score = 0
    sample = children[0]
    sample_text = sample.get_text(strip=True)
    
    if len(sample_text.split()) < 5:
        return 0
    
    if sample.find(True, class_=_USERNAME_CLASS_RE):
        score += 3
    if sample.find("a", href=_PROFILE_HREF_RE):
        score += 3
    
    if sample.find(True, class_=_CONTENT_CLASS_RE):
        score += 2
    
    if sample.find("time") or sample.find(True, class_=_TIME_CLASS_RE):
        score += 1
    
    score += min(len(children), 5)
    return score


def find_best_repeating_wrapper_group(soup):
    best_group = []
    best_score = 0
    
    for parent in soup.find_all(True):
        if parent.name in _NAV_TAGS or parent.name in ["html", "head", "body"]:
            continue
        
        child_groups = group_children_by_signature(parent)
        
        for sig, children in child_groups.items():
            score = score_child_group(children)
            if score > best_score:
                best_score = score
                best_group = children
                
    return best_group


def _extract_from_repeating_wrappers(soup, url):
    """Find the largest group of repeating sibling elements that look like posts.
    
    Instead of matching specific class names, finds ALL groups of same-tag
    siblings and scores them by how much they look like post containers.
    """
    data = []
    best_group = find_best_repeating_wrapper_group(soup)
    
    for container in best_group:
        username = _find_username_deep(container)
        post_text = _find_post_content(container)
        timestamp = _find_timestamp_deep(container)
        
        if post_text and len(post_text.split()) >= 5:
            data.append({
                "username": username,
                "post": post_text[:2000],
                "timestamp": timestamp,
                "source": url,
                "is_preview": False,
            })
    
    return data


def get_username_from_element(elem):
    classes_str = " ".join(elem.get("class", []))
    if _USERNAME_CLASS_RE.search(classes_str):
        text = elem.get_text(strip=True)
        if text and len(text) < 40 and "\n" not in text and text.count(" ") < 3:
            return text
            
    if elem.name == "a" and elem.get("href"):
        match = _PROFILE_HREF_RE.search(elem["href"])
        if match:
            text = elem.get_text(strip=True)
            if text and len(text) < 40:
                return text
            elif match.group(1):
                return match.group(1)
                
    for attr in ["data-username", "data-author", "data-user", "data-name",
                 "data-display-name", "data-poster"]:
        val = elem.get(attr)
        if val and len(val) < 50:
            return val
            
    return None


def find_username_candidates(soup):
    username_candidates = []
    for elem in soup.find_all(True):
        if elem.name in _NAV_TAGS:
            continue
        
        username = get_username_from_element(elem)
        if username:
            pos = _get_element_position(elem)
            username_candidates.append((pos, username, elem))
            
    return username_candidates


def is_valid_post_candidate(elem):
    if elem.name in _NAV_TAGS:
        return False
        
    if elem.name == "div" and elem.find(["p", "div", "table", "article"]):
        classes_str = " ".join(elem.get("class", []))
        if not _CONTENT_CLASS_RE.search(classes_str):
            return False
            
    text = elem.get_text(strip=True)
    if len(text.split()) < 10:
        return False
    if _NOISE_RE.search(text[:100]):
        return False
        
    return True


def find_post_candidates(soup):
    post_candidates = []
    for elem in soup.find_all(["p", "div", "td", "article", "section"]):
        if not is_valid_post_candidate(elem):
            continue
            
        text = elem.get_text(strip=True)
        pos = _get_element_position(elem)
        post_candidates.append((pos, text[:2000], elem))
        
    return _deduplicate_posts(post_candidates)


def find_closest_username(post_pos, username_candidates, used_usernames):
    best_username = "unknown"
    best_distance = float("inf")
    best_idx = -1
    
    for i, (user_pos, username, user_elem) in enumerate(username_candidates):
        if i in used_usernames:
            continue
        
        distance = post_pos - user_pos
        if -5 <= distance <= 200 and abs(distance) < best_distance:
            best_distance = abs(distance)
            best_username = username
            best_idx = i
            
    return best_username, best_idx


def match_posts_to_usernames(post_candidates, username_candidates, url):
    data = []
    used_usernames = set()
    
    for post_pos, post_text, post_elem in post_candidates:
        best_username, best_idx = find_closest_username(post_pos, username_candidates, used_usernames)
        
        if best_idx >= 0:
            used_usernames.add(best_idx)
        
        timestamp = _find_timestamp_near(post_elem)
        
        data.append({
            "username": best_username,
            "post": post_text,
            "timestamp": timestamp,
            "source": url,
            "is_preview": False,
        })
        
    return data


def _extract_by_dom_proximity(soup, url):
    """Find usernames and post content independently, then match by position.
    
    This handles the case where username and post are in completely
    separate elements that share no common container.
    """
    username_candidates = find_username_candidates(soup)
    post_candidates = find_post_candidates(soup)
    return match_posts_to_usernames(post_candidates, username_candidates, url)


def extract_username_from_table_cell(cell):
    cell_text = cell.get_text(strip=True)
    if len(cell_text.split()) <= 5:
        link = cell.find("a", href=_PROFILE_HREF_RE)
        if link:
            return link.get_text(strip=True) or "unknown"
        
        classes_str = " ".join(cell.get("class", []))
        if _USERNAME_CLASS_RE.search(classes_str) and cell_text:
            return cell_text
    return None

def extract_post_from_table_cell(cell):
    cell_text = cell.get_text(strip=True)
    if len(cell_text.split()) >= 10 and not _NOISE_RE.search(cell_text[:100]):
        return cell_text[:2000]
    return None


def extract_post_from_table_row(row, url):
    cells = row.find_all(["td", "th"])
    if len(cells) < 2:
        return None
    
    username = "unknown"
    post_text = ""
    timestamp = "N/A"
    
    for cell in cells:
        uname = extract_username_from_table_cell(cell)
        if uname:
            username = uname
            continue
            
        ptext = extract_post_from_table_cell(cell)
        if ptext:
            post_text = ptext
            ts = _find_timestamp_deep(cell)
            if ts != "N/A":
                timestamp = ts
    
    if post_text and len(post_text.split()) >= 10:
        return {
            "username": username,
            "post": post_text,
            "timestamp": timestamp,
            "source": url,
            "is_preview": False,
        }
    return None


def _extract_from_tables(soup, url):
    """Extract from table-based forum layouts.
    
    Many older forums and darkweb sites use <table> for post layout:
      <tr>
        <td>username</td>
        <td>post content</td>
      </tr>
    """
    data = []
    
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue
        
        for row in rows:
            post_data = extract_post_from_table_row(row, url)
            if post_data:
                data.append(post_data)
                
    return data


# ── Helper functions ─────────────────────────────────────────────

def _try_deep_username_links(container):
    for a in container.find_all("a", href=True):
        match = _PROFILE_HREF_RE.search(a["href"])
        if match:
            text = a.get_text(strip=True)
            if text and len(text) < 40 and "\n" not in text:
                return text
            if match.group(1):
                return match.group(1)
    return None

def _try_deep_username_classes(container):
    found = container.find(True, class_=_USERNAME_CLASS_RE)
    if found:
        text = found.get_text(strip=True)
        if text and len(text) < 40 and "\n" not in text and text.count(" ") < 3:
            return text
    return None

def _try_deep_username_attributes(container):
    for attr in ["data-username", "data-author", "data-user", "data-name",
                 "data-display-name", "data-poster"]:
        found = container.find(attrs={attr: True})
        if found:
            return found[attr]
    return None

def _try_deep_username_bold(container):
    first_bold = container.find(["strong", "b"])
    if first_bold:
        text = first_bold.get_text(strip=True)
        if text and len(text) < 30 and "\n" not in text and text.count(" ") < 2:
            container_text = container.get_text(strip=True)
            if container_text.index(text) < 200 if text in container_text else False:
                return text
    return None


def _find_username_deep(container):
    """Deep search for a username within a container element.
    
    Tries multiple strategies in order of reliability.
    """
    res = _try_deep_username_links(container)
    if res: return res
    
    res = _try_deep_username_classes(container)
    if res: return res
    
    res = _try_deep_username_attributes(container)
    if res: return res
    
    res = _try_deep_username_bold(container)
    if res: return res
    
    return "unknown"


def _try_find_post_content_class(container):
    found = container.find(True, class_=_CONTENT_CLASS_RE)
    if found:
        text = found.get_text(separator=" ", strip=True)
        if len(text.split()) >= 10 and not _NOISE_RE.search(text[:100]):
            return text
    return None

def _try_find_post_content_longest_p(container):
    candidates = []
    for p in container.find_all("p"):
        text = p.get_text(strip=True)
        if len(text.split()) >= 5:
            candidates.append(text)
    
    if candidates:
        return max(candidates, key=len)
    return None

def _try_find_post_content_all_text(container):
    text = container.get_text(separator=" ", strip=True)
    if len(text.split()) >= 10 and not _NOISE_RE.search(text[:100]):
        return text
    return ""


def _find_post_content(container):
    """Find the main post content within a container.
    
    Prefers elements with content-like class names,
    falls back to the longest text block.
    """
    res = _try_find_post_content_class(container)
    if res: return res
    
    res = _try_find_post_content_longest_p(container)
    if res: return res
    
    return _try_find_post_content_all_text(container)


def _try_deep_timestamp_time(container):
    time_elem = container.find("time")
    if time_elem:
        return (time_elem.get("datetime") or 
                time_elem.get("title") or 
                time_elem.get_text(strip=True) or "N/A")
    return None

def _try_deep_timestamp_class(container):
    found = container.find(True, class_=_TIME_CLASS_RE)
    if found:
        text = found.get("datetime") or found.get("title") or found.get_text(strip=True)
        if text and len(text) < 100:
            return text
    return None

def _try_deep_timestamp_attributes(container):
    for attr in ["data-timestamp", "data-time", "data-date", "data-created"]:
        found = container.find(attrs={attr: True})
        if found:
            return found[attr]
    return None

def _try_deep_timestamp_titles(container):
    for elem in container.find_all(True, title=True):
        title = elem["title"]
        if re.search(r"\d{4}|\d{1,2}:\d{2}|ago|hour|minute|day|week|month", title, re.I):
            return title
    return None


def _find_timestamp_deep(container):
    """Deep search for a timestamp within a container."""
    res = _try_deep_timestamp_time(container)
    if res: return res
    
    res = _try_deep_timestamp_class(container)
    if res: return res
    
    res = _try_deep_timestamp_attributes(container)
    if res: return res
    
    res = _try_deep_timestamp_titles(container)
    if res: return res
    
    return "N/A"


def _find_timestamp_near(post_elem):
    """Find a timestamp near a post element (siblings, parent)."""
    ts = _find_timestamp_deep(post_elem)
    if ts != "N/A":
        return ts
    
    sibling = post_elem.find_next_sibling()
    if sibling and isinstance(sibling, Tag):
        ts = _find_timestamp_deep(sibling)
        if ts != "N/A":
            return ts
    
    sibling = post_elem.find_previous_sibling()
    if sibling and isinstance(sibling, Tag):
        ts = _find_timestamp_deep(sibling)
        if ts != "N/A":
            return ts
    
    if post_elem.parent and isinstance(post_elem.parent, Tag):
        ts = _find_timestamp_deep(post_elem.parent)
        if ts != "N/A":
            return ts
    
    return "N/A"


def _get_element_position(elem):
    """Get a numeric position for an element in the document.
    
    Uses the element's source position (line number) as a proxy.
    Falls back to counting preceding elements.
    """
    if hasattr(elem, 'sourceline') and elem.sourceline:
        return elem.sourceline
    
    pos = 0
    for sibling in elem.previous_siblings:
        pos += 1
    if elem.parent and isinstance(elem.parent, Tag):
        pos += _get_element_position(elem.parent) * 1000
    return pos


def _deduplicate_posts(post_candidates):
    """Remove overlapping post candidates (when one is a subset of another)."""
    if len(post_candidates) < 2:
        return post_candidates
    
    sorted_posts = sorted(post_candidates, key=lambda x: len(x[1]), reverse=True)
    
    kept = []
    kept_texts = set()
    
    for pos, text, elem in sorted_posts:
        text_key = text[:100]
        is_duplicate = False
        for kept_text in kept_texts:
            if text_key in kept_text or kept_text in text_key:
                is_duplicate = True
                break
        
        if not is_duplicate:
            kept.append((pos, text, elem))
            kept_texts.add(text_key)
    
    kept.sort(key=lambda x: x[0])
    return kept
