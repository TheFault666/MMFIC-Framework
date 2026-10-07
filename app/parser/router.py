from app.parser.classifier import classify_page
from app.parser.forum_parser import parse_forum, parse_thread_page
from app.parser.blog_parser import parse_blog
from app.parser.generic_parser import parse_generic
from app.parser.proximity_parser import extract_by_proximity
from app.parser.schema_parser import parse_with_schema
from app.parser.llm_parser import parse_with_llm, LLMQuotaExceeded
from app.parser.post_processor import post_process
import json


def is_json_response(html: str) -> bool:
    cleaned = html.strip()
    return cleaned.startswith("{") or cleaned.startswith("[")


def try_parse_json(html: str, url: str):
    if is_json_response(html):
        try:
            json_data = json.loads(html)
            return _parse_raw_json(json_data, url)
        except Exception:
            pass
    return None


def extract_data(html, url):
    """For INDEX pages — lightweight extraction, NO LLM calls.
    
    Index pages are used for thread discovery, not for ML data.
    Calling the LLM here was wasteful since index previews
    are mostly nav text with 'unknown' usernames.
    """
    json_res = try_parse_json(html, url)
    if json_res is not None:
        return json_res

    page_type = classify_page(html, url)
    print(f"[+] Detected page type: {page_type}")

    if page_type == "forum":
        data = parse_forum(html, url)
    elif page_type in ["blog", "listing"]:
        data = parse_blog(html, url)
    else:
        data = parse_generic(html, url)

    if len(data) < 1 and page_type != "generic":
        print("[!] Weak extraction, trying generic parser...")
        data = parse_generic(html, url)

    return post_process(data)


def count_real_posts(data):
    return sum(1 for d in data if d.get("username", "unknown") != "unknown")


def try_schema_parser(html, url):
    data = parse_with_schema(html, url)
    return data, count_real_posts(data)


def try_manual_parser(html, url):
    data = parse_thread_page(html, url)
    return data, count_real_posts(data)


def try_proximity_parser(html, url):
    data = extract_by_proximity(html, url)
    return data, count_real_posts(data)


def try_llm_parser(html, url):
    llm_data = parse_with_llm(html, url)
    return llm_data or [], count_real_posts(llm_data or [])


def should_use_llm(data, real_posts_count):
    if len(data) < 2:
        return True, "too few posts extracted"
    if len(data) >= 2 and real_posts_count == 0:
        return True, "got content but no usernames"
    return False, ""


def extract_thread_data(html, url):
    """For THREAD pages: extract full post bodies.
    
    Pipeline:
      0. Schema parser (known CSS selectors — most reliable)
      1. Manual parser (CSS class-based heuristics)
      2. Proximity parser (DOM structure-based)
      3. LLM fallback (last resort)
    
    Raises:
        LLMQuotaExceeded: propagated from LLM parser — caller should stop crawl.
    """
    data, real_count = try_schema_parser(html, url)
    if len(data) >= 2 and real_count >= 1:
        print(f"[+] Schema parser succeeded: {len(data)} posts, {real_count} with usernames")
        return post_process(data)
    
    manual_data, manual_real = try_manual_parser(html, url)
    if manual_real > real_count or (len(manual_data) > len(data) and manual_real >= real_count):
        data = manual_data
        real_count = manual_real
    
    if len(data) >= 2 and real_count >= 2:
        return post_process(data)
    
    print("[!] Manual parsing weak — trying proximity-based extraction...")
    prox_data, prox_real = try_proximity_parser(html, url)
    
    if prox_real > real_count:
        print(f"[+] Proximity parser found {prox_real} usernames (vs {real_count} from previous)")
        data = prox_data
        real_count = prox_real
    elif len(prox_data) > len(data):
        print(f"[+] Proximity parser found {len(prox_data)} posts (vs {len(data)} from previous)")
        data = prox_data
        real_count = prox_real
    
    needs_llm, reason = should_use_llm(data, real_count)
    if needs_llm:
        print(f"[!] All parsers weak ({reason}) -- trying LLM...")
        llm_data, llm_real = try_llm_parser(html, url)
        
        if len(llm_data) >= 2 and llm_real > real_count:
            return post_process(llm_data)
        
        if data:
            return post_process(data)
        return post_process(llm_data)
    
    return post_process(data)


def process_json_dict(d, url, results):
    post_content = d.get("post", d.get("content", d.get("message", d.get("body", ""))))
    author = d.get("username", d.get("author", d.get("user", "unknown")))
    timestamp = d.get("timestamp", d.get("date", d.get("created_at", "N/A")))
    
    if post_content and isinstance(post_content, str) and len(post_content.split()) >= 3:
        results.append({
            "username": str(author),
            "post": post_content,
            "timestamp": str(timestamp),
            "source": url
        })


def _parse_raw_json(data, url):
    """Recursively search for post-like dictionaries in a raw JSON response."""
    results = []
    
    def search_dict(d):
        if isinstance(d, dict):
            process_json_dict(d, url, results)
            for v in d.values():
                search_dict(v)
        elif isinstance(d, list):
            for item in d:
                search_dict(item)
                
    search_dict(data)
    
    if len(results) > 0:
        print(f"[+] Successfully extracted {len(results)} posts from raw JSON API response.")
    
    return results
