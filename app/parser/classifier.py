from bs4 import BeautifulSoup
import re


def is_forum_url(url):
    url_lower = url.lower()
    url_forum_hints = [
        "/forum", "/thread", "/topic", "/viewtopic", "/showthread",
        "/post/", "/discussion", "/t/", "?tid=", "?t=",
    ]
    return any(h in url_lower for h in url_forum_hints)


def is_blog_url(url):
    url_lower = url.lower()
    url_blog_hints = ["/blog", "/article", "/news"]
    return any(h in url_lower for h in url_blog_hints)


def calculate_forum_score(soup, is_forum_url_flag):
    forum_score = 0
    
    post_class_patterns = [
        re.compile(r"post|message|comment|reply|thread", re.I),
    ]
    
    for pattern in post_class_patterns:
        for tag in ["div", "li", "article", "tr"]:
            matches = soup.find_all(tag, class_=pattern)
            if len(matches) >= 3:
                forum_score += 3
                break
        if forum_score > 0:
            break
            
    for attr in ["data-post-id", "data-message-id", "data-comment-id"]:
        if len(soup.find_all(attrs={attr: True})) >= 2:
            forum_score += 3
            break
            
    username_patterns = re.compile(r"username|user-name|author|poster|member", re.I)
    username_elems = soup.find_all(True, class_=username_patterns)
    if len(username_elems) >= 2:
        forum_score += 2
        
    time_elems = soup.find_all("time")
    if len(time_elems) >= 2:
        forum_score += 1
        
    if is_forum_url_flag:
        forum_score += 2
        
    return forum_score


def has_articles(soup):
    articles = soup.find_all("article")
    return len(articles) >= 1


def has_many_links(soup):
    return len(soup.find_all("a", href=True)) > 50


def classify_page(html, url):
    """Classify page type using structural analysis, not keyword matching.
    
    Returns: 'forum', 'blog', 'listing', or 'generic'
    """
    soup = BeautifulSoup(html, "html.parser")
    
    url_is_forum = is_forum_url(url)
    url_is_blog  = is_blog_url(url)
    
    forum_score = calculate_forum_score(soup, url_is_forum)
    
    if forum_score >= 3:
        return "forum"
    
    if url_is_blog or has_articles(soup):
        return "blog"
    
    if has_many_links(soup):
        return "listing"
    
    return "generic"