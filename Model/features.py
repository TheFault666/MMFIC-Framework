import numpy as np
import re
import string
from collections import Counter
from datetime import datetime
import dateparser
from urllib.parse import urlparse


# ── Function words (style markers, not content-dependent) ────────
# These are the most reliable stylometric features — people use
# function words unconsciously and consistently across contexts.
FUNCTION_WORDS = [
    "the", "a", "an", "and", "or", "but", "if", "in", "on", "at",
    "to", "for", "of", "with", "by", "from", "as", "is", "was",
    "are", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "shall", "can", "need", "must", "not", "no", "nor",
    "so", "yet", "just", "also", "very", "too", "quite", "rather",
    "than", "then", "that", "this", "these", "those", "it", "its",
    "i", "me", "my", "we", "us", "our", "you", "your", "he", "him",
    "his", "she", "her", "they", "them", "their", "what", "which",
    "who", "whom", "how", "when", "where", "why", "all", "each",
    "every", "both", "few", "more", "most", "some", "any", "much",
    "many", "such", "here", "there", "about", "after", "before",
    "between", "into", "through", "during", "above", "below",
    "under", "over", "within", "without", "until", "since", "while",
    "although", "unless", "though", "maybe", "perhaps", "always",
    "never", "often", "sometimes", "really", "actually", "basically",
    "literally", "honestly", "clearly", "simply", "total", "least",
    "mostly", "usually", "around", "against", "among"
]

# Common internet slang / abbreviations
SLANG_WORDS = [
    "lol", "lmao", "rofl", "bruh", "tbh", "imo", "imho", "afaik",
    "smh", "fwiw", "tl;dr", "iirc", "ftw", "idk", "ngl", "fr",
    "nah", "yeah", "yep", "nope", "gonna", "wanna", "gotta",
    "btw", "omg", "wtf", "stfu", "af", "rn", "irl", "dm",
    "sus", "fam", "bro", "dude", "noob", "n00b", "op",
    "cap", "rizz", "mid", "ratio", "bet", "lowkey", "highkey", 
    "slay", "dox", "opsec", "fud", "skid", "rat", "feds", "fbi",
    "bump", "necro", "leech", "seed", "gtfo", "lmfao", "afk", 
    "brb", "ty", "thx", "pls", "plz", "woke", "based", "cope"
]


def _safe_divide_values(numerator, denominator):
    """Safe division, returns 0 if denominator is 0."""
    return numerator / denominator if denominator > 0 else 0.0


def _calculate_rate_per_100(count, n_words):
    """Normalize a count to a rate per 100 words."""
    return _safe_divide_values(count * 100, n_words)


# ── 1. STYLOMETRIC FEATURES ─────────────────────────────────────

def _compute_word_level_features(words):
    word_lengths = [len(w) for w in words]
    avg_word_len = np.mean(word_lengths)
    std_word_len = np.std(word_lengths) if len(word_lengths) > 1 else 0.0
    return avg_word_len, std_word_len


def _compute_sentence_level_features(sentences):
    sent_lengths = [len(s.split()) for s in sentences]
    avg_sent_len = np.mean(sent_lengths)
    std_sent_len = np.std(sent_lengths) if len(sent_lengths) > 1 else 0.0
    return avg_sent_len, std_sent_len


def _compute_vocabulary_richness(words, n_words):
    unique_words = set(words)
    type_token_ratio = _safe_divide_values(len(unique_words), n_words)
    word_freq = Counter(words)
    hapax = sum(1 for w, c in word_freq.items() if c == 1)
    hapax_ratio = _safe_divide_values(hapax, n_words)
    return type_token_ratio, hapax_ratio, unique_words


def _compute_punctuation_rates(text, n_words):
    punct_counts = Counter(c for c in text if c in string.punctuation)
    comma_rate    = _calculate_rate_per_100(punct_counts.get(',', 0), n_words)
    period_rate   = _calculate_rate_per_100(punct_counts.get('.', 0), n_words)
    exclam_rate   = _calculate_rate_per_100(punct_counts.get('!', 0), n_words)
    question_rate = _calculate_rate_per_100(punct_counts.get('?', 0), n_words)
    ellipsis_rate = _calculate_rate_per_100(text.count('...'), n_words)
    return comma_rate, period_rate, exclam_rate, question_rate, ellipsis_rate


def _compute_capitalization_features(text, words, n_chars, n_words):
    upper_chars = sum(1 for c in text if c.isupper())
    caps_ratio = _safe_divide_values(upper_chars, n_chars)
    cap_words = sum(1 for w in words[1:] if w[0].isupper()) if len(words) > 1 else 0
    cap_word_ratio = _safe_divide_values(cap_words, n_words)
    return caps_ratio, cap_word_ratio


def _extract_stylometric_features(posts):
    """Extract writing style features.
    
    These capture HOW someone writes, independent of WHAT they write about.
    Returns: list of floats (16 features)
    """
    text = " ".join(posts)
    words = text.lower().split()
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    chars = list(text)
    
    if not words:
        return [0.0] * 16
    
    n_words = len(words)
    n_sentences = max(len(sentences), 1)
    n_chars = max(len(chars), 1)
    
    avg_word_len, std_word_len = _compute_word_level_features(words)
    avg_sent_len, std_sent_len = _compute_sentence_level_features(sentences)
    type_token_ratio, hapax_ratio, unique_words = _compute_vocabulary_richness(words, n_words)
    comma_rate, period_rate, exclam_rate, question_rate, ellipsis_rate = _compute_punctuation_rates(text, n_words)
    caps_ratio, cap_word_ratio = _compute_capitalization_features(text, words, n_chars, n_words)
    
    avg_para_len = np.mean([len(p.split()) for p in posts]) if posts else 0.0
    
    return [
        avg_word_len,
        std_word_len,
        avg_sent_len,
        std_sent_len,
        type_token_ratio,
        hapax_ratio,
        comma_rate,
        period_rate,
        exclam_rate,
        question_rate,
        ellipsis_rate,
        caps_ratio,
        cap_word_ratio,
        avg_para_len,
        _safe_divide_values(n_words, n_sentences),
        _safe_divide_values(len(unique_words), n_sentences),
    ]


# ── 2. LEXICAL FEATURES ─────────────────────────────────────────

def _compute_function_word_rates(word_freq, n_words):
    return [_calculate_rate_per_100(word_freq.get(fw, 0), n_words) for fw in FUNCTION_WORDS]


def _compute_slang_rates(word_freq, n_words):
    return [_calculate_rate_per_100(word_freq.get(s, 0), n_words) for s in SLANG_WORDS]


def _compute_special_character_rates(text, words, n_words):
    slang_count = sum(Counter(words).get(s, 0) for s in SLANG_WORDS)
    slang_total_rate = _calculate_rate_per_100(slang_count, n_words)
    digit_words = sum(1 for w in words if any(c.isdigit() for c in w))
    digit_rate = _calculate_rate_per_100(digit_words, n_words)
    url_count = len(re.findall(r'https?://\S+|www\.\S+', text))
    url_rate = _calculate_rate_per_100(url_count, n_words)
    emoticon_count = len(re.findall(r'[:;][-(]?[)D(P/\\|]|<3|xD|XD', text))
    emoji_rate = _calculate_rate_per_100(emoticon_count, n_words)
    quote_count = text.count('"') // 2
    quote_rate = _calculate_rate_per_100(quote_count, n_words)
    contractions = len(re.findall(r"\w+'\w+", text))
    contraction_rate = _calculate_rate_per_100(contractions, n_words)
    return slang_total_rate, digit_rate, url_rate, emoji_rate, quote_rate, contraction_rate


def _compute_vocabulary_stats(words, word_freq):
    freq_ranks = [word_freq[w] for w in words]
    avg_freq_rank = np.mean(freq_ranks) if freq_ranks else 0.0
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    bigram_richness = _safe_divide_values(len(set(bigrams)), len(bigrams)) if bigrams else 0.0
    return avg_freq_rank, bigram_richness


def _extract_lexical_features(posts):
    """Extract word-choice features.
    
    These capture WHAT words someone prefers — function word frequencies
    are the gold standard in computational stylometry.
    Returns: list of floats (len(FUNCTION_WORDS) + 8 features)
    """
    text = " ".join(posts).lower()
    words = text.split()
    n_words = max(len(words), 1)
    word_freq = Counter(words)
    
    fw_features = _compute_function_word_rates(word_freq, n_words)
    slang_features = _compute_slang_rates(word_freq, n_words)
    slang_total_rate, digit_rate, url_rate, emoji_rate, quote_rate, contraction_rate = _compute_special_character_rates(text, words, n_words)
    avg_freq_rank, bigram_richness = _compute_vocabulary_stats(words, word_freq)
    
    return fw_features + slang_features + [
        slang_total_rate,
        digit_rate,
        url_rate,
        emoji_rate,
        quote_rate,
        contraction_rate,
        avg_freq_rank,
        bigram_richness,
    ]


# ── 3. BEHAVIORAL FEATURES ──────────────────────────────────────

def _compute_post_length_stats(post_lengths, n_posts):
    avg_post_len = np.mean(post_lengths)
    std_post_len = np.std(post_lengths) if len(post_lengths) > 1 else 0.0
    median_post_len = np.median(post_lengths)
    short_posts = sum(1 for l in post_lengths if l < 20)
    long_posts = sum(1 for l in post_lengths if l > 100)
    short_ratio = _safe_divide_values(short_posts, n_posts)
    long_ratio = _safe_divide_values(long_posts, n_posts)
    length_cv = _safe_divide_values(std_post_len, avg_post_len)
    return avg_post_len, std_post_len, median_post_len, short_ratio, long_ratio, length_cv


def _compute_temporal_features(timestamps):
    valid_ts = [t for t in timestamps if isinstance(t, (int, float))]
    if len(valid_ts) < 2:
        return 0.0, 0.0
    valid_ts.sort()
    diffs = np.diff(valid_ts)
    avg_time_diff = float(np.mean(diffs))
    time_range = valid_ts[-1] - valid_ts[0]
    post_velocity = len(valid_ts) / (time_range + 1.0)
    return avg_time_diff, post_velocity


def _extract_behavioral_features(posts, timestamps):
    """Extract posting behavior features.
    
    These capture posting HABITS rather than writing style.
    Returns: list of floats (8 features)
    """
    if not posts:
        return [0.0] * 10
    
    post_lengths = [len(p.split()) for p in posts]
    post_char_lengths = [len(p) for p in posts]
    n_posts = len(posts)
    
    avg_post_len, std_post_len, median_post_len, short_ratio, long_ratio, length_cv = _compute_post_length_stats(post_lengths, n_posts)
    
    total_chars = sum(post_char_lengths)
    total_words = sum(post_lengths)
    avg_chars_per_word = _safe_divide_values(total_chars, total_words)
    
    avg_time_diff, post_velocity = _compute_temporal_features(timestamps)
    
    return [
        float(n_posts),
        avg_post_len,
        std_post_len,
        median_post_len,
        short_ratio,
        long_ratio,
        length_cv,
        avg_chars_per_word,
        avg_time_diff,
        post_velocity,
    ]


def _parse_timestamp_to_hour(ts):
    """Parse a single timestamp (string or epoch) to a posting hour (0-23).
    
    Returns the hour integer, or None if parsing fails.
    """
    if isinstance(ts, str):
        dt = dateparser.parse(ts)
        return dt.hour if dt else None
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts).hour
    return None


def _extract_timeprint_features(timestamps):
    """24-dimension histogram of posting hours.
    
    Args:
        timestamps: list of ISO timestamp strings or epochs
    Returns:
        list of floats (24 features, normalized)
    """
    if not timestamps:
        return [0.0] * 24
        
    hours = np.zeros(24)
    valid_count = 0
    
    for ts in timestamps:
        if ts == "N/A" or not ts:
            continue
        hour = _parse_timestamp_to_hour(ts)
        if hour is not None:
            hours[hour] += 1
            valid_count += 1
            
    if valid_count > 0:
        hours = hours / valid_count
        
    return hours.tolist()


def _extract_subforum_features(sources, board_vocab=None):
    """Normalized frequency vector of sub-forum posting behavior.
    
    Args:
        sources: list of source URLs
        board_vocab: list of top N board path segments
    Returns:
        list of floats (len=len(board_vocab) or 10 if none)
    """
    vocab_size = len(board_vocab) if board_vocab else 10
    if not sources or not board_vocab:
        return [0.0] * vocab_size
        
    counts = np.zeros(vocab_size)
    valid_count = 0
    
    for src in sources:
        if not src or src == "manual":
            continue
        parsed = urlparse(src)
        # Use the first meaningful path segment as the board/sub-forum
        parts = [p for p in parsed.path.split('/') if p]
        board = parts[0] if parts else "root"
        
        if board in board_vocab:
            counts[board_vocab.index(board)] += 1
            valid_count += 1
            
    if valid_count > 0:
        counts = counts / valid_count
        
    return counts.tolist()


def _extract_lda_features(posts, lda_model=None, lda_vectorizer=None):
    """Latent Dirichlet Allocation topic distribution.
    
    Args:
        posts: list of strings
        lda_model: fitted scikit-learn LatentDirichletAllocation
        lda_vectorizer: fitted scikit-learn CountVectorizer
    Returns:
        list of floats (len=lda_model.n_components, or empty if no model)
    """
    if not lda_model or not lda_vectorizer:
        return []
        
    if not posts:
        return [0.0] * lda_model.n_components
        
    text = " ".join(posts)
    X = lda_vectorizer.transform([text])
    topic_dist = lda_model.transform(X)[0]
    return topic_dist.tolist()


# ── PUBLIC API ───────────────────────────────────────────────────

def _generate_empty_feature_vector(lda_model=None, board_vocab=None):
    n_stylo = len(_extract_stylometric_features([]))
    n_lexic = len(_extract_lexical_features([]))
    n_behav = 10
    n_time  = 24
    n_board = len(board_vocab) if board_vocab else 10
    n_lda   = lda_model.n_components if lda_model else 0
    return np.zeros(n_stylo + n_lexic + n_behav + n_time + n_board + n_lda)


def extract_user_features(posts, timestamps=None, sources=None, 
                          lda_model=None, lda_vectorizer=None, board_vocab=None):
    """Extract the full feature vector for one user.
    
    Args:
        posts: list of strings (all posts by this user)
        timestamps: list of timestamp strings (optional)
        sources: list of source URL strings (optional)
        lda_model: fitted LDA model (optional)
        lda_vectorizer: fitted CountVectorizer (optional)
        board_vocab: list of board names (optional)
    
    Returns:
        numpy array of shape (n_features,)
    """
    timestamps = timestamps or []
    sources = sources or []
    
    if not posts:
        return _generate_empty_feature_vector(lda_model, board_vocab)
    
    stylo = _extract_stylometric_features(posts)
    lexic = _extract_lexical_features(posts)
    behav = _extract_behavioral_features(posts, timestamps)
    timeprint = _extract_timeprint_features(timestamps)
    subforums = _extract_subforum_features(sources, board_vocab)
    
    feats = stylo + lexic + behav + timeprint + subforums
    
    if lda_model and lda_vectorizer:
        lda_feats = _extract_lda_features(posts, lda_model, lda_vectorizer)
        feats += lda_feats
    
    return np.array(feats, dtype=np.float64)


def get_feature_count():
    """Get the total number of features."""
    dummy = extract_user_features(["test post"])
    return len(dummy)


def get_feature_names(lda_topics=0, board_vocab=None):
    """Get human-readable names for all features."""
    stylo_names = [
        "avg_word_len", "std_word_len", "avg_sent_len", "std_sent_len",
        "type_token_ratio", "hapax_ratio", "comma_rate", "period_rate",
        "exclam_rate", "question_rate", "ellipsis_rate", "caps_ratio",
        "cap_word_ratio", "avg_para_len", "words_per_sentence",
        "unique_words_per_sentence",
    ]
    
    fw_names = [f"fw_{w}" for w in FUNCTION_WORDS]
    slang_names = [f"slang_{w}" for w in SLANG_WORDS]
    
    lexic_names = fw_names + slang_names + [
        "slang_total_rate", "digit_rate", "url_rate", "emoji_rate",
        "quote_rate", "contraction_rate", "avg_freq_rank", "bigram_richness",
    ]
    
    behav_names = [
        "n_posts", "avg_post_len", "std_post_len", "median_post_len",
        "short_post_ratio", "long_post_ratio", "length_cv", "chars_per_word",
        "avg_time_between_posts", "post_velocity"
    ]
    
    time_names = [f"tp_hour_{str(i).zfill(2)}" for i in range(24)]
    
    board_size = len(board_vocab) if board_vocab else 10
    board_names = [f"board_{i}" for i in range(board_size)]
    
    lda_names = [f"lda_topic_{i}" for i in range(lda_topics)]
    
    return stylo_names + lexic_names + behav_names + time_names + board_names + lda_names
