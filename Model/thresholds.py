"""
Canonical Thresholds and Three-Way Decision Standards.

Single source of truth for similarity scoring, confidence thresholds,
and identity attribution verdicts across CLI, Web Dashboard, Fingerprint Store,
and Machine Learning models.
"""

MATCH_THRESHOLD: float = 0.80
MISMATCH_THRESHOLD: float = 0.30
COSINE_TIE_BREAKER: float = 0.85  # Subforum & Timeprint tie-breaker trigger threshold
DEFAULT_CANDIDATE_THRESHOLD: float = 0.50  # Filter threshold for candidate pairing & graph edges

VERDICT_MATCH = "MATCH"
VERDICT_MISMATCH = "MISMATCH"
VERDICT_UNCERTAIN = "UNCERTAIN"


def interpret_similarity(score: float) -> tuple[str, str, str]:
    """
    Map a similarity score to standard verdict, human-readable label, and CLI icon.
    
    Returns:
        (verdict, display_label, cli_icon)
        - MATCH: ("MATCH", "CONFIDENT MATCH", "[+]")
        - MISMATCH: ("MISMATCH", "LIKELY DIFFERENT", "[-]")
        - UNCERTAIN: ("UNCERTAIN", "UNCERTAIN", "[?]")
    """
    if score >= MATCH_THRESHOLD:
        return VERDICT_MATCH, "CONFIDENT MATCH", "[+]"
    if score <= MISMATCH_THRESHOLD:
        return VERDICT_MISMATCH, "LIKELY DIFFERENT", "[-]"
    return VERDICT_UNCERTAIN, "UNCERTAIN", "[?]"


def get_verdict_badge_class(verdict: str) -> str:
    """Return CSS badge class corresponding to a standard verdict."""
    if verdict == VERDICT_MATCH:
        return "verdict-same"
    if verdict == VERDICT_MISMATCH:
        return "verdict-diff"
    return "verdict-unc"
