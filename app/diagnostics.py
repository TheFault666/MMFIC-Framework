"""Shared diagnostics, structured logging, finite-value guards, and crawl event hooks."""

from contextlib import contextmanager
import logging
import os
import sys
import threading

_LOG_LOCK = threading.Lock()
_VERBOSE = False
_DEBUG = False
_LOGGER_NAME = "trowler"

_TRUE_VALUES = {"1", "true", "yes", "on"}


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _TRUE_VALUES


def configure_logging(verbose: bool | None = None, debug: bool | None = None) -> logging.Logger:
    """Configure the shared trowler logger and global verbose/debug toggles."""
    global _VERBOSE, _DEBUG
    with _LOG_LOCK:
        _VERBOSE = _env_flag("TROWLER_VERBOSE") if verbose is None else bool(verbose)
        _DEBUG = _env_flag("TROWLER_DEBUG") if debug is None else bool(debug)

        logger = logging.getLogger(_LOGGER_NAME)
        if _DEBUG:
            logger.setLevel(logging.DEBUG)
        elif _VERBOSE:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)

        if not logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
            logger.addHandler(handler)
        logger.propagate = False
        return logger


def is_verbose() -> bool:
    return _VERBOSE


def is_debug() -> bool:
    return _DEBUG


def get_logger(name: str | None = None) -> logging.Logger:
    root_logger = logging.getLogger(_LOGGER_NAME)
    if not root_logger.handlers:
        configure_logging()
    if not name or name == _LOGGER_NAME:
        return root_logger
    suffix = name.removeprefix(f"{_LOGGER_NAME}.")
    return logging.getLogger(f"{_LOGGER_NAME}.{suffix}")


def add_verbosity_args(parser) -> None:
    """Attach standard --verbose and --debug flags to an ArgumentParser."""
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose diagnostic logging")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging with full tracebacks")


def log_failure(
    context: str,
    exc: BaseException | None = None,
    *,
    level: int = logging.WARNING,
    logger_name: str | None = None,
    **details,
) -> None:
    """Log an error or failure event with structured context and optional traceback."""
    logger = get_logger(logger_name)
    parts = [context]
    if details:
        detail_str = ", ".join(f"{k}={v}" for k, v in details.items())
        parts.append(f"({detail_str})")
    if exc is not None:
        parts.append(f"-> {type(exc).__name__}: {exc}")
    msg = " ".join(parts)

    include_tb = exc is not None and (_DEBUG or _VERBOSE or level >= logging.ERROR)
    logger.log(level, msg, exc_info=exc if include_tb else None)


class FailureTracker:
    """Thread-safe failure and total counter registry for OSINT traceability."""

    def __init__(self):
        self._lock = threading.Lock()
        self._stats: dict[str, dict[str, int]] = {}

    def record(self, stage: str, *, total: int = 0, failed: int = 0) -> None:
        with self._lock:
            entry = self._stats.setdefault(stage, {"total": 0, "failed": 0})
            entry["total"] += int(total)
            entry["failed"] += int(failed)

    def get(self, stage: str) -> dict[str, int]:
        with self._lock:
            entry = self._stats.get(stage, {"total": 0, "failed": 0})
            return dict(entry)

    def snapshot(self) -> dict[str, dict[str, int]]:
        with self._lock:
            return {k: dict(v) for k, v in self._stats.items()}


def sanitize_finite_array(arr, *, context: str = ""):
    """Replace NaN/Inf with 0.0 while logging counts and sample feature indices."""
    import numpy as np

    arr_np = np.asarray(arr)
    bad_mask = ~np.isfinite(arr_np)
    if not np.any(bad_mask):
        return arr_np

    nan_count = int(np.isnan(arr_np).sum())
    inf_count = int(np.isinf(arr_np).sum())
    sample_indices = np.flatnonzero(bad_mask)[:10].tolist()
    ctx = context or "vector"
    log_failure(
        f"Masked non-finite values in {ctx}",
        level=logging.WARNING,
        logger_name="fingerprint",
        nan_count=nan_count,
        inf_count=inf_count,
        sample_indices=sample_indices,
    )
    return np.nan_to_num(arr_np, nan=0.0, posinf=0.0, neginf=0.0)


_EVENT_LOCK = threading.Lock()
_ACTIVE_CALLBACKS = {
    "on_captcha_pending": None,
    "on_posts_found": None,
    "on_log": None,
}


@contextmanager
def crawl_callbacks(*, on_captcha_pending=None, on_posts_found=None, on_log=None):
    """Temporarily register typed crawl lifecycle callbacks on the active run."""
    with _EVENT_LOCK:
        previous = dict(_ACTIVE_CALLBACKS)
        _ACTIVE_CALLBACKS["on_captcha_pending"] = on_captcha_pending
        _ACTIVE_CALLBACKS["on_posts_found"] = on_posts_found
        _ACTIVE_CALLBACKS["on_log"] = on_log
    try:
        yield
    finally:
        with _EVENT_LOCK:
            _ACTIVE_CALLBACKS.update(previous)


def _invoke_callback(key: str, *args) -> bool:
    with _EVENT_LOCK:
        cb = _ACTIVE_CALLBACKS.get(key)
    if cb is None:
        return False
    try:
        cb(*args)
        return True
    except Exception as exc:
        log_failure(f"Crawl callback '{key}' raised exception", exc, level=logging.ERROR)
        return False


def notify_captcha_pending() -> bool:
    """Signal listeners (e.g. dashboard) that HITL CAPTCHA resolution is pending."""
    return _invoke_callback("on_captcha_pending")


def notify_posts_found(count: int) -> bool:
    """Report newly collected post counts cleanly without stdout regex parsing."""
    if not count or count <= 0:
        return False
    return _invoke_callback("on_posts_found", int(count))


def notify_log(message: str) -> bool:
    """Emit a structured log line to an active crawl listener."""
    return _invoke_callback("on_log", str(message))
