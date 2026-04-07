"""
utils/rate_limiter.py
---------------------
Simple in-process rate limiter using a sliding window.
For production, replace with Redis-backed solution (e.g. redis-py + lua script).

Usage:
    from execution.utils.rate_limiter import RateLimiter
    rl = RateLimiter(max_calls=60, period_seconds=60)
    if not rl.allow(user_id="u_123"):
        raise RateLimitExceeded("Too many requests")
"""

import time
from collections import defaultdict, deque
from threading import Lock


class RateLimitExceeded(Exception):
    """Raised when a user exceeds their allowed request rate."""
    pass


class RateLimiter:
    """
    Sliding-window rate limiter (thread-safe, in-process).

    Args:
        max_calls: Maximum number of calls allowed per `period_seconds`.
        period_seconds: Rolling window duration in seconds.
    """

    def __init__(self, max_calls: int = 60, period_seconds: int = 60) -> None:
        self.max_calls = max_calls
        self.period = period_seconds
        self._calls: dict[str, deque] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, user_id: str) -> bool:
        """
        Check whether the user is within their rate limit.

        Returns:
            True if the request is allowed, False if rate-limited.
        """
        now = time.monotonic()
        cutoff = now - self.period

        with self._lock:
            window = self._calls[user_id]
            # Evict calls outside the rolling window
            while window and window[0] < cutoff:
                window.popleft()

            if len(window) >= self.max_calls:
                return False

            window.append(now)
            return True

    def check_or_raise(self, user_id: str) -> None:
        """
        Allow the request or raise RateLimitExceeded.

        Raises:
            RateLimitExceeded: with retry_after hint.
        """
        if not self.allow(user_id):
            raise RateLimitExceeded(
                f"Rate limit exceeded for user '{user_id}'. "
                f"Limit: {self.max_calls} requests / {self.period}s."
            )
