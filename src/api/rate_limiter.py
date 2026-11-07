"""
Rate Limiter for JobClarity API.

Provides thread-safe, sliding-window in-memory rate limiting per client IP:
  - Configurable rate limit (default: 60 requests per minute via RATE_LIMIT_PER_MINUTE).
  - Reverse-proxy awareness for Render, Cloudflare, and Nginx (TRUST_PROXY_HEADERS).
  - Validation of forwarded IP strings using ipaddress to prevent spoofing or header injection.
  - Generates standard HTTP 429 Too Many Requests with Retry-After and X-RateLimit headers.
  - Endpoints like /health and / remain unmetered.
"""

from __future__ import annotations

from collections import deque
import ipaddress
import logging
import os
import threading
import time
from typing import Optional
from fastapi import HTTPException, Request, Response

logger = logging.getLogger(__name__)

DEFAULT_RATE_LIMIT_PER_MINUTE = 60
DEFAULT_WINDOW_SECONDS = 60


def is_valid_ip(ip_str: str) -> bool:
    """Validate whether a string is a syntactically valid IPv4 or IPv6 address."""
    if not isinstance(ip_str, str):
        return False
    stripped = ip_str.strip()
    if not stripped:
        return False
    try:
        ipaddress.ip_address(stripped)
        return True
    except ValueError:
        return False


def get_client_ip(
    request: Request,
    trust_proxy_headers: Optional[bool] = None,
) -> str:
    """
    Extract client IP, accounting for Render / reverse-proxy forwarded headers.

    Safety:
      - Only trusts X-Forwarded-For when trust_proxy_headers is True.
      - Strictly validates candidate IP addresses with ipaddress.ip_address.
      - Discards invalid, malformed, or injected header contents.
      - Falls back to request.client.host if proxy headers are disabled or invalid.
    """
    if trust_proxy_headers is None:
        env_val = os.getenv("TRUST_PROXY_HEADERS", "true").lower().strip()
        trust_proxy_headers = env_val in ("true", "1", "yes")

    if trust_proxy_headers:
        # Render and standard reverse proxies place the real client IP as the first comma-separated entry
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            parts = [p.strip() for p in forwarded_for.split(",") if p.strip()]
            if parts:
                candidate = parts[0]
                if is_valid_ip(candidate):
                    return candidate

        # Secondary proxy header fallback (e.g. Nginx proxy_set_header X-Real-IP)
        real_ip = request.headers.get("x-real-ip")
        if real_ip and is_valid_ip(real_ip):
            return real_ip.strip()

    # Direct client connection fallback
    if request.client and request.client.host:
        return request.client.host

    return "127.0.0.1"


class SlidingWindowRateLimiter:
    """Thread-safe sliding-window rate limiter tracking request timestamps per client IP."""

    def __init__(
        self,
        requests_per_minute: Optional[int] = None,
        window_seconds: int = DEFAULT_WINDOW_SECONDS,
    ):
        self._requests_per_minute = requests_per_minute
        self._window_seconds = window_seconds
        self._clients: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    @property
    def limit(self) -> int:
        if self._requests_per_minute is not None:
            return self._requests_per_minute
        try:
            return int(os.getenv("RATE_LIMIT_PER_MINUTE", str(DEFAULT_RATE_LIMIT_PER_MINUTE)))
        except ValueError:
            return DEFAULT_RATE_LIMIT_PER_MINUTE

    def check_rate_limit(
        self,
        client_id: str,
        current_time: Optional[float] = None,
    ) -> tuple[bool, int, int]:
        """
        Check whether client_id has exceeded rate limits.

        Returns:
            (is_allowed: bool, remaining_requests: int, retry_after_seconds: int)
        """
        now = current_time if current_time is not None else time.time()
        cutoff = now - self._window_seconds

        with self._lock:
            if client_id not in self._clients:
                self._clients[client_id] = deque()

            timestamps = self._clients[client_id]

            # Evict timestamps outside the sliding window
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()

            current_count = len(timestamps)
            limit = self.limit

            if current_count >= limit:
                # Rate limit exceeded
                oldest_ts = timestamps[0]
                retry_after = max(1, int(oldest_ts + self._window_seconds - now))
                return False, 0, retry_after

            # Allow request and append timestamp
            timestamps.append(now)
            remaining = max(0, limit - len(timestamps))
            return True, remaining, 0

    def reset(self) -> None:
        """Clear all in-memory client tracking state (used in testing)."""
        with self._lock:
            self._clients.clear()


# Default singleton rate limiter
rate_limiter = SlidingWindowRateLimiter()


def rate_limit_dependency(request: Request, response: Response) -> None:
    """
    FastAPI dependency that enforces rate limits on incoming requests.

    Raises:
        HTTPException 429 when the client exceeds the allowed request rate.
    """
    # Exclude non-prediction paths if applied globally
    path = request.url.path
    if path in ("/", "/health", "/docs", "/openapi.json", "/redoc"):
        return

    client_ip = get_client_ip(request)
    allowed, remaining, retry_after = rate_limiter.check_rate_limit(client_ip)

    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {rate_limiter.limit} requests per minute allowed.",
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(rate_limiter.limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(retry_after),
            },
        )

    # Attach informational rate limit headers to allowed response
    response.headers["X-RateLimit-Limit"] = str(rate_limiter.limit)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-RateLimit-Reset"] = str(DEFAULT_WINDOW_SECONDS)
