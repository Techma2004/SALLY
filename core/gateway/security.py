from __future__ import annotations

import hmac
import time
from collections import OrderedDict, defaultdict, deque
from threading import Lock

from fastapi import HTTPException, Request

from core.config import settings

_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def _client_host(request: Request) -> str:
    client = request.client
    return client.host if client else ""


async def require_local_or_api_key(request: Request) -> None:
    """
    Guard for endpoints that expose user data or system internals.

    Default-secure behaviour:
      * If SALLY_API_KEY is configured, the request MUST present it via
        `Authorization: Bearer <key>` or `X-API-Key: <key>` — regardless
        of where it comes from.
      * If no API key is configured, the request is only allowed when it
        originates from the loopback interface (i.e. SALLY is being used
        as a local single-user assistant, which is the default deploy
        mode). Anything else is rejected rather than silently trusted.
    """
    api_key = settings.server.api_key

    if api_key:
        header_key = request.headers.get("X-API-Key", "")
        auth_header = request.headers.get("Authorization", "")
        bearer_key = (
            auth_header[len("Bearer "):]
            if auth_header.startswith("Bearer ")
            else ""
        )

        supplied = header_key or bearer_key

        if not supplied or not hmac.compare_digest(supplied, api_key):
            raise HTTPException(status_code=401, detail="Missing or invalid API key.")

        return

    if _client_host(request) not in _LOOPBACK_HOSTS:
        raise HTTPException(
            status_code=403,
            detail=(
                "This SALLY instance is not configured for remote access. "
                "Set SALLY_API_KEY to allow authenticated non-local requests."
            ),
        )


class RateLimiter:
    """
    Small in-memory sliding-window rate limiter keyed by client IP.

    Not a substitute for a real edge/proxy rate limiter in a multi-worker
    deployment, but enough to stop a single unauthenticated caller from
    hammering the local LLM/SQLite layer.
    """

    def __init__(self, limit_per_minute: int) -> None:
        self.limit = max(1, limit_per_minute)
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()
        # Bound how many distinct client keys we track to avoid unbounded
        # growth if callers spoof many different source values.
        self._max_tracked_clients = 5000
        self._order: OrderedDict[str, None] = OrderedDict()

    def check(self, key: str) -> None:
        now = time.monotonic()
        window_start = now - 60.0

        with self._lock:
            if key not in self._hits and len(self._order) >= self._max_tracked_clients:
                oldest_key, _ = self._order.popitem(last=False)
                self._hits.pop(oldest_key, None)

            self._order[key] = None
            self._order.move_to_end(key)

            hits = self._hits[key]
            while hits and hits[0] < window_start:
                hits.popleft()

            if len(hits) >= self.limit:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Please slow down.",
                )

            hits.append(now)


rate_limiter = RateLimiter(settings.server.rate_limit_per_minute)


async def enforce_rate_limit(request: Request) -> None:
    key = _client_host(request) or "unknown"
    rate_limiter.check(key)
