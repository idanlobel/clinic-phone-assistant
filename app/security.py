"""API security: authentication and rate limiting.

These are implemented as FastAPI "dependencies" — reusable functions
that can be injected into any endpoint with `Depends()`. This keeps
the security logic separate from the business logic.
"""

import time
from collections import defaultdict

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from app.config import settings

# ---------------------------------------------------------------------------
# API Key Authentication
# ---------------------------------------------------------------------------

# This tells FastAPI to look for an "X-API-Key" header.
# `auto_error=False` means it won't raise a 403 automatically —
# we handle missing/invalid keys ourselves for better error messages.
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: str | None = Security(_api_key_header),
) -> str:
    """Validate the API key from the X-API-Key header.

    If no API keys are configured (api_keys is empty), auth is disabled
    and all requests are allowed — useful for local development.

    Returns the validated API key (used as client identifier for rate limiting).
    """
    allowed_keys = settings.api_keys_list

    # If no keys configured, auth is disabled (dev mode)
    if not allowed_keys:
        return "anonymous"

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Include an X-API-Key header.",
        )

    if api_key not in allowed_keys:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key.",
        )

    return api_key


# ---------------------------------------------------------------------------
# Rate Limiting (in-memory sliding window)
# ---------------------------------------------------------------------------

# Store request timestamps per API key.
# In production, use Redis instead — this doesn't work across multiple
# server instances.
_request_log: dict[str, list[float]] = defaultdict(list)


async def check_rate_limit(
    api_key: str,
) -> None:
    """Enforce rate limiting using a sliding window algorithm.

    How it works:
    1. Keep a list of timestamps for each API key
    2. Remove timestamps older than 60 seconds (outside the window)
    3. If remaining count >= limit, reject the request
    4. Otherwise, record this request's timestamp

    This is called a "sliding window" because the 60-second window
    moves forward with time, unlike a fixed window that resets at
    the top of each minute.
    """
    if not settings.rate_limit_enabled:
        return

    now = time.time()
    window_seconds = 60.0
    max_requests = settings.rate_limit_rpm

    # Remove expired timestamps (older than 60 seconds)
    timestamps = _request_log[api_key]
    _request_log[api_key] = [t for t in timestamps if now - t < window_seconds]

    # Check if over limit
    if len(_request_log[api_key]) >= max_requests:
        retry_after = int(window_seconds - (now - _request_log[api_key][0])) + 1
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {max_requests} requests per minute.",
            headers={"Retry-After": str(retry_after)},
        )

    # Record this request
    _request_log[api_key].append(now)
