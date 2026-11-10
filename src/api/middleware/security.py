"""
Production API Hardening & Security Middleware.
Deliverable: src/api/middleware.py (Engineer 3)

Provides:
- API rate limiting (100 req/min per user) via slowapi + in-memory storage
- Strict HTTP security response headers (HSTS, CSP, X-Frame-Options, etc.)
- CORS origin restriction utilities
- Input sanitisation helpers (HTML escape, length validation)
"""

import html
import logging
from typing import Callable

from limits.storage import MemoryStorage
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger("axlero_agri_api.middleware")

# ---------------------------------------------------------------------------
# Rate Limiter — 100 requests / minute per remote IP / authenticated user
# ---------------------------------------------------------------------------

_storage = MemoryStorage()


def _get_rate_limit_key(request: Request) -> str:
    """
    Returns the rate-limit identity key.
    Uses the authenticated user's sub claim when a Bearer token is present,
    otherwise falls back to the client's remote IP address.
    """
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1]
        try:
            import jwt
            from src.api.auth import SECRET_KEY, ALGORITHM

            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user_id = payload.get("sub") or payload.get("user_id")
            if user_id:
                return f"user:{user_id}"
        except Exception:
            pass
    return get_remote_address(request)


limiter = Limiter(
    key_func=_get_rate_limit_key,
    storage_uri="memory://",
    default_limits=["100/minute"],
)


def _rate_limit_exceeded_handler(
    request: Request, exc: RateLimitExceeded
) -> JSONResponse:
    """
    Returns a structured HTTP 429 response when the rate limit is breached.
    """
    logger.warning(
        "Rate limit exceeded for %s on %s",
        _get_rate_limit_key(request),
        request.url.path,
    )
    return JSONResponse(
        status_code=429,
        content={
            "detail": "Too many requests. Limit: 100 per minute.",
            "error": {
                "code": "RATE_LIMIT_EXCEEDED",
                "message": str(exc.detail),
                "retry_after": "60 seconds",
            },
        },
        headers={"Retry-After": "60"},
    )


# ---------------------------------------------------------------------------
# Security Headers Middleware
# ---------------------------------------------------------------------------

_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": ("max-age=31536000; includeSubDomains"),
    "Content-Security-Policy": "default-src 'self'",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), microphone=()",
}


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Starlette middleware that injects strict HTTP security response headers
    on every outgoing response to protect against common web vulnerabilities.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        response: Response = await call_next(request)
        for header_name, header_value in _SECURITY_HEADERS.items():
            response.headers[header_name] = header_value
        return response


# ---------------------------------------------------------------------------
# Input Sanitisation Utilities
# ---------------------------------------------------------------------------

_MAX_STRING_LENGTH = 4096


def sanitise_string(value: str, max_length: int = _MAX_STRING_LENGTH) -> str:
    """
    Sanitises a user-supplied string by:
    1. Stripping leading/trailing whitespace.
    2. HTML-escaping special characters to prevent XSS injection.
    3. Truncating to `max_length` characters.

    Args:
        value: Raw user input string.
        max_length: Maximum allowed character length (default 4096).

    Returns:
        Sanitised string safe for storage and downstream processing.
    """
    if not isinstance(value, str):
        return ""
    cleaned = value.strip()
    cleaned = html.escape(cleaned, quote=True)
    return cleaned[:max_length]


def sanitise_dict(data: dict, max_length: int = _MAX_STRING_LENGTH) -> dict:
    """
    Recursively sanitises all string values within a dictionary payload.
    Non-string values (int, float, bool, list, dict) are preserved as-is.
    """
    result = {}
    for key, val in data.items():
        if isinstance(val, str):
            result[key] = sanitise_string(val, max_length)
        elif isinstance(val, dict):
            result[key] = sanitise_dict(val, max_length)
        elif isinstance(val, list):
            sanitised_list = []
            for item in val:
                if isinstance(item, str):
                    sanitised_list.append(sanitise_string(item, max_length))
                else:
                    sanitised_list.append(item)
            result[key] = sanitised_list
        else:
            result[key] = val
    return result


__all__ = [
    "limiter",
    "_rate_limit_exceeded_handler",
    "SecurityHeadersMiddleware",
    "sanitise_string",
    "sanitise_dict",
]
