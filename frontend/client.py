"""HTTP client for the CardQuery API's POST /search endpoint.

Every failure is turned into a :class:`SearchError` carrying a short ``kind``
and a message that is safe to show to users.
"""

from __future__ import annotations

import os

import requests

DEFAULT_API_BASE_URL = "http://localhost:8000"
TIMEOUT_SECONDS = 10
MAX_QUERY_LENGTH = 500
PAGE_SIZE = 20


class SearchError(Exception):
    """A search failed; ``message`` is user-facing, ``kind`` is for code/tests."""

    def __init__(self, kind: str, message: str, status_code: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.status_code = status_code


def api_base_url() -> str:
    return os.environ.get("API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")


def _validation_message(payload) -> str:
    """Summarise FastAPI's 422 ``detail`` without dumping raw JSON on the user."""
    detail = payload.get("detail") if isinstance(payload, dict) else None
    if isinstance(detail, str) and detail.strip():
        return detail.strip()
    if isinstance(detail, list):
        messages = []
        for item in detail:
            if isinstance(item, dict) and isinstance(item.get("msg"), str):
                messages.append(item["msg"])
        if messages:
            return "; ".join(dict.fromkeys(messages))
    return "the request was not valid"


def search(query: str, *, limit: int = PAGE_SIZE, offset: int = 0,
           base_url: str | None = None) -> dict:
    """Send ``{query, limit, offset}`` to ``/search`` and return the response JSON."""
    query = query.strip()
    if not query:
        raise SearchError("validation", "Please type something to search for.")
    if len(query) > MAX_QUERY_LENGTH:
        raise SearchError(
            "validation",
            f"Your search is too long ({len(query)} characters). "
            f"Please keep it under {MAX_QUERY_LENGTH}.",
        )

    url = (base_url or api_base_url()) + "/search"
    body = {"query": query, "limit": limit, "offset": offset}
    try:
        response = requests.post(url, json=body, timeout=TIMEOUT_SECONDS)
    except requests.Timeout:
        raise SearchError(
            "timeout",
            "The search service took too long to answer. Please try again.",
        ) from None
    except requests.RequestException:
        raise SearchError(
            "network",
            "We couldn't reach the search service. Please try again in a moment.",
        ) from None

    status = response.status_code
    try:
        payload = response.json()
    except ValueError:
        payload = None

    if status == 200:
        if not _looks_like_search_response(payload):
            raise SearchError(
                "bad_response",
                "The search service sent back something we couldn't read.",
                status,
            )
        return payload
    if status == 422:
        raise SearchError(
            "validation",
            f"We couldn't understand that search: {_validation_message(payload)}.",
            status,
        )
    if status == 503:
        raise SearchError(
            "unavailable",
            "The search model is temporarily unavailable. Please try again shortly.",
            status,
        )
    raise SearchError(
        "api_error",
        f"The search service ran into a problem (HTTP {status}). Please try again.",
        status,
    )


def _looks_like_search_response(payload) -> bool:
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("filters"), dict)
        and isinstance(payload.get("cards"), list)
        and all(isinstance(card, dict) for card in payload["cards"])
        and isinstance(payload.get("total"), int)
        and not isinstance(payload.get("total"), bool)
    )
