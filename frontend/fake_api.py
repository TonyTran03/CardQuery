"""A tiny stand-in for the Part 3 API, for demos and tests without the backend.

    python -m frontend.fake_api            # serves http://localhost:8000

It implements ``POST /search`` and ``GET /health`` with the agreed response
shapes, searches the three demo cards with a keyword parser, and has trigger
queries for exercising the frontend's error states:

    trigger 503   -> HTTP 503 {"detail": "model_unavailable"}
    trigger 500   -> HTTP 500
    trigger slow  -> waits FAKE_API_SLOW_SECONDS (default 12) before answering

Standard library only, so it never needs the backend's dependencies.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from frontend.fixtures import DEMO_CARDS, GOLDEN_FILTERS, GOLDEN_QUERY

MODEL_VERSION = "stub-v1"
KNOWN_TYPES = ["Colorless", "Darkness", "Dragon", "Fairy", "Fighting", "Fire", "Grass",
               "Lightning", "Metal", "Psychic", "Water"]
TYPE_WORDS = {t.lower(): t for t in KNOWN_TYPES} | {"electric": "Lightning"}


def fake_parse(query: str) -> dict:
    """Keyword parser good enough for demos; returns seven-key Filters."""
    q = " ".join(query.lower().split())
    if q == GOLDEN_QUERY:
        return dict(GOLDEN_FILTERS)
    filters = {"name": None, "types": [], "hp_min": None, "hp_max": None,
               "rarity": None, "year_min": None, "year_max": None}
    for name in sorted({c["name"] for c in DEMO_CARDS}):
        if name.lower() in q:
            filters["name"] = name
    words = re.findall(r"[a-z]+", q)
    filters["types"] = sorted({TYPE_WORDS[w] for w in words if w in TYPE_WORDS})
    for word, rarity in (("rare", "Rare"), ("common", "Common")):
        if word in words:
            filters["rarity"] = rarity

    def number(pattern):
        m = re.search(pattern, q)
        return int(m.group(1)) if m else None

    if (n := number(r"(?:at least|minimum|min) (\d+) ?hp")) is not None:
        filters["hp_min"] = n
    if (n := number(r"(?:over|more than|above) (\d+) ?hp")) is not None:
        filters["hp_min"] = n + 1
    if (n := number(r"(?:at most|maximum|max) (\d+) ?hp")) is not None:
        filters["hp_max"] = n
    if (n := number(r"(?:under|less than|below) (\d+) ?hp")) is not None:
        filters["hp_max"] = max(n - 1, 0)
    if (n := number(r"after (\d{4})")) is not None:
        filters["year_min"] = n + 1
    if (n := number(r"before (\d{4})")) is not None:
        filters["year_max"] = n - 1
    if (n := number(r"(?:from|in) (\d{4})")) is not None:
        filters["year_min"] = filters["year_max"] = n
    return filters


def fake_search(filters: dict, limit: int, offset: int) -> tuple[list[dict], int]:
    """Search semantics from the team guide (page 15), over the demo cards."""

    def within(value, low, high):
        if low is None and high is None:
            return True
        if value is None:
            return False
        return (low is None or value >= low) and (high is None or value <= high)

    wanted_types = {t.lower() for t in filters["types"]}
    matches = []
    for card in sorted(DEMO_CARDS, key=lambda c: c["card_id"]):
        if filters["name"] and filters["name"].lower() not in card["name"].lower():
            continue
        if wanted_types and not wanted_types & {t.lower() for t in card["types"]}:
            continue
        if filters["rarity"] and (card["rarity"] or "").lower() != filters["rarity"].lower():
            continue
        if not within(card["hp"], filters["hp_min"], filters["hp_max"]):
            continue
        if not within(card["release_year"], filters["year_min"], filters["year_max"]):
            continue
        matches.append(card)
    return matches[offset:offset + limit], len(matches)


def _validation_errors(body) -> list[dict]:
    def err(loc, msg, kind):
        return {"loc": ["body", *loc], "msg": msg, "type": kind}

    if not isinstance(body, dict):
        return [err([], "Input should be a valid dictionary", "dict_type")]
    errors = [err([k], "Extra inputs are not permitted", "extra_forbidden")
              for k in body if k not in ("query", "limit", "offset")]
    query = body.get("query")
    if not isinstance(query, str):
        errors.append(err(["query"], "Field required" if query is None
                          else "Input should be a valid string", "string_type"))
    elif not 1 <= len(query.strip()) <= 500:
        errors.append(err(["query"], "Query must be 1 to 500 characters", "value_error"))
    for key, low, high in (("limit", 1, 100), ("offset", 0, None)):
        value = body.get(key, 20 if key == "limit" else 0)
        if not isinstance(value, int) or isinstance(value, bool):
            errors.append(err([key], "Input should be a valid integer", "int_type"))
        elif value < low or (high is not None and value > high):
            errors.append(err([key], f"{key} is out of range", "value_error"))
    return errors


class FakeApiHandler(BaseHTTPRequestHandler):
    slow_seconds = float(os.environ.get("FAKE_API_SLOW_SECONDS", "12"))

    def log_message(self, fmt, *args):  # keep test output quiet
        if os.environ.get("FAKE_API_VERBOSE"):
            super().log_message(fmt, *args)

    def _send(self, status: int, payload) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok"})
        else:
            self._send(404, {"detail": "Not Found"})

    def do_POST(self):
        if self.path != "/search":
            self._send(404, {"detail": "Not Found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"null")
        except ValueError:
            self._send(422, {"detail": [{"loc": ["body"], "msg": "Invalid JSON",
                                         "type": "json_invalid"}]})
            return
        errors = _validation_errors(body)
        if errors:
            self._send(422, {"detail": errors})
            return

        query = body["query"].strip()
        trigger = " ".join(query.lower().split())
        if trigger == "trigger 503":
            self._send(503, {"detail": "model_unavailable"})
            return
        if trigger == "trigger 500":
            self._send(500, {"detail": "Internal Server Error"})
            return
        if trigger == "trigger slow":
            time.sleep(self.slow_seconds)

        filters = fake_parse(query)
        cards, total = fake_search(filters, body.get("limit", 20), body.get("offset", 0))
        self._send(200, {"filters": filters, "cards": cards, "total": total,
                         "model_version": MODEL_VERSION})


def start_server(host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    """Start in a background thread (port 0 = any free port). Call ``shutdown()``."""
    server = ThreadingHTTPServer((host, port), FakeApiHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print(f"Fake CardQuery API on http://localhost:{port}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("0.0.0.0", port), FakeApiHandler).serve_forever()
