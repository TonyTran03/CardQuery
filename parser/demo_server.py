"""Local HTML model tester, independent of the team's Part 3 API/Part 4 UI.

Run from the repository root: python -m parser.demo_server
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from catalog.cards import load_cards

from .contracts import InvalidQueryError, empty_filters
from .parser import load_parser, parse_query

ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = Path(__file__).with_name("demo") / "index.html"


def preview_cards(cards, filters, limit=8):
    """Small diagnostic preview only; production search remains Part 3's work."""
    def matches(card):
        if filters["name"] and filters["name"].casefold() not in card["name"].casefold():
            return False
        if filters["rarity"] and (card["rarity"] or "").casefold() != filters["rarity"].casefold():
            return False
        if filters["types"] and not {v.casefold() for v in filters["types"]} & {v.casefold() for v in card["types"]}:
            return False
        for field, prefix in (("hp", "hp"), ("release_year", "year")):
            low, high, value = filters[f"{prefix}_min"], filters[f"{prefix}_max"], card[field]
            if (low is not None or high is not None) and value is None:
                return False
            if low is not None and value < low or high is not None and value > high:
                return False
        return True
    matching = sorted((card for card in cards if matches(card)), key=lambda card: card["card_id"])
    return {"cards": matching[:limit], "total": len(matching)}


def demo_cases():
    # Hand-authored expectations, independent of model output.
    return [
        {"query": "fire charizards with at least 150 hp", "expected": {**empty_filters(), "name": "Charizard", "types": ["Fire"], "hp_min": 150}},
        {"query": "show me Pikachu cards", "expected": {**empty_filters(), "name": "Pikachu"}},
        {"query": "water or fire cards", "expected": {**empty_filters(), "types": ["Fire", "Water"]}},
        {"query": "cards with over 150 hp", "expected": {**empty_filters(), "hp_min": 151}},
        {"query": "cards between 1999 and 2016", "expected": {**empty_filters(), "year_min": 1999, "year_max": 2016}},
        {"query": "cards with between 60 and 120 hp", "expected": {**empty_filters(), "hp_min": 60, "hp_max": 120}},
        {"query": "cards with at least 200 hp and under 100 hp", "expected_status": 422},
        {"query": "cards before 0001", "expected_status": 422},
    ]


class DemoHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, model, cards, report, **kwargs):
        self.model, self.cards, self.report = model, cards, report
        super().__init__(*args, **kwargs)

    def reply(self, status, payload, content_type="application/json; charset=utf-8"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path in ("/", "/index.html"):
            self.reply(200, HTML_PATH.read_bytes(), "text/html; charset=utf-8")
        elif path == "/demo/info":
            self.reply(200, {"parser_kind": self.model["parser_kind"], "C": self.model["C"],
                             "n_training_rows": self.model["n_training_rows"], "n_cards": len(self.cards),
                             "vocabulary": self.model["vocabulary"],
                             "test_metrics": self.report.get("test_metrics"),
                             "baseline_test_metrics": self.report.get("baseline_test_metrics"),
                             "data_note": self.report.get("data_note"), "cases": demo_cases()})
        elif path == "/favicon.ico":
            self.reply(204, b"", "image/x-icon")
        else:
            self.reply(404, {"detail": "not_found"})

    def do_POST(self):
        if urlsplit(self.path).path != "/demo/parse":
            self.reply(404, {"detail": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 8192:
                raise InvalidQueryError("expected a JSON body of at most 8192 bytes")
            if self.headers.get_content_type() != "application/json":
                raise InvalidQueryError("Content-Type must be application/json")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict) or set(payload) != {"query"}:
                raise InvalidQueryError("request must have exactly one field: query")
        except (ValueError, UnicodeDecodeError) as exc:
            self.reply(422, {"detail": str(exc)})
            return
        started = time.perf_counter()
        try:
            filters = parse_query(self.model, payload["query"])
        except InvalidQueryError as exc:
            self.reply(422, {"detail": str(exc)})
            return
        except Exception:
            self.log_error("model prediction failed")
            self.reply(503, {"detail": "model_unavailable"})
            return
        self.reply(200, {"filters": filters, "model_version": "local-part2-v1",
                         "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                         "preview": preview_cards(self.cards, filters)})


def create_server(model, cards, report=None, host="127.0.0.1", port=8765):
    handler = partial(DemoHandler, model=model, cards=cards, report=report or {})
    return ThreadingHTTPServer((host, port), handler)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model", type=Path, default=ROOT / "artifacts/parser/model.joblib")
    cli.add_argument("--cards", type=Path, default=ROOT / "data/cards.csv")
    cli.add_argument("--port", type=int, default=8765)
    args = cli.parse_args()
    if not args.model.exists():
        cli.error("model not found; run python -m parser.train first")
    model = load_parser(str(args.model))
    report_path = args.model.with_name("evaluation.json")
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    # Reports belong to the shipped artifact; custom models need their own report.
    if report.get("model_sha256") != hashlib.sha256(args.model.read_bytes()).hexdigest():
        report = {}
    server = create_server(model, load_cards(str(args.cards)), report, port=args.port)
    print(f"CardQuery model tester: http://127.0.0.1:{server.server_port}", flush=True)
    print("Local development server. Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
