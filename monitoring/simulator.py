"""Traffic simulator (Part 4).

Sends unseen labelled queries to the API's ``POST /search``, records one
TrafficLog per request, attaches the known labels and hands the logs to
Part 6's ``save_traffic_logs`` for PostgreSQL storage.

    python -m monitoring.simulator --mode normal --count 100 --api-url http://localhost:8000
    python -m monitoring.simulator --mode drift  --count 100 --api-url http://localhost:8000

Query pools default to ``data/unseen_normal.csv`` / ``data/unseen_drift.csv``
(columns ``query,filters_json``); ``--cases`` points elsewhere. ``DATABASE_URL``
(or ``--database-url``) is required unless ``--dry-run`` is given.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import requests

MODES = ("normal", "drift")
FILTER_KEYS = ("name", "types", "hp_min", "hp_max", "rarity", "year_min", "year_max")
TIMEOUT_SECONDS = 10
SEED = 42
REPO_ROOT = Path(__file__).resolve().parents[1]

SaveFn = Callable[[list, str], None]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _is_filters(value) -> bool:
    return isinstance(value, dict) and set(value) == set(FILTER_KEYS)


def run_traffic(base_url: str, queries: list[str], mode: str) -> list[dict]:
    """POST each query to ``{base_url}/search`` and return one TrafficLog per request.

    ``expected_filters`` is always ``None`` here; the CLI attaches labels.
    Failed requests get ``predicted_filters=None``; network failures and
    timeouts get ``status_code=0``. Latency is measured either way.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("base_url must be a non-empty string")
    if isinstance(queries, str) or not all(isinstance(q, str) for q in queries):
        raise ValueError("queries must be a list of strings")

    url = base_url.strip().rstrip("/") + "/search"
    logs = []
    for query in queries:
        log = {
            "request_id": str(uuid.uuid4()),
            "timestamp": _utc_now(),
            "query": query,
            "mode": mode,
            "status_code": 0,
            "latency_ms": 0.0,
            "model_version": None,
            "predicted_filters": None,
            "expected_filters": None,
        }
        started = time.perf_counter()
        try:
            response = requests.post(
                url, json={"query": query, "limit": 20, "offset": 0}, timeout=TIMEOUT_SECONDS
            )
        except requests.RequestException:
            response = None
        log["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)

        if response is not None:
            log["status_code"] = response.status_code
            try:
                payload = response.json()
            except ValueError:
                payload = None
            if isinstance(payload, dict):
                version = payload.get("model_version")
                log["model_version"] = version if isinstance(version, str) else None
                if response.status_code == 200 and _is_filters(payload.get("filters")):
                    log["predicted_filters"] = payload["filters"]
        logs.append(log)
    return logs


def load_cases(path: str | os.PathLike) -> list[dict]:
    """Read a ``query,filters_json`` CSV into ``[{"query", "filters"}]``."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != ["query", "filters_json"]:
            raise ValueError(f"{path}: header must be exactly query,filters_json "
                             f"(got {reader.fieldnames})")
        cases = []
        for line_no, row in enumerate(reader, start=2):
            query = (row["query"] or "").strip()
            if not query:
                raise ValueError(f"{path}:{line_no}: empty query")
            try:
                filters = json.loads(row["filters_json"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: filters_json is not valid JSON") from exc
            if not _is_filters(filters):
                raise ValueError(f"{path}:{line_no}: filters_json must have exactly the keys "
                                 f"{', '.join(FILTER_KEYS)}")
            cases.append({"query": query, "filters": filters})
    if not cases:
        raise ValueError(f"{path}: no cases")
    return cases


def select_cases(cases: list[dict], count: int, seed: int = SEED) -> list[dict]:
    """Shuffle once with ``seed``, then cycle through for exactly ``count`` cases."""
    if count < 1:
        raise ValueError("count must be at least 1")
    order = list(cases)
    random.Random(seed).shuffle(order)
    return [order[i % len(order)] for i in range(count)]


def simulate(api_url: str, cases: list[dict], mode: str, count: int,
             seed: int = SEED) -> list[dict]:
    """Pick ``count`` cases, send them, and attach each case's known label."""
    chosen = select_cases(cases, count, seed)
    logs = run_traffic(api_url, [c["query"] for c in chosen], mode)
    for log, case in zip(logs, chosen):
        log["expected_filters"] = case["filters"]
    return logs


def _default_saver() -> SaveFn:
    try:
        from monitoring.persistence import save_traffic_logs
    except ImportError as exc:
        raise SystemExit(
            "error: Part 6's storage adapter (monitoring.persistence.save_traffic_logs) "
            "is not installed. Use --dry-run to test without PostgreSQL."
        ) from exc
    return save_traffic_logs


class InMemoryCollector:
    """Stand-in for ``save_traffic_logs`` until Part 6 lands (used by --dry-run/tests)."""

    def __init__(self):
        self.rows: list[dict] = []
        self.calls = 0

    def __call__(self, rows: list[dict], database_url: str) -> None:
        self.calls += 1
        self.rows.extend(rows)


def _summary(logs: list[dict]) -> str:
    by_status: dict[int, int] = {}
    for log in logs:
        by_status[log["status_code"]] = by_status.get(log["status_code"], 0) + 1
    statuses = ", ".join(f"{code}: {n}" for code, n in sorted(by_status.items()))
    mean = sum(log["latency_ms"] for log in logs) / len(logs)
    return f"{len(logs)} requests ({statuses}); mean latency {mean:.1f} ms"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m monitoring.simulator",
                                     description="Send unseen labelled traffic to the CardQuery API.")
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--api-url", default=os.environ.get("API_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--cases", help="query pool CSV (default: data/unseen_<mode>.csv)")
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--jsonl", help="optional: also write logs to this JSONL file")
    parser.add_argument("--dry-run", action="store_true",
                        help="keep logs in memory instead of PostgreSQL (no DATABASE_URL needed)")
    return parser


def main(argv: list[str] | None = None, saver: SaveFn | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.count < 1:
        print("error: --count must be at least 1", file=sys.stderr)
        return 2

    cases_path = Path(args.cases) if args.cases else REPO_ROOT / "data" / f"unseen_{args.mode}.csv"
    try:
        cases = load_cases(cases_path)
    except (OSError, ValueError) as exc:
        print(f"error: could not load cases: {exc}", file=sys.stderr)
        return 2

    if saver is None:
        if args.dry_run:
            saver = InMemoryCollector()
        else:
            if not args.database_url:
                print("error: set DATABASE_URL (or --database-url), or use --dry-run",
                      file=sys.stderr)
                return 2
            saver = _default_saver()

    logs = simulate(args.api_url, cases, args.mode, args.count, args.seed)

    if args.jsonl:
        with open(args.jsonl, "w", encoding="utf-8") as fh:
            for log in logs:
                fh.write(json.dumps(log) + "\n")

    try:
        saver(logs, args.database_url or "")
    except Exception as exc:  # storage must fail loudly, never silently
        print(f"error: storing traffic logs failed: {exc}", file=sys.stderr)
        return 1

    where = "memory (dry run)" if args.dry_run else "PostgreSQL"
    print(f"[{args.mode}] {_summary(logs)}; stored in {where}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
