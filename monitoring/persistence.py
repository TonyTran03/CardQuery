"""PostgreSQL storage for prediction logs and drift runs (Part 6).

Public API:
    save_traffic_logs(rows, database_url) -> None   (Part 4's simulator calls this)
    apply_schema(database_url) -> None
    fetch_latest_logs(database_url, n) -> list[dict]
    save_drift_run(record, database_url) -> bool    (False if run_id already stored)

PostgreSQL is required: if it is unreachable these functions raise, they never
fall back to file storage.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_PATH = Path(__file__).parent / "sql" / "schema.sql"

LOG_KEYS = (
    "request_id", "timestamp", "query", "mode", "status_code",
    "latency_ms", "model_version", "predicted_filters", "expected_filters",
)

INSERT_LOG = """
INSERT INTO prediction_logs
  (request_id, timestamp, query, mode, status_code, latency_ms,
   model_version, predicted_filters, expected_filters)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
ON CONFLICT (request_id) DO NOTHING
"""

SELECT_LATEST = """
SELECT request_id, timestamp, query, mode, status_code, latency_ms,
       model_version, predicted_filters, expected_filters
FROM (
    SELECT * FROM prediction_logs ORDER BY timestamp DESC, request_id DESC LIMIT %s
) latest
ORDER BY timestamp, request_id
"""

INSERT_RUN = """
INSERT INTO drift_runs
  (run_id, window_start, window_end, drift_detected, drift_share,
   drifted_columns, model_metrics, evidently_report)
VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb)
ON CONFLICT (run_id) DO NOTHING
"""


def _connect(database_url: str):
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("psycopg is not installed; see monitoring/requirements.txt") from exc
    try:
        return psycopg.connect(database_url)
    except Exception as exc:
        raise RuntimeError(f"PostgreSQL unavailable: {exc}") from exc


def _json(value: Any) -> str | None:
    return None if value is None else json.dumps(value, sort_keys=True)


def _validate_log(row: dict) -> None:
    missing = [k for k in LOG_KEYS if k not in row]
    if missing:
        raise ValueError(f"traffic log missing keys: {missing}")
    if not isinstance(row["request_id"], str) or not row["request_id"]:
        raise ValueError("request_id must be a nonempty string")
    if not isinstance(row["query"], str):
        raise ValueError("query must be a string")
    if row["mode"] not in ("normal", "drift"):
        raise ValueError("mode must be 'normal' or 'drift'")
    if isinstance(row["status_code"], bool) or not isinstance(row["status_code"], int):
        raise ValueError("status_code must be an integer")
    if not isinstance(row["latency_ms"], (int, float)) or isinstance(row["latency_ms"], bool):
        raise ValueError("latency_ms must be a number")
    if not math.isfinite(row["latency_ms"]) or row["latency_ms"] < 0:
        raise ValueError("latency_ms must be finite and nonnegative")
    if row["model_version"] is not None and not isinstance(row["model_version"], str):
        raise ValueError("model_version must be a string or None")
    for key in ("predicted_filters", "expected_filters"):
        if row[key] is not None and not isinstance(row[key], dict):
            raise ValueError(f"{key} must be a dict or None")
    try:
        dt = datetime.fromisoformat(str(row["timestamp"]).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            raise ValueError("timestamp must include timezone")
    except ValueError as exc:
        raise ValueError(f"bad timestamp: {row['timestamp']!r}") from exc


def apply_schema(database_url: str) -> None:
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with _connect(database_url) as conn:
        conn.execute(sql)


def save_traffic_logs(rows: list[dict], database_url: str) -> None:
    """Insert once per request_id; retries never duplicate rows."""
    for r in rows:
        _validate_log(r)
    with _connect(database_url) as conn:
        conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
        with conn.cursor() as cur:
            cur.executemany(
                INSERT_LOG,
                [
                    (
                        r["request_id"], r["timestamp"], r["query"], r["mode"],
                        r["status_code"], float(r["latency_ms"]), r["model_version"],
                        _json(r["predicted_filters"]), _json(r["expected_filters"]),
                    )
                    for r in rows
                ],
            )


def _z(ts: Any) -> str:
    if isinstance(ts, datetime):
        return ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return str(ts)


def fetch_latest_logs(database_url: str, n: int = 100) -> list[dict]:
    """Latest n logs, returned in timestamp, request_id order."""
    with _connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(SELECT_LATEST, (n,))
            out = []
            for rec in cur.fetchall():
                row = dict(zip(LOG_KEYS, rec))
                row["timestamp"] = _z(row["timestamp"])
                out.append(row)
            return out


def save_drift_run(record: dict, database_url: str) -> bool:
    """Insert a drift_runs row. Returns False when run_id already existed."""
    with _connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                INSERT_RUN,
                (
                    record["run_id"], record["window_start"], record["window_end"],
                    record["drift_detected"], record["drift_share"],
                    json.dumps(record["drifted_columns"]),
                    json.dumps(record["model_metrics"]),
                    json.dumps(record["evidently_report"], default=str),
                ),
            )
            return cur.rowcount == 1


def drift_run_exists(run_id: str, database_url: str) -> bool:
    with _connect(database_url) as conn:
        return conn.execute("SELECT 1 FROM drift_runs WHERE run_id = %s", (run_id,)).fetchone() is not None
