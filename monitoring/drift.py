"""Data drift on query text features + the watch-loop runner (Part 6).

Public API:
    compute_drift(reference_rows, current_rows) -> DriftSummary dict
    compute_drift_with_report(...) -> (summary, report_dict)

Run:
    python -m monitoring.drift --reference data/reference_traffic.jsonl --watch-seconds 60
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd

log = logging.getLogger("monitoring.drift")

FEATURES = ("query_length", "token_count")
MIN_ROWS = 100
P_THRESHOLD = 0.05      # KS p-value below this => column drifted
SHARE_THRESHOLD = 0.5   # drift_share at/above this => drift_detected
WINDOW_SIZE = 100


def _parse_ts(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _fmt_ts(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def derive_features(rows: list[dict]) -> pd.DataFrame:
    """Only text-derived features. Timestamps, status, mode and labels are excluded."""
    return pd.DataFrame(
        {
            "query_length": [len(r["query"]) for r in rows],
            "token_count": [len(r["query"].split()) for r in rows],
        }
    )


def _ks_scipy(ref: pd.DataFrame, cur: pd.DataFrame) -> dict[str, float]:
    from scipy.stats import ks_2samp

    return {c: float(ks_2samp(ref[c], cur[c]).pvalue) for c in FEATURES}


def _ks_evidently(ref: pd.DataFrame, cur: pd.DataFrame):
    """Evidently two-sample KS per feature. Returns (pvalues, report snapshot)."""
    from evidently.metrics import ColumnDriftMetric  # type: ignore
    from evidently.report import Report  # type: ignore

    report = Report(
        metrics=[
            ColumnDriftMetric(column_name=c, stattest="ks", stattest_threshold=P_THRESHOLD)
            for c in FEATURES
        ]
        )
    from evidently import ColumnMapping

    column_mapping = ColumnMapping(
        numerical_features=list(FEATURES),
        categorical_features=[]
    )

    report.run(
        reference_data=ref,
        current_data=cur,
        column_mapping=column_mapping
    )
    snapshot = json.loads(report.json())
    pvalues = {}
    for metric in snapshot["metrics"]:
        res = metric["result"]
        pvalues[res["column_name"]] = float(res["drift_score"])
    return pvalues, snapshot


def compute_drift_with_report(
    reference_rows: list[dict], current_rows: list[dict]
) -> tuple[dict, dict]:
    if len(reference_rows) < MIN_ROWS or len(current_rows) < MIN_ROWS:
        raise ValueError(
            f"need at least {MIN_ROWS} reference and {MIN_ROWS} current rows; "
            f"got {len(reference_rows)} and {len(current_rows)}"
        )
    ref, cur = derive_features(reference_rows), derive_features(current_rows)

    engine = os.environ.get("CARDQUERY_DRIFT_ENGINE", "evidently").lower()
    if engine not in ("evidently", "scipy"):
        raise ValueError("drift engine must be evidently or scipy")
    pvalues, snapshot = None, None
    if engine == "evidently":
        try:
            pvalues, snapshot = _ks_evidently(ref, cur)
            report: dict[str, Any] = {"engine": "evidently", "snapshot": snapshot}
        except ImportError:
            if engine == "evidently":
                raise
    if pvalues is None:
        pvalues = _ks_scipy(ref, cur)
        report = {
            "engine": "scipy-ks-fallback",
            "note": "Evidently not installed; scipy.stats.ks_2samp used.",
        }
    report["p_values"] = pvalues

    drifted = sorted(c for c in FEATURES if pvalues[c] < P_THRESHOLD)
    share = len(drifted) / len(FEATURES)
    stamps = [_parse_ts(r["timestamp"]) for r in current_rows]
    summary = {
        "window_start": _fmt_ts(min(stamps)),
        "window_end": _fmt_ts(max(stamps)),
        "drift_detected": share >= SHARE_THRESHOLD,
        "drifted_columns": drifted,
        "drift_share": share,
    }
    return summary, report


def compute_drift(reference_rows: list[dict], current_rows: list[dict]) -> dict:
    """DriftSummary: window_start, window_end, drift_detected, drifted_columns, drift_share."""
    return compute_drift_with_report(reference_rows, current_rows)[0]


# ---------------------------------------------------------------- runner

def _content_hash(rows: list[dict]) -> str:
    blob = json.dumps(rows, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def make_run_id(reference_rows: list[dict], current_rows: list[dict]) -> str:
    """Deterministic id from reference + current row contents (dedupes reruns)."""
    return hashlib.sha256(
        (_content_hash(reference_rows) + _content_hash(current_rows)).encode()
    ).hexdigest()[:32]


def load_reference(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def evaluate_window(reference_rows: list[dict], current_rows: list[dict]) -> dict:
    """Build the drift_runs record for one window (no database access)."""
    from monitoring.performance import compute_performance_with_report

    drift, drift_report = compute_drift_with_report(reference_rows, current_rows)
    perf, perf_report = compute_performance_with_report(current_rows)
    return {
        "run_id": make_run_id(reference_rows, current_rows),
        "window_start": drift["window_start"],
        "window_end": drift["window_end"],
        "drift_detected": drift["drift_detected"],
        "drift_share": drift["drift_share"],
        "drifted_columns": drift["drifted_columns"],
        "model_metrics": perf,
        "evidently_report": {"drift": drift_report, "performance": perf_report},
    }


def run_once(reference_rows: list[dict], database_url: str) -> str:
    """One watch tick. Returns 'waiting', 'unchanged' or 'saved'."""
    from monitoring import persistence

    current = persistence.fetch_latest_logs(database_url, WINDOW_SIZE)
    if len(current) < MIN_ROWS:
        log.info("waiting for rows: %d/%d", len(current), MIN_ROWS)
        return "waiting"
    run_id = make_run_id(reference_rows, current)
    if persistence.drift_run_exists(run_id, database_url):
        return "unchanged"
    record = evaluate_window(reference_rows, current)
    if persistence.save_drift_run(record, database_url):
        log.info(
            "saved run %s drift_share=%.2f exact_match=%s",
            record["run_id"], record["drift_share"], record["model_metrics"]["exact_match"],
        )
        return "saved"
    log.info("window unchanged (run %s already stored)", record["run_id"])
    return "unchanged"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="monitoring.drift")
    ap.add_argument("--reference", required=True, help="JSONL of 100 normal TrafficLogs")
    ap.add_argument("--watch-seconds", type=int, default=60)
    ap.add_argument("--once", action="store_true", help="evaluate one window and exit")
    args = ap.parse_args(argv)

    if args.watch_seconds <= 0:
        ap.error("--watch-seconds must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL is not set")

    from monitoring import persistence

    persistence.apply_schema(database_url)
    reference = load_reference(args.reference)
    if len(reference) < MIN_ROWS:
        raise SystemExit(f"reference needs >= {MIN_ROWS} rows, found {len(reference)}")

    while True:
        try:
            run_once(reference, database_url)
        except Exception:  # keep the worker alive; Compose logs show the traceback
            log.exception("drift tick failed")
            if args.once:
                raise
        if args.once:
            return
        time.sleep(args.watch_seconds)


if __name__ == "__main__":
    main()
