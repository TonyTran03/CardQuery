"""Labeled model performance from logged predictions (Part 6).

Public API:
    compute_performance(rows) -> dict
    compute_performance_with_report(rows) -> (dict, report_dict)

A *labeled success* is an HTTP 200 row whose ``expected_filters`` and
``predicted_filters`` are both valid Filters. Labels come from the unseen case
files, never from predictions.
"""
from __future__ import annotations

import json
import os
from typing import Any

FILTER_KEYS = (
    "name", "types", "hp_min", "hp_max", "rarity", "year_min", "year_max",
)


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def canonicalize_filters(f: Any) -> dict | None:
    """Return a canonical copy of a Filters dict, or None if it is invalid.

    Canonical = trimmed name/rarity, types deduplicated case-insensitively and
    sorted case-insensitively. Spelling of each type is preserved.
    """
    if not isinstance(f, dict) or set(f.keys()) != set(FILTER_KEYS):
        return None

    out: dict[str, Any] = {}
    for key in ("name", "rarity"):
        v = f[key]
        if v is None:
            out[key] = None
        elif isinstance(v, str) and v.strip():
            out[key] = v.strip()
        else:
            return None

    types = f["types"]
    if not isinstance(types, list):
        return None
    seen: dict[str, str] = {}
    for t in types:
        if not isinstance(t, str) or not t.strip():
            return None
        seen.setdefault(t.strip().lower(), t.strip())
    out["types"] = [seen[k] for k in sorted(seen)]

    for key in ("hp_min", "hp_max", "year_min", "year_max"):
        v = f[key]
        if v is not None and not _is_int(v):
            return None
        out[key] = v
    for key in ("hp_min", "hp_max"):
        if out[key] is not None and out[key] < 0:
            return None
    for key in ("year_min", "year_max"):
        if out[key] is not None and not 1 <= out[key] <= 9999:
            return None
    if (out["hp_min"] is not None and out["hp_max"] is not None
            and out["hp_min"] > out["hp_max"]):
        return None
    if (out["year_min"] is not None and out["year_max"] is not None
            and out["year_min"] > out["year_max"]):
        return None
    return {k: out[k] for k in FILTER_KEYS}


def filters_label(f: dict) -> str:
    """Sorted-key JSON string used as the multiclass label for Accuracy."""
    return json.dumps(f, sort_keys=True, separators=(",", ":"))


def slot_pairs(f: dict) -> set[tuple[str, Any]]:
    """Each non-null scalar -> (field, value); each type -> ("types", value)."""
    pairs: set[tuple[str, Any]] = set()
    for key in FILTER_KEYS:
        if key == "types":
            pairs.update(("types", t) for t in f["types"])
        elif f[key] is not None:
            pairs.add((key, f[key]))
    return pairs


def micro_f1(expected: list[dict], predicted: list[dict]) -> float:
    """TP/FP/FN summed over queries; 2TP / (2TP + FP + FN); 1.0 if denom is 0."""
    tp = fp = fn = 0
    for exp, pred in zip(expected, predicted):
        e, p = slot_pairs(exp), slot_pairs(pred)
        tp += len(e & p)
        fp += len(p - e)
        fn += len(e - p)
    denom = 2 * tp + fp + fn
    return 1.0 if denom == 0 else 2 * tp / denom


def labeled_successes(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Canonical (expected, predicted) lists for labeled-success rows."""
    expected, predicted = [], []
    for r in rows:
        if r.get("status_code") != 200:
            continue
        e = canonicalize_filters(r.get("expected_filters"))
        p = canonicalize_filters(r.get("predicted_filters"))
        if e is None or p is None:
            continue
        expected.append(e)
        predicted.append(p)
    return expected, predicted


def _evidently_accuracy(targets: list[str], preds: list[str]):
    """Run Evidently's classification quality metric. Returns (accuracy, report)."""
    import pandas as pd
    from evidently import ColumnMapping  # type: ignore
    from evidently.metrics import ClassificationQualityMetric  # type: ignore
    from evidently.report import Report  # type: ignore

    df = pd.DataFrame({"target": targets, "prediction": preds})
    report = Report(metrics=[ClassificationQualityMetric()])
    report.run(
        reference_data=None,
        current_data=df,
        column_mapping=ColumnMapping(target="target", prediction="prediction"),
    )
    snapshot = json.loads(report.json())
    accuracy = snapshot["metrics"][0]["result"]["current"]["accuracy"]
    return float(accuracy), snapshot


def _engine() -> str:
    return os.environ.get("CARDQUERY_METRICS_ENGINE", "evidently").lower()


def compute_performance_with_report(rows: list[dict]) -> tuple[dict, dict]:
    n_requests = len(rows)
    expected, predicted = labeled_successes(rows)
    n_ok = len(expected)
    coverage = n_ok / n_requests if n_requests else 0.0

    summary: dict[str, Any] = {
        "n_requests": n_requests,
        "n_labeled_successes": n_ok,
        "label_coverage": coverage,
        "exact_match": None,
        "micro_f1": None,
    }
    report: dict[str, Any] = {"engine": "none", "note": "no labeled successes"}
    if n_ok == 0:
        return summary, report

    targets = [filters_label(e) for e in expected]
    preds = [filters_label(p) for p in predicted]
    direct_exact = sum(t == p for t, p in zip(targets, preds)) / n_ok

    engine = _engine()
    if engine not in ("evidently", "fallback"):
        raise ValueError("metrics engine must be evidently or fallback")
    accuracy, report = None, None
    if engine == "evidently":
        try:
            accuracy, report = _evidently_accuracy(targets, preds)
            report = {"engine": "evidently", "snapshot": report}
        except ImportError:
            if engine == "evidently":
                raise
    if accuracy is None:
        accuracy = direct_exact
        report = {
            "engine": "fallback",
            "note": "Evidently not installed; accuracy computed directly.",
            "accuracy": direct_exact,
        }

    summary["exact_match"] = accuracy
    summary["micro_f1"] = micro_f1(expected, predicted)
    return summary, report


def compute_performance(rows: list[dict]) -> dict:
    """PerformanceSummary: n_requests, n_labeled_successes, label_coverage,
    exact_match, micro_f1 (both scores None when no labeled successes)."""
    return compute_performance_with_report(rows)[0]
