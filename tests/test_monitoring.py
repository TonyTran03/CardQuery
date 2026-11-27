"""Offline tests for Part 6. No PostgreSQL or Evidently required.

Opt-in live test: set TEST_DATABASE_URL to a throwaway PostgreSQL database.
"""
import copy
import json
import os
from pathlib import Path

import pytest

from monitoring import persistence
from monitoring.drift import (
    compute_drift,
    evaluate_window,
    make_run_id,
    load_reference,
)
from monitoring.performance import (
    canonicalize_filters,
    compute_performance,
    micro_f1,
)

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures" / "monitoring"

GOLDEN_FILTERS = {
    "name": "Charizard", "types": ["Fire"], "hp_min": 150, "hp_max": None,
    "rarity": None, "year_min": None, "year_max": None,
}


@pytest.fixture(autouse=True)
def force_fallback_engines(monkeypatch):
    # Keeps tests identical whether or not Evidently is installed locally.
    monkeypatch.setenv("CARDQUERY_DRIFT_ENGINE", "scipy")
    monkeypatch.setenv("CARDQUERY_METRICS_ENGINE", "fallback")


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


@pytest.fixture
def reference():
    return read_jsonl(ROOT / "data" / "reference_traffic.jsonl")


@pytest.fixture
def normal_current():
    return read_jsonl(FIX / "normal_current.jsonl")


@pytest.fixture
def shifted_current():
    return read_jsonl(FIX / "shifted_current.jsonl")


def row(**over):
    base = {
        "request_id": "r1", "timestamp": "2026-11-23T20:00:00Z",
        "query": "fire charizards with at least 150 hp", "mode": "normal",
        "status_code": 200, "latency_ms": 35.2, "model_version": "stub-v1",
        "predicted_filters": copy.deepcopy(GOLDEN_FILTERS),
        "expected_filters": copy.deepcopy(GOLDEN_FILTERS),
    }
    base.update(over)
    return base


# ------------------------------------------------------------ drift logic

def test_identical_windows_do_not_drift(reference):
    summary = compute_drift(reference, reference)
    assert summary["drift_detected"] is False
    assert summary["drifted_columns"] == []
    assert summary["drift_share"] == 0.0


def test_same_distribution_new_sample_does_not_drift(reference, normal_current):
    assert compute_drift(reference, normal_current)["drift_detected"] is False


def test_shifted_text_is_flagged(reference, shifted_current):
    summary = compute_drift(reference, shifted_current)
    assert summary["drift_detected"] is True
    assert summary["drifted_columns"] == ["query_length", "token_count"]
    assert summary["drift_share"] == 1.0
    assert summary["window_start"] == "2026-11-23T20:00:00Z"
    assert summary["window_end"] == "2026-11-23T20:49:30Z"


def test_drift_ignores_mode_labels_and_status(reference, normal_current):
    # Relabeling mode must not create drift: only text features are used.
    relabeled = [dict(r, mode="drift", status_code=500) for r in normal_current]
    assert compute_drift(reference, relabeled)["drift_detected"] is False


def test_drift_requires_100_rows(reference):
    with pytest.raises(ValueError, match="at least 100"):
        compute_drift(reference[:99], reference)
    with pytest.raises(ValueError, match="at least 100"):
        compute_drift(reference, reference[:50])


def test_drift_counts_failed_requests(reference, normal_current):
    failed = [dict(r, status_code=0, predicted_filters=None) for r in normal_current]
    assert compute_drift(reference, failed)["drift_detected"] is False


# ------------------------------------------------------------ performance

def test_perfect_predictions_score_one():
    perf = compute_performance([row(request_id=f"r{i}") for i in range(10)])
    assert perf == {
        "n_requests": 10, "n_labeled_successes": 10, "label_coverage": 1.0,
        "exact_match": 1.0, "micro_f1": 1.0,
    }


def test_incorrect_predictions_reduce_exact_match():
    good = [row(request_id=f"g{i}") for i in range(5)]
    wrong = dict(GOLDEN_FILTERS, hp_min=999)
    bad = [row(request_id=f"b{i}", predicted_filters=wrong) for i in range(5)]
    perf = compute_performance(good + bad)
    assert perf["exact_match"] == 0.5
    assert perf["micro_f1"] < 1.0


def test_unlabeled_and_failed_rows_lower_coverage():
    rows = [
        row(request_id="a"),
        row(request_id="b", expected_filters=None),
        row(request_id="c", status_code=503, predicted_filters=None),
        row(request_id="d", status_code=0, predicted_filters=None),
    ]
    perf = compute_performance(rows)
    assert perf["n_requests"] == 4
    assert perf["n_labeled_successes"] == 1
    assert perf["label_coverage"] == 0.25


def test_no_labeled_successes_gives_null_scores():
    perf = compute_performance([row(expected_filters=None)])
    assert perf["exact_match"] is None and perf["micro_f1"] is None
    assert perf["label_coverage"] == 0.0


def test_type_order_and_case_do_not_change_exact_match():
    exp = dict(GOLDEN_FILTERS, types=["Fire", "Water"])
    pred = dict(GOLDEN_FILTERS, types=["water", "FIRE", "fire"])
    # Dedup + sort are case-insensitive; spelling differences still differ.
    assert canonicalize_filters(pred)["types"] == ["FIRE", "water"]
    pred_same = dict(GOLDEN_FILTERS, types=["Water", "Fire", "Fire"])
    perf = compute_performance([row(expected_filters=exp, predicted_filters=pred_same)])
    assert perf["exact_match"] == 1.0


def test_invalid_filters_are_not_counted():
    bad = dict(GOLDEN_FILTERS, hp_min=True)  # bool is not an integer
    assert compute_performance([row(predicted_filters=bad)])["n_labeled_successes"] == 0
    missing = {k: v for k, v in GOLDEN_FILTERS.items() if k != "rarity"}
    assert compute_performance([row(predicted_filters=missing)])["n_labeled_successes"] == 0
    inverted = dict(GOLDEN_FILTERS, hp_min=200, hp_max=100)
    assert compute_performance([row(predicted_filters=inverted)])["n_labeled_successes"] == 0


def test_micro_f1_slot_arithmetic():
    exp = canonicalize_filters(GOLDEN_FILTERS)
    pred = canonicalize_filters(dict(GOLDEN_FILTERS, hp_min=None))
    # expected slots: name, types:Fire, hp_min -> 3; predicted: 2; TP=2 FN=1 FP=0
    assert micro_f1([exp], [pred]) == pytest.approx(4 / 5)


def test_micro_f1_empty_filters_is_one():
    empty = canonicalize_filters(dict(GOLDEN_FILTERS, name=None, types=[], hp_min=None))
    assert micro_f1([empty], [empty]) == 1.0


# ------------------------------------------------------------ windows / runner

def test_run_id_is_deterministic_and_content_sensitive(reference, normal_current):
    a = make_run_id(reference, normal_current)
    assert a == make_run_id(reference, normal_current)
    changed = copy.deepcopy(normal_current)
    changed[0]["query"] += " x"
    assert a != make_run_id(reference, changed)


def test_evaluate_window_record_shape(reference, shifted_current):
    rec = evaluate_window(reference, shifted_current)
    assert set(rec) == {
        "run_id", "window_start", "window_end", "drift_detected", "drift_share",
        "drifted_columns", "model_metrics", "evidently_report",
    }
    assert rec["drift_detected"] is True
    assert 0.0 <= rec["model_metrics"]["exact_match"] < 0.9
    json.dumps(rec)  # must be JSON-serialisable for JSONB columns


def test_load_reference_has_100_rows():
    assert len(load_reference(str(ROOT / "data" / "reference_traffic.jsonl"))) == 100


# ------------------------------------------------------------ persistence

def test_insert_sql_is_idempotent_by_request_id():
    assert "ON CONFLICT (request_id) DO NOTHING" in persistence.INSERT_LOG
    assert "ON CONFLICT (run_id) DO NOTHING" in persistence.INSERT_RUN


def test_schema_has_both_required_tables():
    sql = persistence.SCHEMA_PATH.read_text()
    assert "CREATE TABLE IF NOT EXISTS prediction_logs" in sql
    assert "CREATE TABLE IF NOT EXISTS drift_runs" in sql


@pytest.mark.parametrize(
    "over",
    [
        {"request_id": ""},
        {"mode": "weird"},
        {"status_code": "200"},
        {"status_code": True},
        {"latency_ms": "fast"},
        {"timestamp": "yesterday"},
        {"predicted_filters": "nope"},
    ],
)
def test_save_rejects_invalid_rows_before_connecting(monkeypatch, over):
    def boom(url):
        raise AssertionError("must not connect when validation fails")

    monkeypatch.setattr(persistence, "_connect", boom)
    with pytest.raises(ValueError):
        persistence.save_traffic_logs([row(**over)], "postgresql://unused")


def test_save_rejects_missing_keys(monkeypatch):
    r = row()
    del r["expected_filters"]
    with pytest.raises(ValueError, match="missing keys"):
        persistence.save_traffic_logs([r], "postgresql://unused")


def test_unavailable_postgres_raises_clearly(monkeypatch):
    pytest.importorskip("psycopg")
    with pytest.raises(RuntimeError, match="PostgreSQL unavailable"):
        persistence.save_traffic_logs([row()], "postgresql://nobody:x@127.0.0.1:1/none")


@pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="needs TEST_DATABASE_URL")
def test_live_roundtrip_and_no_duplicates(monkeypatch, reference, shifted_current):
    monkeypatch.setenv("CARDQUERY_DRIFT_ENGINE", "evidently")
    monkeypatch.setenv("CARDQUERY_METRICS_ENGINE", "evidently")
    url = os.environ["TEST_DATABASE_URL"]
    persistence.save_traffic_logs(shifted_current, url)
    persistence.save_traffic_logs(shifted_current, url)  # retry must not duplicate
    latest = persistence.fetch_latest_logs(url, 100)
    assert len(latest) == 100
    rec = evaluate_window(reference, latest)
    assert persistence.save_drift_run(rec, url) is True
    assert persistence.save_drift_run(rec, url) is False
    with persistence._connect(url) as conn:
        stored = conn.execute("SELECT model_metrics, evidently_report FROM drift_runs WHERE run_id = %s", (rec["run_id"],)).fetchone()
        assert stored == (rec["model_metrics"], rec["evidently_report"])
        count = conn.execute("SELECT count(*) FROM prediction_logs WHERE request_id = ANY(%s)", ([r["request_id"] for r in shifted_current],)).fetchone()[0]
        assert count == 100


def test_real_evidently_reports(monkeypatch, reference, normal_current, shifted_current):
    pytest.importorskip("evidently")
    monkeypatch.setenv("CARDQUERY_DRIFT_ENGINE", "evidently")
    monkeypatch.setenv("CARDQUERY_METRICS_ENGINE", "evidently")
    normal = evaluate_window(reference, normal_current)
    shifted = evaluate_window(reference, shifted_current)
    assert normal["drift_detected"] is False
    assert shifted["drift_share"] == 1.0
    for rec in (normal, shifted):
        for kind in ("drift", "performance"):
            report = rec["evidently_report"][kind]
            assert report["engine"] == "evidently"
            assert report["snapshot"]["metrics"]
    assert shifted["model_metrics"]["exact_match"] < normal["model_metrics"]["exact_match"]
