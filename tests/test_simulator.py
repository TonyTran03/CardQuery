"""Offline tests for Part 4's traffic simulator.

Uses mocked ``requests.post`` and the stdlib fake API (``frontend.fake_api``);
no backend, parser or PostgreSQL is needed.
"""

import copy
import csv
import json
import re
import socket
import uuid
from datetime import datetime
from pathlib import Path

import pytest
import requests

from frontend.fake_api import start_server
from frontend.fixtures import GOLDEN_FILTERS, GOLDEN_QUERY, GOLDEN_RESPONSE
from monitoring import simulator
from monitoring.simulator import (FILTER_KEYS, InMemoryCollector, load_cases, main,
                                  run_traffic, select_cases)

FIXTURES = Path(__file__).parent / "fixtures"
NORMAL_POOL = FIXTURES / "unseen_normal.csv"
DRIFT_POOL = FIXTURES / "unseen_drift.csv"
TRAINING_CSV = Path(__file__).resolve().parents[1] / "data" / "query_examples.csv"
LOG_KEYS = {"request_id", "timestamp", "query", "mode", "status_code", "latency_ms",
            "model_version", "predicted_filters", "expected_filters"}


class FakeResponse:
    def __init__(self, status_code, payload=None, raw=False):
        self.status_code = status_code
        self._payload = payload
        self._raw = raw

    def json(self):
        if self._raw:
            raise ValueError("not JSON")
        return copy.deepcopy(self._payload)


@pytest.fixture
def mock_post(monkeypatch):
    calls = []
    state = {"respond": lambda: FakeResponse(200, GOLDEN_RESPONSE)}

    def fake_post(url, json=None, timeout=None, **kwargs):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return state["respond"]()

    monkeypatch.setattr(simulator.requests, "post", fake_post)
    return calls, state


@pytest.fixture
def fake_api():
    server = start_server()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def closed_port_url() -> str:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}"


# --- run_traffic --------------------------------------------------------------

def test_run_traffic_golden_log(mock_post):
    calls, _ = mock_post
    logs = run_traffic("http://localhost:8000/", [GOLDEN_QUERY], mode="normal")

    assert calls == [{"url": "http://localhost:8000/search",
                      "json": {"query": GOLDEN_QUERY, "limit": 20, "offset": 0},
                      "timeout": 10}]
    assert len(logs) == 1
    log = logs[0]
    assert set(log) == LOG_KEYS
    uuid.UUID(log["request_id"])
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z", log["timestamp"])
    datetime.fromisoformat(log["timestamp"].replace("Z", "+00:00"))
    assert log["query"] == GOLDEN_QUERY and log["mode"] == "normal"
    assert log["status_code"] == 200
    assert isinstance(log["latency_ms"], float) and log["latency_ms"] >= 0
    assert log["model_version"] == "stub-v1"
    assert log["predicted_filters"] == GOLDEN_FILTERS
    assert log["expected_filters"] is None


def test_request_ids_are_unique(mock_post):
    logs = run_traffic("http://x", [GOLDEN_QUERY] * 50, mode="drift")
    assert len({log["request_id"] for log in logs}) == 50
    assert all(log["mode"] == "drift" for log in logs)


def test_http_error_records_no_prediction(mock_post):
    _, state = mock_post
    state["respond"] = lambda: FakeResponse(503, {"detail": "model_unavailable"})
    [log] = run_traffic("http://x", ["pikachu"], mode="normal")
    assert log["status_code"] == 503
    assert log["predicted_filters"] is None
    assert log["model_version"] is None
    assert log["latency_ms"] >= 0


@pytest.mark.parametrize("response", [
    FakeResponse(200, raw=True),
    FakeResponse(200, {"filters": {"name": "Pikachu"}, "model_version": "v2"}),
    FakeResponse(200, {"filters": None, "model_version": 7}),
])
def test_unusable_success_body_is_not_turned_into_a_prediction(mock_post, response):
    _, state = mock_post
    state["respond"] = lambda: response
    [log] = run_traffic("http://x", ["pikachu"], mode="normal")
    assert log["status_code"] == 200
    assert log["predicted_filters"] is None
    assert log["model_version"] in (None, "v2")


@pytest.mark.parametrize("exc", [requests.Timeout(), requests.ConnectionError()])
def test_network_failure_has_status_zero_and_measured_latency(mock_post, monkeypatch, exc):
    _, state = mock_post
    ticks = iter([100.0, 100.25])
    monkeypatch.setattr(simulator.time, "perf_counter", lambda: next(ticks))

    def boom():
        raise exc

    state["respond"] = boom
    [log] = run_traffic("http://x", ["pikachu"], mode="normal")
    assert log["status_code"] == 0
    assert log["predicted_filters"] is None and log["model_version"] is None
    assert log["latency_ms"] == 250.0


@pytest.mark.parametrize("kwargs", [
    {"base_url": "http://x", "queries": ["q"], "mode": "spike"},
    {"base_url": "http://x", "queries": ["q"], "mode": "Normal"},
    {"base_url": "  ", "queries": ["q"], "mode": "normal"},
    {"base_url": "http://x", "queries": "not a list", "mode": "normal"},
    {"base_url": "http://x", "queries": ["ok", 5], "mode": "normal"},
])
def test_run_traffic_rejects_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        run_traffic(**kwargs)


# --- query pools ----------------------------------------------------------------

@pytest.mark.parametrize("pool", [NORMAL_POOL, DRIFT_POOL])
def test_pools_are_labelled_seven_key_filters(pool):
    cases = load_cases(pool)
    assert len(cases) >= 10
    for case in cases:
        assert tuple(case["filters"]) == FILTER_KEYS
        assert isinstance(case["filters"]["types"], list)


def test_pools_are_disjoint_and_absent_from_training_data():
    normal = {c["query"].lower() for c in load_cases(NORMAL_POOL)}
    drift = {c["query"].lower() for c in load_cases(DRIFT_POOL)}
    assert not normal & drift
    if TRAINING_CSV.exists():  # Part 2's training data, once merged
        with TRAINING_CSV.open(encoding="utf-8") as fh:
            trained = {" ".join(row["query"].lower().split()) for row in csv.DictReader(fh)}
        assert not (normal | drift) & trained


def test_drift_pool_has_a_different_text_distribution():
    """Drift must change the text, not just the mode label."""
    def tokens(cases):
        return [re.findall(r"[a-z0-9]+", c["query"].lower()) for c in cases]

    normal, drift = tokens(load_cases(NORMAL_POOL)), tokens(load_cases(DRIFT_POOL))
    normal_vocab = {t for q in normal for t in q}
    drift_tokens = [t for q in drift for t in q]
    unseen_share = sum(t not in normal_vocab for t in drift_tokens) / len(drift_tokens)
    mean_len = lambda qs: sum(map(len, qs)) / len(qs)  # noqa: E731
    assert unseen_share >= 0.3
    assert mean_len(drift) > 1.5 * mean_len(normal)


def test_load_cases_rejects_bad_files(tmp_path):
    bad_header = tmp_path / "a.csv"
    bad_header.write_text("text,filters\nhi,{}\n")
    bad_json = tmp_path / "b.csv"
    bad_json.write_text('query,filters_json\nhi,"{not json"\n')
    missing_key = tmp_path / "c.csv"
    missing_key.write_text('query,filters_json\nhi,"{""name"": null}"\n')
    empty = tmp_path / "d.csv"
    empty.write_text("query,filters_json\n")
    for path in (bad_header, bad_json, missing_key, empty):
        with pytest.raises(ValueError):
            load_cases(path)


def test_select_cases_is_seeded_and_cycles_to_exact_count():
    cases = [{"query": f"q{i}", "filters": {}} for i in range(7)]
    picked = select_cases(cases, 100)
    assert len(picked) == 100
    assert picked == select_cases(cases, 100, seed=42)
    assert picked[:7] != cases  # shuffled
    assert sorted(c["query"] for c in picked[:7]) == sorted(c["query"] for c in cases)
    assert picked[7:14] == picked[:7]  # cycles in the same order
    with pytest.raises(ValueError):
        select_cases(cases, 0)


# --- CLI ------------------------------------------------------------------------

@pytest.mark.parametrize("mode,pool", [("normal", NORMAL_POOL), ("drift", DRIFT_POOL)])
def test_cli_sends_exact_count_attaches_labels_and_stores(fake_api, mode, pool, capsys):
    saver = InMemoryCollector()
    code = main(["--mode", mode, "--count", "25", "--api-url", fake_api,
                 "--cases", str(pool), "--database-url", "postgresql://db/test"], saver=saver)

    assert code == 0
    assert saver.calls == 1 and len(saver.rows) == 25
    labels = {c["query"]: c["filters"] for c in load_cases(pool)}
    for log in saver.rows:
        assert set(log) == LOG_KEYS
        assert log["mode"] == mode and log["status_code"] == 200
        assert log["expected_filters"] == labels[log["query"]]
        assert tuple(log["predicted_filters"]) == FILTER_KEYS
    assert f"[{mode}] 25 requests" in capsys.readouterr().out


def test_cli_passes_database_url_from_environment(fake_api, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://env/db")
    seen = {}

    def saver(rows, database_url):
        seen["url"] = database_url

    assert main(["--mode", "normal", "--count", "2", "--api-url", fake_api,
                 "--cases", str(NORMAL_POOL)], saver=saver) == 0
    assert seen["url"] == "postgresql://env/db"


def test_cli_records_failures_without_inventing_predictions():
    saver = InMemoryCollector()
    code = main(["--mode", "normal", "--count", "3", "--api-url", closed_port_url(),
                 "--cases", str(NORMAL_POOL), "--database-url", "postgresql://x"], saver=saver)
    assert code == 0
    assert [log["status_code"] for log in saver.rows] == [0, 0, 0]
    assert all(log["predicted_filters"] is None for log in saver.rows)
    assert all(log["expected_filters"] is not None for log in saver.rows)


def test_cli_requires_database_url_unless_dry_run(fake_api, monkeypatch, capsys):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    args = ["--mode", "drift", "--count", "3", "--api-url", fake_api, "--cases", str(DRIFT_POOL)]
    assert main(args) == 2
    assert "DATABASE_URL" in capsys.readouterr().err
    assert main(args + ["--dry-run"]) == 0


def test_cli_fails_loudly_when_storage_fails(fake_api, capsys):
    def broken_saver(rows, database_url):
        raise ConnectionError("could not connect to server")

    code = main(["--mode", "normal", "--count", "2", "--api-url", fake_api,
                 "--cases", str(NORMAL_POOL), "--database-url", "postgresql://x"],
                saver=broken_saver)
    assert code == 1
    assert "storing traffic logs failed" in capsys.readouterr().err


def test_cli_optional_jsonl_export(fake_api, tmp_path):
    out = tmp_path / "logs.jsonl"
    assert main(["--mode", "normal", "--count", "4", "--api-url", fake_api,
                 "--cases", str(NORMAL_POOL), "--dry-run", "--jsonl", str(out)]) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 4 and all(set(r) == LOG_KEYS for r in rows)


@pytest.mark.parametrize("args", [
    ["--mode", "spike"],
    ["--mode", "normal", "--count", "abc"],
])
def test_cli_rejects_invalid_arguments(args):
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2


def test_cli_rejects_zero_count_and_missing_pool(tmp_path):
    assert main(["--mode", "normal", "--count", "0", "--dry-run",
                 "--cases", str(NORMAL_POOL)]) == 2
    assert main(["--mode", "normal", "--dry-run", "--cases", str(tmp_path / "nope.csv")]) == 2
