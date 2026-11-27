"""Real HTTP integration and pinned-model contract checks."""

import json
import threading
from contextlib import contextmanager

import pandas as pd
import pytest
import uvicorn
from fastapi.testclient import TestClient

from api.main import create_app, create_local_app
from api.model_loader import load_registered_parser
from frontend.app import create_app as create_frontend
from monitoring.metrics import instrument_api
from monitoring.simulator import simulate
from shared.fixtures import DEMO_CARDS, GOLDEN_FILTERS, GOLDEN_QUERY


@contextmanager
def live_api(app):
    import socket

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        import time

        deadline = time.monotonic() + 15
        while not server.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("API did not start")
            time.sleep(0.02)
        yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()


def test_real_model_frontend_http_and_labeled_simulator(monkeypatch):
    monkeypatch.setenv("CARDQUERY_API_MODE", "local")
    monkeypatch.delenv("CARDQUERY_MODEL_PATH", raising=False)
    monkeypatch.delenv("CARDQUERY_CARDS_PATH", raising=False)
    with live_api(create_local_app()) as url:
        monkeypatch.setenv("API_BASE_URL", url)
        page = create_frontend().test_client().get("/", query_string={"q": GOLDEN_QUERY})
        assert page.status_code == 200
        assert b"Charizard" in page.data and b"HP &gt;= 150" in page.data
        logs = simulate(url, [{"query": GOLDEN_QUERY, "filters": GOLDEN_FILTERS}], "normal", 2)
        assert len({row["request_id"] for row in logs}) == 2
        for row in logs:
            assert row["status_code"] == 200
            assert row["predicted_filters"] == row["expected_filters"] == GOLDEN_FILTERS


def test_metrics_counts_validation_errors_and_bounds_path_cardinality():
    app = create_app(DEMO_CARDS, lambda query: GOLDEN_FILTERS, instrument=instrument_api)
    with TestClient(app) as client:
        client.post("/search", json={"query": GOLDEN_QUERY})
        client.post("/search", json={"query": ""})
        client.get("/random-user-input")
        client.get("/metrics")
        metrics = client.get("/metrics").text
    assert 'api_requests_total{method="POST",path="/search",status="200"} 1.0' in metrics
    assert 'api_requests_total{method="POST",path="/search",status="422"} 1.0' in metrics
    assert 'path="unmatched",status="404"} 1.0' in metrics
    assert 'path="/metrics"' not in metrics
    assert 'api_request_duration_seconds_count{method="POST",path="/search"} 2.0' in metrics


@pytest.mark.parametrize("manifest", [
    {"model_uri": "models:/pokemon-query-parser/latest", "model_version": "1"},
    {"model_uri": "models:/pokemon-query-parser/2", "model_version": "1"},
    {"model_uri": "models:/pokemon-query-parser/1", "model_version": 1},
])
def test_registry_manifest_requires_exact_matching_version(tmp_path, manifest):
    pytest.importorskip("mlflow")
    path = tmp_path / "active.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_registered_parser(path)


def test_model_adapter_rejects_malformed_predictions(monkeypatch, tmp_path):
    mlflow = pytest.importorskip("mlflow")

    class BrokenModel:
        def predict(self, frame):
            return pd.DataFrame({"filters_json": ['{"types": []}']})

    monkeypatch.setattr(mlflow.pyfunc, "load_model", lambda uri: BrokenModel())
    path = tmp_path / "active.json"
    path.write_text(json.dumps({"model_uri": "models:/pokemon-query-parser/1", "model_version": "1"}))
    with pytest.raises(ValueError, match="exactly"):
        load_registered_parser(path)


def test_metrics_record_unexpected_server_errors():
    app = create_app(DEMO_CARDS, lambda query: GOLDEN_FILTERS, instrument=instrument_api)

    @app.get("/broken")
    def broken():
        raise RuntimeError("test server failure")

    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.get("/broken").status_code == 500
        metrics = client.get("/metrics").text
    assert 'api_requests_total{method="GET",path="/broken",status="500"} 1.0' in metrics


def test_frozen_reference_matches_real_normal_pool_and_detects_drift(monkeypatch):
    from monitoring.drift import compute_drift, load_reference
    from monitoring.simulator import REPO_ROOT, load_cases, select_cases

    monkeypatch.setenv("CARDQUERY_DRIFT_ENGINE", "scipy")
    reference = load_reference(REPO_ROOT / "data/reference_simulator.jsonl")
    for mode in ("normal", "drift"):
        rows = [{"query": case["query"], "timestamp": "2026-11-27T12:00:00Z"}
                for case in select_cases(load_cases(REPO_ROOT / f"data/unseen_{mode}.csv"), 100)]
        assert compute_drift(reference, rows)["drift_detected"] == (mode == "drift")
