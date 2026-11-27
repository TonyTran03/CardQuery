import pytest
from fastapi.testclient import TestClient

from api.main import create_app, create_local_app
from api.stubs import stub_parse
from frontend.fixtures import DEMO_CARDS, GOLDEN_FILTERS, GOLDEN_QUERY, GOLDEN_RESPONSE
from parser import load_parser
from parser.contracts import InvalidQueryError, empty_filters


@pytest.fixture
def client():
    with TestClient(create_app(DEMO_CARDS, stub_parse)) as test_client:
        yield test_client


def test_stub_serves_all_routes(client):
    assert client.get("/health").json() == {"status": "ok"}
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert metrics.headers["content-type"].startswith("text/plain")
    assert "cardquery_api_info 1" in metrics.text

    search = client.post("/search", json={"query": f"  {GOLDEN_QUERY}  "})
    assert search.status_code == 200
    assert search.json() == GOLDEN_RESPONSE
    assert client.post("/parse", json={"query": GOLDEN_QUERY}).json() == {
        "filters": GOLDEN_FILTERS, "model_version": "stub-v1",
    }
    assert client.post("/cards/search", json={
        "filters": empty_filters(), "limit": 1, "offset": 1,
    }).json() == {"cards": [DEMO_CARDS[1]], "total": 3, "limit": 1, "offset": 1}
    assert client.post("/cards/search", json={
        "filters": dict(empty_filters(), types=["Unknown"]),
    }).json() == {"cards": [], "total": 0, "limit": 20, "offset": 0}


@pytest.mark.parametrize(("path", "body"), [
    ("/parse", {"query": "  "}),
    ("/parse", {"query": "x" * 501}),
    ("/parse", {"query": GOLDEN_QUERY, "limit": 1}),
    ("/search", {"query": GOLDEN_QUERY, "limit": True}),
    ("/search", {"query": GOLDEN_QUERY, "offset": "0"}),
    ("/cards/search", {"filters": {"types": []}}),
    ("/cards/search", {"filters": dict(GOLDEN_FILTERS, extra=None)}),
    ("/cards/search", {"filters": dict(GOLDEN_FILTERS, hp_min=150.0)}),
    ("/cards/search", {"filters": dict(GOLDEN_FILTERS, hp_max=149)}),
])
def test_invalid_body_returns_422(client, path, body):
    response = client.post(path, json=body)
    assert response.status_code == 422
    assert "detail" in response.json()


@pytest.mark.parametrize("path", ["/parse", "/search"])
@pytest.mark.parametrize(("error", "status", "detail"), [
    (InvalidQueryError("Invalid HP range"), 422, "Invalid HP range"),
    (RuntimeError("prediction failed"), 503, "model_unavailable"),
])
def test_parser_failures_map_to_http_errors(path, error, status, detail):
    def broken_parser(query):
        raise error

    with TestClient(create_app(DEMO_CARDS, broken_parser)) as client:
        response = client.post(path, json={"query": GOLDEN_QUERY})
    assert response.status_code == status
    assert response.json() == {"detail": detail}


def test_instrument_installer_owns_metrics():
    calls = []

    def instrument(app):
        calls.append(app)
        app.add_api_route("/metrics", lambda: {"custom": True})

    app = create_app(DEMO_CARDS, stub_parse, instrument=instrument)
    with TestClient(app) as client:
        assert client.get("/metrics").json() == {"custom": True}
    assert calls == [app]


def test_saved_model_and_catalog(monkeypatch):
    monkeypatch.setenv("CARDQUERY_API_MODE", "local")
    monkeypatch.delenv("CARDQUERY_CARDS_PATH", raising=False)
    monkeypatch.delenv("CARDQUERY_MODEL_PATH", raising=False)
    monkeypatch.setenv("CARDQUERY_MODEL_VERSION", "test-model")
    with TestClient(create_local_app()) as client:
        ready = client.get("/ready")
        response = client.post("/search", json={"query": GOLDEN_QUERY, "limit": 1})
        invalid = client.post("/parse", json={"query": "between 200 and 100 hp"})
    assert ready.status_code == 200
    assert ready.json() == {"status": "ok"}
    assert response.status_code == 200
    body = response.json()
    assert body["filters"] == GOLDEN_FILTERS
    assert body["model_version"] == "test-model"
    assert body["total"] >= 1 and len(body["cards"]) == 1
    assert "Charizard" in body["cards"][0]["name"]
    assert body["cards"][0]["hp"] >= 150
    assert invalid.status_code == 422


def test_missing_model_does_not_disable_catalog_only_search(monkeypatch, tmp_path):
    monkeypatch.setenv("CARDQUERY_API_MODE", "local")
    monkeypatch.delenv("CARDQUERY_CARDS_PATH", raising=False)
    monkeypatch.setenv("CARDQUERY_MODEL_PATH", str(tmp_path / "missing.joblib"))
    with TestClient(create_local_app()) as client:
        assert client.get("/health").json() == {"status": "ok"}
        ready = client.get("/ready")
        assert client.post("/cards/search", json={"filters": GOLDEN_FILTERS}).status_code == 200
        response = client.post("/search", json={"query": GOLDEN_QUERY})
    assert ready.status_code == 503
    assert ready.json() == {"detail": "model_unavailable"}
    assert response.status_code == 503
    assert response.json() == {"detail": "model_unavailable"}


def test_model_that_loads_but_cannot_predict_is_not_ready(monkeypatch):
    def load_unusable_model(path):
        return dict(load_parser(path), vectorizer=None)

    monkeypatch.setenv("CARDQUERY_API_MODE", "local")
    monkeypatch.delenv("CARDQUERY_CARDS_PATH", raising=False)
    monkeypatch.delenv("CARDQUERY_MODEL_PATH", raising=False)
    monkeypatch.setattr("api.main.load_parser", load_unusable_model)
    with TestClient(create_local_app()) as client:
        assert client.get("/ready").status_code == 503
