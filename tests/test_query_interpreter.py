"""Reported search failures, exercised with the real saved parser and catalog."""

import pytest
from fastapi.testclient import TestClient

from api.main import create_local_app


@pytest.fixture(scope="module")
def client():
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("CARDQUERY_API_MODE", "local")
        patch.delenv("CARDQUERY_MODEL_PATH", raising=False)
        patch.delenv("CARDQUERY_CARDS_PATH", raising=False)
        with TestClient(create_local_app()) as test_client:
            yield test_client


@pytest.mark.parametrize("query", [
    "charizard less than 150 hp", "CHARIZARD LESS THAN 150 HP",
    "charizard less   than 150 hp",
])
def test_less_than_matches_strict_under_search(client, query):
    response = client.post("/search", json={"query": query})
    expected = client.post("/search", json={"query": "charizard under 150 hp"})
    assert response.status_code == 200
    assert response.json() == expected.json()
    result = response.json()
    assert result["filters"]["name"] == "Charizard"
    assert result["filters"]["hp_max"] == 149
    assert result["total"] > 0
    assert all(card["hp"] < 150 for card in result["cards"])


def test_less_than_excludes_equal_hp_while_at_most_includes_it(client):
    strict = client.post("/search", json={"query": "charizard less than 150 hp"}).json()
    inclusive = client.post("/search", json={"query": "charizard at most 150 hp"}).json()
    assert strict["total"] < inclusive["total"]
    assert any(card["hp"] == 150 for card in inclusive["cards"])


@pytest.mark.parametrize("query", [
    "charizard less than 0 hp", "charizard less than -1 hp", "charizard less than 2.5 hp",
    "charizard at least 150 hp and less than 150 hp",
])
def test_invalid_less_than_constraints_are_rejected(client, query):
    assert client.post("/parse", json={"query": query}).status_code == 422


@pytest.mark.parametrize("query", ["winston", "steven", "hello", "flanfy", "winston pokemon"])
@pytest.mark.parametrize("path", ["/parse", "/search"])
def test_unknown_names_do_not_turn_into_flaaffy(client, query, path):
    response = client.post(path, json={"query": query})
    assert response.status_code == 422
    assert "No supported Pokémon name or filter" in response.json()["detail"]


@pytest.mark.parametrize("query", ["water pokemon", "fire cards", "pokemon with under 100 hp"])
def test_filter_only_queries_do_not_acquire_a_pokemon_name(client, query):
    response = client.post("/search", json={"query": query})
    assert response.status_code == 200
    result = response.json()
    assert result["filters"]["name"] is None
    assert result["total"] > 0


@pytest.mark.parametrize("query", ["cards", "show me cards", "pokemon"])
def test_browsing_still_works(client, query):
    response = client.post("/search", json={"query": query})
    assert response.status_code == 200
    assert response.json()["filters"]["name"] is None
    assert response.json()["total"] > 0


@pytest.mark.parametrize("query", ["pikachu", "blastoise", "snorlax", "flaaffy", "fire charizards with at least 150 hp"])
def test_real_names_still_work(client, query):
    response = client.post("/search", json={"query": query})
    assert response.status_code == 200
    assert response.json()["total"] > 0


def test_supplied_normal_and_drift_queries_remain_supported():
    from pathlib import Path

    from api.query_interpreter import QueryInterpreter, normalize_search_query
    from catalog.cards import load_cards
    from parser import load_parser
    from parser.data import load_rows
    from parser.parser import parse_queries

    root = Path(__file__).resolve().parents[1]
    interpreter = QueryInterpreter(load_cards(root / "data/cards.csv"))
    model = load_parser(root / "artifacts/parser/model.joblib")
    for mode in ("normal", "drift"):
        cases = load_rows(root / f"data/unseen_{mode}.csv")
        queries = [normalize_search_query(case["query"]) for case in cases]
        for query, prediction in zip(queries, parse_queries(model, queries)):
            assert interpreter.validate_prediction(query, prediction) == prediction
