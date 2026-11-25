"""Local test-harness HTTP integration, enabled with RUN_LOCAL_DEMO_TESTS=1."""

import json
import os
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from catalog.cards import load_cards
from parser import load_parser
from parser.demo_server import create_server, demo_cases

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(os.environ.get("RUN_LOCAL_DEMO_TESTS") != "1", reason="opt-in localhost integration: RUN_LOCAL_DEMO_TESTS=1")


@pytest.fixture(scope="module")
def address():
    server = create_server(load_parser(str(ROOT / "artifacts/parser/model.joblib")), load_cards(str(ROOT / "data/cards.csv")), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


def post(address, payload, content_type="application/json"):
    data = json.dumps(payload).encode()
    request = urllib.request.Request(address + "/demo/parse", data=data, headers={"Content-Type": content_type})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as response:
        return response.code, json.load(response)


def test_page_and_model_information(address):
    with urllib.request.urlopen(address, timeout=10) as response:
        assert response.status == 200 and b"CardQuery" in response.read()
    with urllib.request.urlopen(address + "/demo/info", timeout=10) as response:
        info = json.load(response)
        assert info["n_cards"] == 517
        assert len(info["vocabulary"]["names"]) == 250


@pytest.mark.parametrize("case", demo_cases())
def test_live_model_examples(address, case):
    status, body = post(address, {"query": case["query"]})
    assert status == case.get("expected_status", 200)
    if "expected" in case:
        assert body["filters"] == case["expected"]
        assert "cards" in body["preview"] and "total" in body["preview"]
    else:
        assert "detail" in body


@pytest.mark.parametrize("payload", [{}, {"query": ""}, {"query": 1}, {"query": "cards", "extra": 1}])
def test_invalid_requests_are_422(address, payload):
    assert post(address, payload)[0] == 422


def test_wrong_content_type_is_422(address):
    assert post(address, {"query": "cards"}, "text/plain")[0] == 422


def test_unknown_route_is_404(address):
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(address + "/does-not-exist", timeout=10)
    assert error.value.code == 404
