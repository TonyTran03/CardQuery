"""Offline tests for Part 4's search page. No backend source code or service needed."""

import copy
import html

import pytest
import requests

from frontend import client
from frontend.app import card_view, create_app, describe_filters
from frontend.fixtures import DEMO_CARDS, GOLDEN_FILTERS, GOLDEN_QUERY, GOLDEN_RESPONSE


class FakeResponse:
    def __init__(self, status_code, payload=None, raw=None):
        self.status_code = status_code
        self._payload = payload
        self._raw = raw

    def json(self):
        if self._raw is not None:
            raise ValueError("not JSON")
        return copy.deepcopy(self._payload)


@pytest.fixture
def page(monkeypatch):
    """Return (get(query, **params) -> (status, html), list of recorded API calls)."""
    calls = []
    state = {"respond": lambda **kw: FakeResponse(200, GOLDEN_RESPONSE)}

    def fake_post(url, json=None, timeout=None, **kwargs):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return state["respond"](url=url, json=json)

    monkeypatch.setattr(client.requests, "post", fake_post)
    monkeypatch.setenv("API_BASE_URL", "http://api.test:8000")
    test_client = create_app().test_client()

    def get(query=None, **params):
        if query is not None:
            params["q"] = query
        resp = test_client.get("/", query_string=params)
        return resp.status_code, resp.get_data(as_text=True)

    def respond_with(fn):
        state["respond"] = fn

    get.respond_with = respond_with
    return get, calls


def text_of(body: str) -> str:
    """Rendered text with tags stripped and whitespace collapsed (for assertions)."""
    import re
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", body)).split())


# --- golden path -----------------------------------------------------------

def test_golden_query_renders_expected_card_and_filters(page):
    get, calls = page
    status, body = get(GOLDEN_QUERY)

    assert status == 200
    assert calls == [{
        "url": "http://api.test:8000/search",
        "json": {"query": GOLDEN_QUERY, "limit": 20, "offset": 0},
        "timeout": 10,
    }]
    text = text_of(body)
    assert "1 result" in text and "1 results" not in text
    for expected in ("Charizard", "Fire", "180 HP", "Rare", "Demo A", "2020", "ID: demo-001"):
        assert expected in text
    assert "Interpreted filters: name Charizard, type Fire, HP >= 150" in text
    assert "No image" in text  # image_url is null -> placeholder
    assert f'value="{GOLDEN_QUERY}"' in body  # query kept in the box


def test_page_has_form_loading_indicator_and_no_error_initially(page):
    get, calls = page
    status, body = get()
    assert status == 200 and calls == []
    assert 'id="search-form"' in body and 'name="q"' in body
    assert 'id="loading"' in body and "hidden" in body
    assert 'id="error"' not in body and 'id="results"' not in body


def test_api_base_url_trailing_slash_is_ignored(page, monkeypatch):
    get, calls = page
    monkeypatch.setenv("API_BASE_URL", "http://api:8000/")
    get(GOLDEN_QUERY)
    assert calls[0]["url"] == "http://api:8000/search"


def test_query_is_trimmed_before_sending(page):
    get, calls = page
    get(f"   {GOLDEN_QUERY}  ")
    assert calls[0]["json"]["query"] == GOLDEN_QUERY


# --- empty and error states -------------------------------------------------

def test_zero_results_shows_friendly_message_and_filters(page):
    get, _ = page
    empty = {**GOLDEN_RESPONSE, "cards": [], "total": 0}
    get.respond_with(lambda **kw: FakeResponse(200, empty))
    _, body = get("water charizards")
    text = text_of(body)
    assert "0 results" in text
    assert "No cards matched your search" in text
    assert "Interpreted filters:" in text


@pytest.mark.parametrize("status,payload,expected", [
    (422, {"detail": [{"loc": ["body", "query"], "msg": "Query must be 1 to 500 characters",
                       "type": "value_error"}]},
     "We couldn't understand that search: Query must be 1 to 500 characters."),
    (422, {"detail": "unsupported query"}, "We couldn't understand that search: unsupported query."),
    (503, {"detail": "model_unavailable"}, "The search model is temporarily unavailable"),
    (500, {"detail": "Internal Server Error"}, "ran into a problem (HTTP 500)"),
    (502, None, "ran into a problem (HTTP 502)"),
])
def test_api_errors_show_friendly_message_and_keep_query(page, status, payload, expected):
    get, _ = page
    get.respond_with(lambda **kw: FakeResponse(status, payload))
    status_code, body = get("my lost query")
    assert status_code == 200
    assert 'id="error"' in body and expected in text_of(body)
    assert 'value="my lost query"' in body
    assert 'id="results"' not in body


@pytest.mark.parametrize("exc,expected", [
    (requests.Timeout(), "took too long"),
    (requests.ConnectionError(), "couldn't reach the search service"),
])
def test_network_failures_show_friendly_message(page, exc, expected):
    get, _ = page

    def boom(**kw):
        raise exc

    get.respond_with(boom)
    _, body = get("pikachu")
    assert expected in text_of(body)
    assert 'value="pikachu"' in body


@pytest.mark.parametrize("response", [
    FakeResponse(200, raw="<html>oops</html>"),
    FakeResponse(200, {"cards": []}),
    FakeResponse(200, {"filters": {}, "cards": [], "total": "1"}),
])
def test_malformed_success_response_is_an_error_not_a_crash(page, response):
    get, _ = page
    get.respond_with(lambda **kw: response)
    status, body = get("pikachu")
    assert status == 200
    assert "couldn't read" in text_of(body)


def test_blank_query_is_rejected_without_calling_api(page):
    get, calls = page
    _, body = get("   ")
    assert calls == []
    assert "Please type something to search for." in text_of(body)


def test_overlong_query_is_rejected_without_calling_api(page):
    get, calls = page
    _, body = get("x" * 501)
    assert calls == []
    assert "too long (501 characters)" in text_of(body)


# --- safety and display -----------------------------------------------------

def test_query_and_card_text_are_html_escaped(page):
    get, _ = page
    evil_card = {**DEMO_CARDS[0], "name": "<img src=x onerror=alert(1)>",
                 "set_name": "<b>Set</b>"}
    get.respond_with(lambda **kw: FakeResponse(
        200, {**GOLDEN_RESPONSE, "cards": [evil_card]}))
    _, body = get('"><script>alert(1)</script>')
    assert "<script>alert(1)</script>" not in body
    assert "<img src=x" not in body
    assert "<b>Set</b>" not in body
    assert "&lt;script&gt;" in body


def test_card_with_image_url_renders_image():
    view = card_view({**DEMO_CARDS[0], "image_url": "https://images.example/c.png"})
    assert view["image_url"] == "https://images.example/c.png"


@pytest.mark.parametrize("url", [None, "", "javascript:alert(1)", "ftp://x/y.png"])
def test_non_http_image_urls_use_placeholder(url):
    assert card_view({**DEMO_CARDS[0], "image_url": url})["image_url"] is None


def test_missing_card_values_get_friendly_text():
    view = card_view({**DEMO_CARDS[0], "hp": None, "rarity": None,
                      "release_year": None, "types": []})
    assert (view["hp"], view["rarity"], view["year"], view["types"]) == (
        "HP unknown", "Rarity unknown", "Year unknown", "No type")


def test_describe_filters_golden():
    assert describe_filters(GOLDEN_FILTERS) == "name Charizard, type Fire, HP >= 150"


def test_describe_filters_all_fields_and_empty():
    filters = {"name": None, "types": ["Fire", "Water"], "hp_min": 60, "hp_max": 120,
               "rarity": "Rare", "year_min": 2018, "year_max": 2018}
    assert describe_filters(filters) == (
        "type Fire or Water, HP >= 60, HP <= 120, rarity Rare, year = 2018")
    empty = dict.fromkeys(GOLDEN_FILTERS) | {"types": []}
    assert describe_filters(empty) == "none (matching every card)"


def test_pagination_links_when_more_results_exist(page):
    get, calls = page
    cards = [{**DEMO_CARDS[0], "card_id": f"c-{i:03d}"} for i in range(20)]
    get.respond_with(lambda **kw: FakeResponse(
        200, {**GOLDEN_RESPONSE, "cards": cards, "total": 45}))
    _, body = get("fire", offset="20")
    assert calls[0]["json"]["offset"] == 20
    text = text_of(body)
    assert "45 results" in text and "Showing 21–40 of 45" in text
    assert "offset=0" in body and "offset=40" in body


def test_bad_offset_falls_back_to_zero(page):
    get, calls = page
    get("fire", offset="-5")
    get("fire", offset="abc")
    assert [c["json"]["offset"] for c in calls] == [0, 0]


def test_health_and_static_assets():
    test_client = create_app().test_client()
    assert test_client.get("/health").get_json() == {"status": "ok"}
    assert test_client.get("/static/app.js").status_code == 200
    assert test_client.get("/static/style.css").status_code == 200


# --- over real HTTP, against the stdlib fake API ---------------------------------

@pytest.fixture
def live_api(monkeypatch):
    from frontend.fake_api import start_server
    server = start_server()
    monkeypatch.setenv("API_BASE_URL", f"http://127.0.0.1:{server.server_address[1]}")
    yield create_app().test_client()
    server.shutdown()


def test_golden_query_over_http(live_api):
    body = live_api.get("/", query_string={"q": GOLDEN_QUERY}).get_data(as_text=True)
    text = text_of(body)
    assert "1 result" in text and "ID: demo-001" in text
    assert "name Charizard, type Fire, HP >= 150" in text


def test_503_over_http(live_api):
    body = live_api.get("/", query_string={"q": "trigger 503"}).get_data(as_text=True)
    assert "temporarily unavailable" in text_of(body)
