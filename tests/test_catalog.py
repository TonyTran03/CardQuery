"""Offline tests for Part 1 (card catalog)."""

import copy
import csv
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from catalog.cards import CARD_FIELDS, load_cards, normalize_card, write_cards_csv
from catalog.pokemontcg_source import build_cards, map_source_card

try:  # the integrator publishes shared/fixtures.py
    from shared.fixtures import DEMO_CARDS
except ImportError:  # same three cards, copied from the team guide (page 3)
    DEMO_CARDS = [
        {"card_id": "demo-001", "name": "Charizard", "types": ["Fire"], "hp": 180,
         "rarity": "Rare", "set_name": "Demo A", "release_year": 2020, "image_url": None},
        {"card_id": "demo-002", "name": "Pikachu", "types": ["Lightning"], "hp": 60,
         "rarity": "Common", "set_name": "Demo B", "release_year": 2018, "image_url": None},
        {"card_id": "demo-003", "name": "Charizard", "types": ["Fire"], "hp": 120,
         "rarity": "Rare", "set_name": "Demo C", "release_year": 2016, "image_url": None},
    ]

REAL_CSV = Path(__file__).resolve().parents[1] / "data" / "cards.csv"


def raw_charizard(**overrides):
    raw = {
        "card_id": "demo-001", "name": "Charizard",
        "types": '["Fire"]', "hp": "180", "rarity": "Rare",
        "set_name": "Demo A", "release_year": "2020",
        "image_url": "",
    }
    raw.update(overrides)
    return raw


# ---------------------------------------------------------------- normalize

def test_guide_example_normalizes_to_demo_card():
    assert normalize_card(raw_charizard()) == DEMO_CARDS[0]


def test_types_accepts_python_list():
    assert normalize_card(raw_charizard(types=["Fire"]))["types"] == ["Fire"]


def test_missing_hp_becomes_none_not_invented():
    card = normalize_card(raw_charizard(hp=""))
    assert card["hp"] is None
    raw = raw_charizard()
    del raw["hp"]
    assert normalize_card(raw)["hp"] is None


def test_missing_nullable_fields_and_types():
    raw = {"card_id": "x-1", "name": "Ditto", "set_name": "Demo"}
    assert normalize_card(raw) == {
        "card_id": "x-1", "name": "Ditto", "types": [], "hp": None,
        "rarity": None, "set_name": "Demo", "release_year": None, "image_url": None,
    }


@pytest.mark.parametrize("field,value", [
    ("hp", "abc"), ("hp", "180.5"), ("hp", 180.0), ("hp", True), ("hp", "-10"),
    ("release_year", "0"), ("release_year", "10000"), ("release_year", "20x0"),
])
def test_invalid_numeric_raises(field, value):
    with pytest.raises(ValueError):
        normalize_card(raw_charizard(**{field: value}))


@pytest.mark.parametrize("field", ["card_id", "name", "set_name"])
@pytest.mark.parametrize("value", ["", "   ", None, 123])
def test_invalid_required_text_raises(field, value):
    with pytest.raises(ValueError):
        normalize_card(raw_charizard(**{field: value}))


@pytest.mark.parametrize("value", ["ftp://x.com/a.png", "not a url", "[Fire]"])
def test_invalid_url_or_types_raises(value):
    field = "types" if value.startswith("[") else "image_url"
    with pytest.raises(ValueError):
        normalize_card(raw_charizard(**{field: value}))


def test_unknown_field_raises():
    with pytest.raises(ValueError):
        normalize_card(raw_charizard(pokemon_name="Charizard"))


# ---------------------------------------------------------------- CSV

def test_demo_cards_round_trip(tmp_path):
    original = copy.deepcopy(DEMO_CARDS)
    path = tmp_path / "cards.csv"
    write_cards_csv(DEMO_CARDS, str(path))
    assert load_cards(str(path)) == DEMO_CARDS
    assert DEMO_CARDS == original  # writing does not mutate input


def test_csv_format_header_blanks_and_types_cell(tmp_path):
    path = tmp_path / "cards.csv"
    write_cards_csv(DEMO_CARDS, str(path))
    text = path.read_text(encoding="utf-8")
    assert text.splitlines()[0] == ",".join(CARD_FIELDS)
    assert "None" not in text and "null" not in text
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    assert rows[0]["types"] == '["Fire"]'
    assert rows[0]["image_url"] == ""


def test_round_trip_keeps_missing_values_and_multi_types(tmp_path):
    cards = [
        {"card_id": "b", "name": "Flabébé", "types": ["Fairy", "Psychic"], "hp": None,
         "rarity": None, "set_name": "Set, with comma", "release_year": None,
         "image_url": "https://images.example.com/b.png"},
        {"card_id": "a", "name": "Ditto", "types": [], "hp": 0,
         "rarity": "Common", "set_name": "S", "release_year": 1999, "image_url": None},
    ]
    path = tmp_path / "cards.csv"
    write_cards_csv(cards, str(path))
    assert load_cards(str(path)) == sorted(cards, key=lambda c: c["card_id"])


def test_load_sorts_by_card_id(tmp_path):
    path = tmp_path / "cards.csv"
    write_cards_csv(list(reversed(DEMO_CARDS)), str(path))
    assert [c["card_id"] for c in load_cards(str(path))] == ["demo-001", "demo-002", "demo-003"]


def test_write_rejects_duplicate_ids(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        write_cards_csv([DEMO_CARDS[0], DEMO_CARDS[0]], str(tmp_path / "cards.csv"))


def test_load_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "cards.csv"
    write_cards_csv(DEMO_CARDS, str(path))
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines + [lines[1]]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_cards(str(path))


def test_load_rejects_wrong_header(tmp_path):
    path = tmp_path / "cards.csv"
    path.write_text("name,card_id\nCharizard,demo-001\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        load_cards(str(path))


def test_load_reports_line_of_bad_value(tmp_path):
    path = tmp_path / "cards.csv"
    write_cards_csv(DEMO_CARDS, str(path))
    text = path.read_text(encoding="utf-8").replace(",180,", ",lots,")
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        load_cards(str(path))


# ---------------------------------------------------------------- source adapter (API)

def api_card(**overrides):
    raw = {
        "id": "base1-4", "name": "Charizard", "supertype": "Pokémon",
        "types": ["Fire"], "hp": "120", "rarity": "Rare Holo",
        "set": {"id": "base1", "name": "Base", "releaseDate": "1999/01/09"},
        "images": {"small": "https://images.pokemontcg.io/base1/4.png"},
    }
    raw.update(overrides)
    return raw


def test_map_source_card_converts_source_formats():
    assert map_source_card(api_card()) == {
        "card_id": "base1-4", "name": "Charizard", "types": ["Fire"], "hp": 120,
        "rarity": "Rare Holo", "set_name": "Base", "release_year": 1999,
        "image_url": "https://images.pokemontcg.io/base1/4.png",
    }


def test_map_source_card_keeps_missing_values_missing():
    raw = {"id": "x-1", "name": "Mystery", "supertype": "Pokémon", "set": {"name": "X Set"}}
    card = map_source_card(raw)
    assert card["types"] == [] and card["hp"] is None
    assert card["rarity"] is None and card["release_year"] is None and card["image_url"] is None


class FakeApi:
    """Serves pages like api.pokemontcg.io/v2/cards, without any network."""

    def __init__(self, cards_by_set, page_size=2):
        self.cards_by_set = cards_by_set
        self.page_size = page_size
        self.urls = []

    def __call__(self, url, api_key=None):
        self.urls.append(url)
        params = parse_qs(urlparse(url).query)
        set_id = params["q"][0].split(":", 1)[1]
        page = int(params["page"][0])
        cards = self.cards_by_set.get(set_id, [])
        start = (page - 1) * self.page_size
        return {"data": cards[start:start + self.page_size], "page": page,
                "pageSize": self.page_size, "totalCount": len(cards)}


def test_build_cards_paginates_and_keeps_only_pokemon():
    fake = FakeApi({"base1": [
        api_card(id="base1-1"), api_card(id="base1-2"),
        api_card(id="base1-80", name="Potion", supertype="Trainer", hp=None, types=None),
        api_card(id="base1-3"),
    ]})
    cards, raw = build_cards(["base1"], fetch_json=fake)
    assert [c["card_id"] for c in cards] == ["base1-1", "base1-2", "base1-3"]
    assert len(raw) == 3
    assert len(fake.urls) == 2  # 4 records at page size 2 -> two pages


def test_build_cards_fails_loudly_on_empty_set():
    with pytest.raises(ValueError, match="no cards"):
        build_cards(["nope"], fetch_json=FakeApi({}))


# ---------------------------------------------------------------- committed catalog

@pytest.mark.skipif(not REAL_CSV.exists(), reason="data/cards.csv not built yet")
def test_committed_catalog_is_valid():
    cards = load_cards(str(REAL_CSV))
    assert len(cards) >= 100
    assert len({c["card_id"] for c in cards}) == len(cards)
    assert any(c["name"] == "Charizard" and "Fire" in c["types"] for c in cards)
