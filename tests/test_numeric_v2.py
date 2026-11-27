"""Natural numeric phrasing and exact inclusive/exclusive boundaries."""

import pytest

from parser.contracts import InvalidQueryError
from parser.numeric_v2 import parse_numeric_v2
from parser.semantic import EntityResolver


@pytest.mark.parametrize("query,expected", [
    ("charizard more than 150 hp", {"hp_min": 151}),
    ("charizard greater than 150 health points", {"hp_min": 151}),
    ("HP > 150", {"hp_min": 151}), ("HP>=150", {"hp_min": 150}),
    ("at least 150 hit points", {"hp_min": 150}), ("no less than 150 hp", {"hp_min": 150}),
    ("less than 150 hps", {"hp_max": 149}), ("HP < 150", {"hp_max": 149}),
    ("HP <= 150", {"hp_max": 150}), ("no more than 150 hp", {"hp_max": 150}),
    ("no greater than 150 hp", {"hp_max": 150}), ("150+ hp", {"hp_min": 150}),
    ("150 hp exactly", {"hp_min": 150, "hp_max": 150}),
    ("between 120 and 180 hp", {"hp_min": 120, "hp_max": 180}),
    ("HP between 120 and 180", {"hp_min": 120, "hp_max": 180}),
    ("120-180hp", {"hp_min": 120, "hp_max": 180}),
    ("released after 2020", {"year_min": 2021}), ("released since 2020", {"year_min": 2020}),
    ("older than 2000", {"year_max": 1999}), ("from the year 2000", {"year_min": 2000, "year_max": 2000}),
    ("between 1999 and 2020", {"year_min": 1999, "year_max": 2020}),
    ("between 50 and 100 hp between 1999 and 2020", {"hp_min": 50, "hp_max": 100, "year_min": 1999, "year_max": 2020}),
    ("more than 150 hp and less than 200 hp", {"hp_min": 151, "hp_max": 199}),
    ("at most 60 hp between 2016 and 2018", {"hp_max": 60, "year_min": 2016, "year_max": 2018}),
    ("under 150 hp between 1999 and 2001", {"hp_max": 149, "year_min": 1999, "year_max": 2001}),
])
def test_comparison_meaning(query, expected):
    assert parse_numeric_v2(query) == {
        **dict.fromkeys(["hp_min", "hp_max", "year_min", "year_max"]), **expected,
    }


@pytest.mark.parametrize("query", [
    "less than 0 hp", "more than -1 hp", "more than 2.5 hp", "hp < 0",
    "between 180 and 120 hp", "hp between 180 and 120", "more than 150 hp and less than 151 hp",
    "after 9999", "before 0001", "since 0000", "from 10000", "after 20",
])
def test_invalid_and_contradictory_bounds(query):
    with pytest.raises(InvalidQueryError):
        parse_numeric_v2(query)


@pytest.fixture
def entities():
    return EntityResolver(["Charizard", "Charizard ex", "Charizard-EX", "M Charizard-EX", "Pikachu", "Flaaffy", "Nidoran ♀"])


@pytest.mark.parametrize("query,name", [
    ("charizard more than 150 hp", "Charizard"), ("charizards", "Charizard"),
    ("Charizard ex", "Charizard ex"), ("M Charizard-EX", "M Charizard-EX"),
    ("Charizard-EX", "Charizard-EX"), ("Nidoran female", "Nidoran ♀"),
    ("charzard more than 150 hp", "Charizard"),
    ("winston", None), ("hello", None), ("water pokemon", None),
])
def test_entities_are_grounded_and_variants_stay_distinct(entities, query, name):
    assert entities.resolve(query)[0] == name


def test_multiple_names_rejected(entities):
    with pytest.raises(InvalidQueryError, match="one Pokémon"):
        entities.resolve("charizard and pikachu")


@pytest.mark.parametrize("query", ["no less than 150 hp", "not less than 150 hp", "less than 150 hp"])
def test_api_normalization_preserves_comparison_direction(query):
    from api.query_interpreter import normalize_search_query

    assert parse_numeric_v2(normalize_search_query(query)) == parse_numeric_v2(query)
