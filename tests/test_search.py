import pytest

from frontend.fixtures import DEMO_CARDS, GOLDEN_FILTERS
from parser.contracts import empty_filters
from search.search import CardSearchIndex, search_cards


def test_hp_bound_changes_the_card_matches():
    cards = DEMO_CARDS[::-1]
    assert search_cards(cards, GOLDEN_FILTERS)["cards"] == [DEMO_CARDS[0]]
    assert search_cards(cards, dict(GOLDEN_FILTERS, hp_min=120))["cards"] == [
        DEMO_CARDS[0], DEMO_CARDS[2],
    ]


def test_case_insensitive_matching_missing_values_and_pagination():
    cards = [dict(DEMO_CARDS[0], hp=None), DEMO_CARDS[2], DEMO_CARDS[1]]
    filters = dict(empty_filters(), types=[" fire ", "FIRE", "lightning"], hp_min=60)
    assert search_cards(cards, filters, limit=1, offset=1) == {
        "cards": [DEMO_CARDS[2]], "total": 2, "limit": 1, "offset": 1,
    }
    assert search_cards(cards, dict(filters, types=["Unknown"]))["total"] == 0
    assert search_cards(cards, filters, offset=2)["cards"] == []


def test_index_deduplicates_multi_type_matches_across_pages():
    dual_type = dict(DEMO_CARDS[0], types=["Fire", "Lightning"])
    index = CardSearchIndex([DEMO_CARDS[2], DEMO_CARDS[1], dual_type])
    filters = dict(empty_filters(), types=["Fire", "Lightning"])

    assert index.search(filters, limit=1) == {
        "cards": [dual_type], "total": 3, "limit": 1, "offset": 0,
    }
    assert index.search(filters, limit=1, offset=1) == {
        "cards": [DEMO_CARDS[1]], "total": 3, "limit": 1, "offset": 1,
    }
    assert index.search(dict(filters, name="riza", rarity="rare"))["cards"] == [
        dual_type, DEMO_CARDS[2],
    ]


@pytest.mark.parametrize("pagination", [{"limit": True}, {"limit": 101}, {"offset": -1}])
def test_invalid_pagination_raises_value_error(pagination):
    with pytest.raises(ValueError):
        search_cards(DEMO_CARDS, GOLDEN_FILTERS, **pagination)
