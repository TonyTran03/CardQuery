"""Shared handoff examples (team guide, page 3).

The integrator publishes ``shared/fixtures.py``; until then we keep an exact copy
so Part 4 runs and tests on its own.
"""

try:  # pragma: no cover - depends on the integrator's package
    from shared.fixtures import DEMO_CARDS, GOLDEN_FILTERS, GOLDEN_QUERY
except ImportError:
    DEMO_CARDS = [
        {"card_id": "demo-001", "name": "Charizard", "types": ["Fire"], "hp": 180,
         "rarity": "Rare", "set_name": "Demo A", "release_year": 2020, "image_url": None},
        {"card_id": "demo-002", "name": "Pikachu", "types": ["Lightning"], "hp": 60,
         "rarity": "Common", "set_name": "Demo B", "release_year": 2018, "image_url": None},
        {"card_id": "demo-003", "name": "Charizard", "types": ["Fire"], "hp": 120,
         "rarity": "Rare", "set_name": "Demo C", "release_year": 2016, "image_url": None},
    ]

    GOLDEN_QUERY = "fire charizards with at least 150 hp"

    GOLDEN_FILTERS = {
        "name": "Charizard",
        "types": ["Fire"],
        "hp_min": 150, "hp_max": None,
        "rarity": None,
        "year_min": None, "year_max": None,
    }

FILTER_KEYS = ("name", "types", "hp_min", "hp_max", "rarity", "year_min", "year_max")

GOLDEN_RESPONSE = {
    "filters": GOLDEN_FILTERS,
    "cards": [DEMO_CARDS[0]],
    "total": 1,
    "model_version": "stub-v1",
}
