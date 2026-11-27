"""Add labeled natural paraphrases without using predictions as labels."""

import itertools
import random
from pathlib import Path

from catalog.cards import load_cards
from parser.contracts import empty_filters
from parser.data import assert_disjoint, load_rows, write_rows

ROOT = Path(__file__).resolve().parents[1]
TYPE_FORMS = {
    "Fire": ["fire", "flame", "fiery", "flames", "heat-based"],
    "Water": ["water", "aquatic", "watery", "water-based"],
    "Grass": ["grass", "plant", "leafy", "grass-type"],
    "Lightning": ["lightning", "electric", "electricity", "electric-type"],
    "Psychic": ["psychic", "telepathic", "mind-powered"],
    "Fighting": ["fighting", "martial arts", "combat"],
    "Darkness": ["darkness", "dark", "shadow", "dark-type"],
    "Metal": ["metal", "steel", "metallic"],
    "Colorless": ["colorless", "normal-type", "normal element"],
    "Dragon": ["dragon", "draconic", "dragon-type"],
    "Fairy": ["fairy", "fairy-type", "fairy element"],
}
TEMPLATES = ["{kind} pokemon", "i need cards with {kind} abilities", "looking for {kind} cards for a deck",
             "find creatures of the {kind} element", "do you have any {kind} pokemon cards",
             "cards related to {kind}", "pokemon using {kind} attacks", "a selection of {kind} creatures"]


def build():
    rows = load_rows(ROOT / "data/query_examples.csv")
    cards = load_cards(ROOT / "data/cards.csv")
    rng = random.Random(782)

    def add(query, **fields):
        rows.append({"query": query, "filters": {**empty_filters(), **fields}})

    for kind, forms in TYPE_FORMS.items():
        for form, template, number in itertools.product(forms, TEMPLATES, [None, 75, 175]):
            query = template.format(kind=form)
            fields = {"types": [kind]}
            if number is not None:
                query += f" with more than {number} hp"
                fields["hp_min"] = number + 1
            add(query, **fields)
    for left, right in itertools.combinations(TYPE_FORMS, 2):
        for template in TEMPLATES:
            phrase = rng.choice(TYPE_FORMS[left]) + " or " + rng.choice(TYPE_FORMS[right])
            add(template.format(kind=phrase), types=[left, right])
    numeric = [
        ("more than 120 hp", {"hp_min": 121}),
        ("less than 180 health points", {"hp_max": 179}),
        ("hp above 90", {"hp_min": 91}),
        ("no more than 200 hit points", {"hp_max": 200}),
        ("hp >= 150", {"hp_min": 150}),
        ("150+ hp", {"hp_min": 150}),
        ("hp between 80 and 180", {"hp_min": 80, "hp_max": 180}),
        ("released since 2000", {"year_min": 2000}),
    ]
    for name in sorted({card["name"] for card in cards}):
        for phrase, fields in numeric:
            add(f"i would like to find {name} with {phrase}", name=name, **fields)
        for index in range(3):
            kind = rng.choice(list(TYPE_FORMS))
            form = rng.choice(TYPE_FORMS[kind])
            add(f"could you look for {name} cards with {form} powers for my deck {index}",
                name=name, types=[kind])
    for rarity in sorted({card["rarity"] for card in cards if card["rarity"]}):
        for template in ["i want {rarity} pokemon", "can you find {rarity} cards", "cards of {rarity} rarity"]:
            add(template.format(rarity=rarity), rarity=rarity)
    for query in ["cards", "pokemon", "all pokemon cards", "show all available pokemon", "please browse the cards"]:
        add(query)
    for word in ["alfred", "timothy", "zurblo", "zedoria", "sunshine", "weather", "sandwich", "football"]:
        for prefix in ["", "show me ", "find ", "pokemon called "]:
            add(prefix + word)
    assert_disjoint(rows, load_rows(ROOT / "data/unseen_normal.csv"), load_rows(ROOT / "data/unseen_drift.csv"))
    write_rows(ROOT / "data/query_examples_semantic.csv", rows)
    return len(load_rows(ROOT / "data/query_examples_semantic.csv"))


if __name__ == "__main__":
    print(f"Wrote {build()} labeled semantic training examples")
