"""Reproducible synthetic labels grounded in the committed Part 1 catalog.

Labels are built from recipe choices, never from model predictions or the
numeric parser. This script is optional; the generated CSVs are committed.
"""

from __future__ import annotations

import argparse
import itertools
import random
from collections import defaultdict
from pathlib import Path

from catalog.cards import load_cards

from .contracts import empty_filters
from .data import assert_disjoint, write_rows

ROOT = Path(__file__).resolve().parents[1]
PREFIXES = ["", "show me ", "find ", "search for ", "i want ", "list ", "get ", "please show "]


def _hp(rng):
    value = rng.choice([30, 50, 60, 80, 100, 120, 150, 180, 200, 250, 300])
    mode = rng.randrange(6)
    if mode == 0:
        return "", {}
    if mode == 1:
        return f" with at least {value} hp", {"hp_min": value}
    if mode == 2:
        return f" with over {value} hp", {"hp_min": value + 1}
    if mode == 3:
        return f" with at most {value} hp", {"hp_max": value}
    if mode == 4:
        return f" with under {value} hp", {"hp_max": value - 1}
    return f" with between {value} and {value + 70} hp", {"hp_min": value, "hp_max": value + 70}


def _year(rng):
    year = rng.choice([1999, 2000, 2004, 2010, 2016, 2018, 2020, 2023])
    mode = rng.randrange(5)
    if mode == 0:
        return "", {}
    if mode == 1:
        return f" after {year}", {"year_min": year + 1}
    if mode == 2:
        return f" before {year}", {"year_max": year - 1}
    if mode == 3:
        return f" from {year}", {"year_min": year, "year_max": year}
    return f" between {year} and {year + 2}", {"year_min": year, "year_max": year + 2}


def _row(rng, name=None, types=(), rarity=None, *, prefix=None, numeric=True, plural=False):
    hp_text, hp = _hp(rng) if numeric else ("", {})
    year_text, year = _year(rng) if numeric else ("", {})
    filters = {**empty_filters(), "name": name, "types": list(types), "rarity": rarity, **hp, **year}
    text_name = name or ""
    if plural and name and name.isascii() and name.isalpha():
        text_name += "s"
    components = [rarity or "", " or ".join(types), text_name, "cards"]
    text = " ".join(item for item in components if item)
    return {"query": (rng.choice(PREFIXES) if prefix is None else prefix) + text + hp_text + year_text,
            "filters": filters}


def build_examples(cards: list[dict], seed=42):
    rng = random.Random(seed)
    by_name = defaultdict(list)
    for card in cards:
        by_name[card["name"]].append(card)
    types = sorted({t for card in cards for t in card["types"]})
    rarities = sorted({card["rarity"] for card in cards if card["rarity"]})
    training = []
    for name in sorted(by_name):
        for i in range(48):
            card = rng.choice(by_name[name])
            training.append(_row(rng, name,
                                 card["types"] if i % 4 in (1, 3) else (),
                                 card["rarity"] if i % 4 in (2, 3) else None,
                                 plural=i % 3 == 0))
        # Short queries avoid learning that a long template is necessary.
        for prefix in PREFIXES:
            training.append(_row(rng, name, prefix=prefix, numeric=False))
    # Explicit, independent examples prevent inferring an unmentioned type or
    # rarity from the Pokémon name. Include multi-type OR queries as required.
    for selected_types in itertools.chain(((t,) for t in types), itertools.combinations(types, 2)):
        for i in range(32):
            training.append(_row(rng, types=selected_types,
                                 rarity=rng.choice(rarities) if i % 4 == 0 else None))
        for prefix in PREFIXES[:4]:
            training.append(_row(rng, types=selected_types, prefix=prefix, numeric=False))
    for rarity in rarities:
        for i in range(80):
            training.append(_row(rng, rarity=rarity, types=[rng.choice(types)] if i % 3 == 0 else ()))
        for prefix in PREFIXES:
            training.append(_row(rng, rarity=rarity, prefix=prefix, numeric=False))
    for _ in range(180):
        training.append(_row(rng))
    # Ordinary plural phrasing, without the optional word "cards".
    for name in sorted(by_name):
        if name.isascii() and name.isalpha():
            for i in range(8):
                card = rng.choice(by_name[name])
                row = _row(rng, name, card["types"] if i % 2 else (), plural=True)
                row["query"] = row["query"].replace(" cards", "")
                training.append(row)
    training.append({"query": "fire charizards with at least 150 hp", "filters": {
        **empty_filters(), "name": "Charizard", "types": ["Fire"], "hp_min": 150}})

    normal, drift = [], []
    normal_prefixes = ["could you find ", "please look up ", "i am looking for "]
    drift_prefixes = [
        "for the collection i am organizing this evening i would really appreciate it if you could look through the catalog and find ",
        "as part of my ongoing trading card collection project please carefully review the available entries and help me locate ",
        "i have been searching through my collection notes for quite a while and would now like you to show me all of the following ",
    ]
    for i in range(160):
        card = rng.choice(cards)
        kwargs = {"name": card["name"] if i % 4 != 0 else None,
                  "types": card["types"] if i % 3 != 0 else (),
                  "rarity": card["rarity"] if i % 2 else None}
        normal.append(_row(rng, **kwargs, prefix=rng.choice(normal_prefixes)))
        drift.append(_row(rng, **kwargs, prefix=rng.choice(drift_prefixes)))
    assert_disjoint(training, normal, drift)
    return training, normal, drift


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--catalog", type=Path, default=ROOT / "data/cards.csv")
    cli.add_argument("--out-dir", type=Path, default=ROOT / "data")
    args = cli.parse_args()
    for name, rows in zip(("query_examples", "unseen_normal", "unseen_drift"), build_examples(load_cards(str(args.catalog)))):
        write_rows(args.out_dir / f"{name}.csv", rows)
        print(f"Wrote {name}.csv")


if __name__ == "__main__":
    main()
