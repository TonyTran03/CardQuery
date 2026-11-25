"""Full-filter exact match and micro-F1 over non-null slots, per team rules."""

from __future__ import annotations

from .contracts import validate_filters
from .data import prepare_rows
from .parser import parse_queries


def filter_slots(filters: dict) -> set[tuple]:
    filters = validate_filters(filters)
    return {(key, value) for key, value in filters.items() if key != "types" and value is not None} | {
        ("types", value) for value in filters["types"]}


def score_predictions(predictions: list[dict], labels: list[dict]) -> dict:
    if not labels or len(predictions) != len(labels):
        raise ValueError("predictions and labels must have the same nonzero length")
    tp = fp = fn = matches = 0
    for prediction, label in zip(predictions, labels):
        prediction, label = validate_filters(prediction), validate_filters(label)
        matches += prediction == label
        predicted, expected = filter_slots(prediction), filter_slots(label)
        tp += len(predicted & expected)
        fp += len(predicted - expected)
        fn += len(expected - predicted)
    denominator = 2 * tp + fp + fn
    return {"exact_match": matches / len(labels), "micro_f1": 2 * tp / denominator if denominator else 1.0}


def evaluate_parser(model, rows: list[dict]) -> dict:
    rows = prepare_rows(rows)
    # Reuse the canonical spelling learned from labels, including sorted types.
    maps = {field: {value.casefold(): value for value in model["vocabulary"][key]}
            for field, key in (("name", "names"), ("rarity", "rarities"), ("types", "types"))}
    labels = []
    for row in rows:
        label = row["filters"].copy()
        for field in ("name", "rarity"):
            if label[field] is not None:
                label[field] = maps[field].get(label[field].casefold(), label[field])
        label["types"] = [maps["types"].get(t.casefold(), t) for t in label["types"]]
        labels.append(validate_filters(label))
    predictions = parse_queries(model, [row["query"] for row in rows])
    return score_predictions(predictions, labels)
