"""Lightweight parser for independent workflow testing."""

import json
from collections import Counter
from pathlib import Path

from parser.evaluation import score_predictions


def normalize_query(query):
    return " ".join(query.lower().split())


def train_parser(rows, seed=42, C=1.0):
    if not rows:
        raise ValueError("Training data cannot be empty")

    examples = {normalize_query(row["query"]): row["filters"] for row in rows}

    counts = Counter(json.dumps(row["filters"], sort_keys=True) for row in rows)

    majority = min(
        counts,
        key=lambda label: (-counts[label], label),
    )

    return {
        "parser_kind": "fake-parser-v1",
        "numeric_rules_version": "fake-v1",
        "seed": seed,
        "C": C,
        "examples": examples,
        "default": json.loads(majority),
    }


def parse_query(model, query):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string")

    filters = model["examples"].get(
        normalize_query(query),
        model["default"],
    )

    return json.loads(json.dumps(filters))


def save_parser(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(model),
        encoding="utf-8",
    )


def load_parser(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evaluate_parser(model, rows):
    predictions = [parse_query(model, row["query"]) for row in rows]

    expected = [row["filters"] for row in rows]

    return score_predictions(predictions, expected)
