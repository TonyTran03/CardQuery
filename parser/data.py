"""CSV labels, canonicalization, duplicate detection, and the agreed split."""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

from .contracts import normalize_query, validate_filters


def label_json(filters: dict) -> str:
    return json.dumps(validate_filters(filters), sort_keys=True, ensure_ascii=False)


def prepare_rows(rows: list[dict]) -> list[dict]:
    """Deduplicate normalized query text; reject conflicting labels and schemas."""
    if not isinstance(rows, list) or not rows:
        raise ValueError("rows must be a nonempty list of labeled queries")
    unique = {}
    spellings = {field: {} for field in ("name", "rarity", "types")}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"query", "filters"}:
            raise ValueError("each row must have exactly query and filters (decode filters_json first)")
        query = normalize_query(row["query"])
        filters = validate_filters(row["filters"])
        # Case-only spelling differences share one canonical label.
        for field in ("name", "rarity", "types"):
            values = filters[field] if field == "types" else [filters[field]]
            for value in values:
                if value is not None:
                    spellings[field].setdefault(value.casefold(), value)
        if query in unique and _casefold_label(unique[query]) != _casefold_label(filters):
            raise ValueError(f"conflicting labels for normalized query: {query!r}")
        unique.setdefault(query, filters)
    result = []
    for query in sorted(unique):
        filters = unique[query].copy()
        for field in ("name", "rarity"):
            if filters[field] is not None:
                filters[field] = spellings[field][filters[field].casefold()]
        filters["types"] = [spellings["types"][v.casefold()] for v in filters["types"]]
        result.append({"query": query, "filters": validate_filters(filters)})
    return result


def _casefold_label(filters):
    return {key: sorted(v.casefold() for v in value) if isinstance(value, list)
            else value.casefold() if isinstance(value, str) else value
            for key, value in filters.items()}


def load_rows(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["query", "filters_json"]:
            raise ValueError("query CSV must have exactly query,filters_json as its header")
        rows = []
        for line, row in enumerate(reader, 2):
            try:
                if None in row or row["filters_json"] is None:
                    raise ValueError("incorrect number of CSV cells")
                rows.append({"query": row["query"], "filters": json.loads(row["filters_json"])})
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}, line {line}: {exc}") from exc
    return prepare_rows(rows)


def write_rows(path: str | Path, rows: list[dict]) -> None:
    rows = prepare_rows(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["query", "filters_json"])
        writer.writerows((row["query"], label_json(row["filters"])) for row in rows)


def split_rows(rows: list[dict], seed: int = 42) -> tuple[list, list, list]:
    rows = prepare_rows(rows)
    if len(rows) < 5:
        raise ValueError("at least five distinct queries are required for a 60/20/20 split")
    random.Random(seed).shuffle(rows)
    size = math.ceil(0.2 * len(rows))
    test, validation, train = rows[:size], rows[size:2 * size], rows[2 * size:]
    return train, validation, test


def majority_label(rows: list[dict]) -> dict:
    counts = Counter(label_json(row["filters"]) for row in prepare_rows(rows))
    winner = min(counts, key=lambda label: (-counts[label], label))
    return json.loads(winner)


def assert_disjoint(*pools: list[dict]) -> None:
    seen = set()
    for pool in pools:
        queries = {row["query"] for row in prepare_rows(pool)}
        overlap = queries & seen
        if overlap:
            raise ValueError(f"query pools overlap: {sorted(overlap)[:3]}")
        seen.update(queries)
