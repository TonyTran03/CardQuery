"""Card catalog: the shared Card format and its CSV handoff (Part 1).

A Card is a dict with exactly these eight keys:

    card_id       non-empty str (unique across the catalog)
    name          non-empty str
    types         list[str] (never None; [] when unknown)
    hp            int >= 0, or None
    rarity        non-empty str, or None
    set_name      non-empty str
    release_year  int in 1..9999, or None
    image_url     http(s) URL str, or None

Missing source values stay None. Nothing is invented.
"""

from __future__ import annotations

import csv
import json
from urllib.parse import urlparse

CARD_FIELDS = (
    "card_id",
    "name",
    "types",
    "hp",
    "rarity",
    "set_name",
    "release_year",
    "image_url",
)

_REQUIRED_TEXT = ("card_id", "name", "set_name")


# --------------------------------------------------------------------------
# Field helpers
# --------------------------------------------------------------------------

def _is_missing(value) -> bool:
    """None or a blank string counts as a missing source value."""
    return value is None or (isinstance(value, str) and value.strip() == "")


def _required_text(raw: dict, field: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(f"{field} must be a non-empty string, got {value!r}")
    return value.strip()


def _optional_text(raw: dict, field: str) -> str | None:
    value = raw.get(field)
    if _is_missing(value):
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string or empty, got {value!r}")
    return value.strip()


def _optional_int(raw: dict, field: str, low: int, high: int | None) -> int | None:
    value = raw.get(field)
    if _is_missing(value):
        return None
    # bool is a subclass of int; reject it explicitly. Floats are rejected too,
    # so "180.5" or 180.0 never silently becomes an HP value.
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer, got {value!r}")
    if isinstance(value, int):
        number = value
    elif isinstance(value, str) and value.strip().lstrip("-").isdigit():
        number = int(value.strip())
    else:
        raise ValueError(f"{field} must be an integer, got {value!r}")
    if number < low or (high is not None and number > high):
        bound = f"{low}..{high}" if high is not None else f">= {low}"
        raise ValueError(f"{field} must be {bound}, got {number}")
    return number


def _types(raw: dict) -> list[str]:
    value = raw.get("types")
    if _is_missing(value):
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"types must be a JSON list string, got {value!r}") from exc
    if not isinstance(value, list):
        raise ValueError(f"types must be a list, got {value!r}")
    cleaned = []
    for item in value:
        if not isinstance(item, str) or item.strip() == "":
            raise ValueError(f"types must contain non-empty strings, got {item!r}")
        cleaned.append(item.strip())
    return cleaned


def _image_url(raw: dict) -> str | None:
    value = _optional_text(raw, "image_url")
    if value is None:
        return None
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"image_url must be an http(s) URL, got {value!r}")
    return value


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def normalize_card(raw: dict) -> dict:
    """Turn one record with canonical field names into a valid Card.

    - types may be a Python list or a JSON list string; missing -> [].
    - Missing nullable fields (blank string or absent) -> None.
    - Invalid card_id / name / set_name, or invalid numbers -> ValueError.
    - Unknown keys -> ValueError (source-specific mapping belongs in the
      source adapter, not here).
    """
    if not isinstance(raw, dict):
        raise ValueError(f"card must be a dict, got {type(raw).__name__}")
    unknown = set(raw) - set(CARD_FIELDS)
    if unknown:
        raise ValueError(f"unknown card fields: {sorted(unknown)}")

    return {
        "card_id": _required_text(raw, "card_id"),
        "name": _required_text(raw, "name"),
        "types": _types(raw),
        "hp": _optional_int(raw, "hp", 0, None),
        "rarity": _optional_text(raw, "rarity"),
        "set_name": _required_text(raw, "set_name"),
        "release_year": _optional_int(raw, "release_year", 1, 9999),
        "image_url": _image_url(raw),
    }


def _check_unique(cards: list[dict]) -> None:
    seen = set()
    for card in cards:
        if card["card_id"] in seen:
            raise ValueError(f"duplicate card_id: {card['card_id']!r}")
        seen.add(card["card_id"])


def _to_cell(value) -> str:
    """Nullable scalars become blank cells, never the text 'None' or 'null'."""
    return "" if value is None else str(value)


def write_cards_csv(cards: list[dict], path: str) -> None:
    """Validate cards and write them to CSV, sorted by card_id.

    types is stored as a JSON list inside a quoted CSV cell.
    """
    normalized = [normalize_card(card) for card in cards]
    _check_unique(normalized)
    normalized.sort(key=lambda c: c["card_id"])

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(CARD_FIELDS)
        for card in normalized:
            writer.writerow([
                card["card_id"],
                card["name"],
                json.dumps(card["types"], ensure_ascii=False),
                _to_cell(card["hp"]),
                _to_cell(card["rarity"]),
                card["set_name"],
                _to_cell(card["release_year"]),
                _to_cell(card["image_url"]),
            ])


def load_cards(csv_path: str) -> list[dict]:
    """Read a cards CSV and return valid Card dicts sorted by card_id."""
    with open(csv_path, encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{csv_path} is empty") from exc
        if tuple(header) != CARD_FIELDS:
            raise ValueError(
                f"unexpected header {header}; expected {list(CARD_FIELDS)}"
            )
        cards = []
        for line_no, row in enumerate(reader, start=2):
            if len(row) != len(CARD_FIELDS):
                raise ValueError(
                    f"line {line_no}: expected {len(CARD_FIELDS)} cells, got {len(row)}"
                )
            try:
                cards.append(normalize_card(dict(zip(CARD_FIELDS, row))))
            except ValueError as exc:
                raise ValueError(f"line {line_no}: {exc}") from exc

    _check_unique(cards)
    cards.sort(key=lambda c: c["card_id"])
    return cards
