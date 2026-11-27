"""Deterministic filtering over normalized catalog records."""

from parser.contracts import validate_filters


def _within(value: int | None, lower: int | None, upper: int | None) -> bool:
    if lower is None and upper is None:
        return True
    return value is not None and (lower is None or lower <= value) and (
        upper is None or value <= upper
    )


def search_cards(
    cards: list[dict], filters: dict, limit: int = 20, offset: int = 0
) -> dict:
    """Count all matches, then return the requested page sorted by card ID."""
    if not isinstance(cards, list):
        raise ValueError("cards must be a list of catalog records")
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("limit must be an integer from 1 to 100")
    if type(offset) is not int or offset < 0:
        raise ValueError("offset must be a nonnegative integer")

    filters = validate_filters(filters)
    name = (filters["name"] or "").casefold()
    rarity = (filters["rarity"] or "").casefold()
    types = {value.casefold() for value in filters["types"]}

    matches = []
    for card in cards:
        if name and name not in card["name"].casefold():
            continue
        if rarity and rarity != (card["rarity"] or "").casefold():
            continue
        if types and not types.intersection(value.casefold() for value in card["types"]):
            continue
        if not _within(card["hp"], filters["hp_min"], filters["hp_max"]):
            continue
        if not _within(card["release_year"], filters["year_min"], filters["year_max"]):
            continue
        matches.append(card)

    matches.sort(key=lambda card: card["card_id"])
    return {
        "cards": matches[offset:offset + limit],
        "total": len(matches),
        "limit": limit,
        "offset": offset,
    }
