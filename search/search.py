"""Deterministic filtering over normalized catalog records."""

from collections import defaultdict
from heapq import merge
from itertools import groupby

from parser.contracts import empty_filters, validate_filters


def _within(value: int | None, lower: int | None, upper: int | None) -> bool:
    if lower is None and upper is None:
        return True
    return value is not None and (lower is None or lower <= value) and (
        upper is None or value <= upper
    )


class CardSearchIndex:
    """Build once for a catalog whose card records stay fixed."""

    def __init__(self, cards: list[dict]) -> None:
        if not isinstance(cards, list):
            raise ValueError("cards must be a list of catalog records")
        self._cards = sorted(cards, key=lambda card: card["card_id"])
        self._by_name: dict[str, list[int]] = defaultdict(list)
        self._by_type: dict[str, list[int]] = defaultdict(list)
        self._by_rarity: dict[str, list[int]] = defaultdict(list)
        for position, card in enumerate(self._cards):
            self._by_name[card["name"].casefold()].append(position)
            if card["rarity"]:
                self._by_rarity[card["rarity"].casefold()].append(position)
            for value in {value.casefold() for value in card["types"]}:
                self._by_type[value].append(position)

    def search(self, filters: dict, limit: int = 20, offset: int = 0) -> dict:
        """Count all matches and keep only the requested page in memory."""
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer from 1 to 100")
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a nonnegative integer")

        filters = validate_filters(filters)
        page_end = offset + limit
        if filters == empty_filters():
            return {
                "cards": self._cards[offset:page_end],
                "total": len(self._cards),
                "limit": limit,
                "offset": offset,
            }

        name = (filters["name"] or "").casefold()
        rarity = (filters["rarity"] or "").casefold()
        types = {value.casefold() for value in filters["types"]}

        # Each list contains positions in card ID order. Start with the
        # smallest group of lists and check the remaining filters as we go.
        groups = []
        if name:
            groups.append([positions for key, positions in self._by_name.items() if name in key])
        if rarity:
            groups.append([self._by_rarity.get(rarity, [])])
        if types:
            groups.append([self._by_type.get(value, []) for value in types])
        if groups:
            smallest = min(groups, key=lambda lists: sum(map(len, lists)))
            positions = (position for position, _ in groupby(merge(*smallest)))
        else:
            positions = iter(range(len(self._cards)))

        total = 0
        page = []
        for position in positions:
            card = self._cards[position]
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
            if offset <= total < page_end:
                page.append(card)
            total += 1

        return {"cards": page, "total": total, "limit": limit, "offset": offset}


def search_cards(
    cards: list[dict], filters: dict, limit: int = 20, offset: int = 0
) -> dict:
    """Search a catalog once. Reuse CardSearchIndex for multiple searches."""
    return CardSearchIndex(cards).search(filters, limit, offset)
