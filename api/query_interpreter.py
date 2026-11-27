"""Normalize supported wording and reject ungrounded model name predictions."""

import re

from parser.numeric_v2 import parse_numeric_v2 as parse_numeric
from parser.parser import preprocess_query
from parser.semantic import resolver
from shared.contracts import InvalidQueryError, normalize_query

_LESS_THAN_HP = re.compile(r"(?<!no )(?<!not )\bless than(?= [+-]?\d+(?:\.\d+)? hp\b)")
_BROWSE_WORDS = {
    "all", "any", "available", "browse", "can", "card", "cards", "could", "find",
    "give", "i", "like", "list", "look", "me", "please", "pokemon", "pokémon",
    "see", "show", "some", "the", "to", "want", "would", "you",
}


def normalize_search_query(query: str) -> str:
    return _LESS_THAN_HP.sub("under", normalize_query(query))


def _pattern(label: str, plural: bool = False):
    text = re.escape(preprocess_query(label))
    return re.compile(r"(?<!\w)" + text + (r"s?" if plural else "") + r"(?!\w)")


class QueryInterpreter:
    def __init__(self, cards):
        self.names = {name: _pattern(name, plural=True)
                      for name in {card["name"] for card in cards}}
        labels = {value for card in cards for value in card["types"]}
        labels.update(card["rarity"] for card in cards if card["rarity"])
        self.categories = [_pattern(label) for label in labels]

    def validate_prediction(self, query: str, filters: dict) -> dict:
        """Require query evidence for a name; filter-only requests can omit it."""
        text = preprocess_query(query)
        mentioned_names = {name for name, pattern in self.names.items() if pattern.search(text)}
        resolved_name, _ = resolver(tuple(sorted(self.names))).resolve(query)
        if resolved_name:
            mentioned_names.add(resolved_name)
        numeric = parse_numeric(query)
        has_filter = any(pattern.search(text) for pattern in self.categories) or any(
            value is not None for value in numeric.values()
        )
        words = set(re.findall(r"\w+", query))
        semantic_context = bool(words & {
            "pokemon", "pokémon", "card", "cards", "type", "types", "element", "powers",
            "abilities", "attacks", "creatures",
        })
        unambiguous = len(filters["types"]) <= 1 or bool(words & {"and", "or"})
        inferred_filter = semantic_context and unambiguous and (
            bool(filters["types"]) or filters["rarity"] is not None
        )
        has_filter = has_filter or inferred_filter
        browse = bool(words & {"card", "cards", "pokemon", "pokémon"}) and words <= _BROWSE_WORDS
        if not mentioned_names and not has_filter and not browse:
            raise InvalidQueryError(
                "No supported Pokémon name or filter was recognized. "
                "Check the spelling, or try 'pikachu', 'fire cards', or 'charizard under 150 hp'"
            )

        result = dict(filters)
        if not mentioned_names:
            # A type/year/HP-only search must not acquire an invented Pokémon name.
            result["name"] = None
        elif result["name"] not in mentioned_names:
            raise InvalidQueryError(
                "The model could not reliably identify the Pokémon name. "
                "Try the exact name with a simple filter, such as 'charizard under 150 hp'"
            )
        return result
