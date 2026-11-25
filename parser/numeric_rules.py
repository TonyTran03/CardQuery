"""Version v1: explicit inclusive integer HP/year constraints, independent of ML."""

from __future__ import annotations

import re

from .contracts import InvalidQueryError, empty_filters, normalize_query, validate_filters

NUMERIC_RULES_VERSION = "v1"
# Match signed/decimal values too, so a recognized phrase rejects invalid values
# instead of accidentally extracting the trailing digits as a valid constraint.
_NUMBER = r"[+-]?[0-9]+(?:\.[0-9]+)?"
_END = r"(?![\w.])"
_HP_RANGE = re.compile(rf"\bbetween\s+({_NUMBER})\s+and\s+({_NUMBER})\s+hp\b")
_HP_SINGLE = re.compile(rf"\b(at least|at most|over|under)\s+({_NUMBER})\s+hp\b")
_YEAR_RANGE = re.compile(rf"\bbetween\s+({_NUMBER})\s+and\s+({_NUMBER}){_END}")
_YEAR_SINGLE = re.compile(rf"\b(after|before|from)\s+({_NUMBER}){_END}")


def _integer(text: str, *, year: bool = False) -> int:
    if not re.fullmatch(r"[0-9]{4}" if year else r"[0-9]+", text):
        raise InvalidQueryError("years require four digits" if year else "HP must be a nonnegative integer")
    value = int(text)
    if year and not 1 <= value <= 9999:
        raise InvalidQueryError("year must be in 0001..9999")
    return value


def parse_numeric(query: str) -> dict:
    """Return the four numeric bounds; repeated constraints are intersected."""
    text = normalize_query(query)
    filters = empty_filters()

    def constrain(prefix, low=None, high=None):
        if low is not None and high is not None and low > high:
            raise InvalidQueryError(f"contradictory {prefix} range")
        for suffix, value, combine in (("min", low, max), ("max", high, min)):
            if value is not None:
                key = f"{prefix}_{suffix}"
                old = filters[key]
                filters[key] = value if old is None else combine(old, value)

    hp_spans = []
    for match in _HP_RANGE.finditer(text):
        hp_spans.append(match.span())
        constrain("hp", _integer(match[1]), _integer(match[2]))
    for match in _HP_SINGLE.finditer(text):
        value = _integer(match[2])
        operator = match[1]
        if operator in ("at least", "over"):
            constrain("hp", low=value + (operator == "over"))
        else:
            constrain("hp", high=value - (operator == "under"))
    for match in _YEAR_RANGE.finditer(text):
        if any(start <= match.start() < end for start, end in hp_spans):
            continue
        constrain("year", _integer(match[1], year=True), _integer(match[2], year=True))
    for match in _YEAR_SINGLE.finditer(text):
        value = _integer(match[2], year=True)
        if match[1] == "after":
            constrain("year", low=value + 1)
        elif match[1] == "before":
            constrain("year", high=value - 1)
        else:
            constrain("year", value, value)
    try:
        filters = validate_filters(filters)
    except ValueError as exc:
        raise InvalidQueryError(str(exc)) from exc
    return {field: filters[field] for field in ("hp_min", "hp_max", "year_min", "year_max")}
