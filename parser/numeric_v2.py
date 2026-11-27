"""Natural HP/year comparisons with exact inclusive integer bounds."""

import re

from .contracts import InvalidQueryError, empty_filters, normalize_query, validate_filters
from .numeric_rules import _integer

VERSION = "v2"
NUMBER = r"[+-]?[0-9]+(?:\.[0-9]+)?"
UNIT = r"(?:hps?|health\s+points?|hit\s+points?)\b"
OPERATORS = {
    "no more than": "max", "not more than": "max", "no greater than": "max",
    "less than or equal to": "max", "at most": "max", "up to": "max",
    "maximum": "max", "max": "max", "<=": "max", "≤": "max",
    "no less than": "min", "not less than": "min", "greater than or equal to": "min",
    "at least": "min", "minimum": "min", "min": "min", ">=": "min", "≥": "min",
    "more than": "over", "greater than": "over", "over": "over", "above": "over", ">": "over",
    "less than": "under", "fewer than": "under", "under": "under", "below": "under", "<": "under",
    "exactly": "equal", "equal to": "equal", "equals": "equal", "=": "equal",
}
OPS = "|".join(re.escape(word) for word in sorted(OPERATORS, key=len, reverse=True))
PREFIX = re.compile(rf"(?<!\w)({OPS})\s*({NUMBER})\s*{UNIT}")
SUFFIX = re.compile(rf"\b{UNIT}\s*({OPS})\s*({NUMBER})(?![\w.])")
RANGES = [
    re.compile(rf"\bbetween\s+({NUMBER})\s+and\s+({NUMBER})\s*{UNIT}"),
    re.compile(rf"\b{UNIT}\s+between\s+({NUMBER})\s+and\s+({NUMBER})(?![\w.])"),
    re.compile(rf"(?<!\w)({NUMBER})\s*[-–]\s*({NUMBER})\s*{UNIT}"),
]
PLUS = re.compile(rf"(?<!\w)({NUMBER})\s*\+\s*{UNIT}")
EXACT = re.compile(rf"(?<![\w.])({NUMBER})\s*{UNIT}(?:\s+exactly\b)?")
YEAR_RANGE = re.compile(rf"\bbetween\s+({NUMBER})\s+and\s+({NUMBER})(?![\w.])")
YEAR = re.compile(rf"\b(after|before|since|newer than|older than|from|in)\s+(?:the\s+year\s+)?({NUMBER})(?![\w.])")


def parse_numeric_v2(query):
    text = normalize_query(query)
    result = empty_filters()
    spans = []

    def constrain(prefix, low=None, high=None):
        if low is not None and high is not None and low > high:
            raise InvalidQueryError(f"contradictory {prefix} range")
        for suffix, value, combine in (("min", low, max), ("max", high, min)):
            if value is not None:
                key = f"{prefix}_{suffix}"
                result[key] = value if result[key] is None else combine(result[key], value)

    def overlap(match):
        return any(match.start() < end and match.end() > start for start, end in spans)

    for index, pattern in enumerate(RANGES):
        for match in pattern.finditer(text):
            # In "at most 60 hp between 2016 and 2018", hp belongs
            # to 60; the following range is a year range, not an HP range.
            if index == 1 and re.search(r"[0-9]\s*$", text[:match.start()]):
                continue
            if not overlap(match):
                constrain("hp", _integer(match[1]), _integer(match[2]))
                spans.append(match.span())
    for pattern in (PREFIX, SUFFIX):
        for match in pattern.finditer(text):
            if overlap(match):
                continue
            operator, value = OPERATORS[match[1]], _integer(match[2])
            if operator in ("min", "over", "equal"):
                constrain("hp", low=value + (operator == "over"))
            if operator in ("max", "under", "equal"):
                constrain("hp", high=value - (operator == "under"))
            spans.append(match.span())
    for match in PLUS.finditer(text):
        if not overlap(match):
            constrain("hp", low=_integer(match[1]))
            spans.append(match.span())
    for match in EXACT.finditer(text):
        if not overlap(match):
            value = _integer(match[1])
            constrain("hp", value, value)
            spans.append(match.span())
    for match in YEAR_RANGE.finditer(text):
        if not overlap(match):
            constrain("year", _integer(match[1], year=True), _integer(match[2], year=True))
    for match in YEAR.finditer(text):
        if overlap(match):
            continue
        operator, value = match[1], _integer(match[2], year=True)
        if operator in ("after", "newer than", "since"):
            constrain("year", low=value + (operator != "since"))
        elif operator in ("before", "older than"):
            constrain("year", high=value - 1)
        else:
            constrain("year", value, value)
    try:
        result = validate_filters(result)
    except ValueError as error:
        raise InvalidQueryError(str(error)) from error
    return {key: result[key] for key in ("hp_min", "hp_max", "year_min", "year_max")}
