"""Local contract shim until the integrator supplies shared.contracts.

Use the shared exception when available so the API can catch the same type.
The validator stays here rather than creating/overwriting integrator-owned files.
"""

from __future__ import annotations

try:
    from shared.contracts import InvalidQueryError
except ModuleNotFoundError as exc:
    if exc.name not in {"shared", "shared.contracts"}:
        raise

    class InvalidQueryError(ValueError):
        """Invalid query text or recognized contradictory numeric constraints."""


FILTER_FIELDS = ("name", "types", "hp_min", "hp_max", "rarity", "year_min", "year_max")


def normalize_query(query: str) -> str:
    if not isinstance(query, str) or not 1 <= len(query.strip()) <= 500:
        raise InvalidQueryError("query must be a nonempty string of at most 500 characters")
    return " ".join(query.strip().lower().split())


def validate_filters(filters: dict) -> dict:
    if not isinstance(filters, dict) or set(filters) != set(FILTER_FIELDS):
        raise ValueError(f"Filters must contain exactly {list(FILTER_FIELDS)}")
    result = dict(filters)
    for field in ("name", "rarity"):
        value = filters[field]
        if value is not None:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be nonempty text or None")
            result[field] = value.strip()
    if not isinstance(filters["types"], list):
        raise ValueError("types must be a list of nonempty strings")
    unique = {}
    for value in filters["types"]:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("types must contain nonempty strings")
        unique.setdefault(value.strip().casefold(), value.strip())
    result["types"] = [unique[key] for key in sorted(unique)]
    for field in ("hp_min", "hp_max", "year_min", "year_max"):
        value = filters[field]
        if value is None:
            continue
        low, high = (1, 9999) if field.startswith("year") else (0, None)
        if type(value) is not int or value < low or (high is not None and value > high):
            raise ValueError(f"{field} must be an integer in {low}..{high or 'unbounded'} or None")
    for prefix in ("hp", "year"):
        low, high = result[f"{prefix}_min"], result[f"{prefix}_max"]
        if low is not None and high is not None and low > high:
            raise ValueError(f"{prefix}_min must not exceed {prefix}_max")
    return result


def empty_filters() -> dict:
    return {field: [] if field == "types" else None for field in FILTER_FIELDS}
