"""CardQuery search page (Part 4).

Run locally:   python -m frontend.app        (serves 0.0.0.0:5000)
Environment:   API_BASE_URL=http://localhost:8000  (Docker: http://api:8000)
"""

from __future__ import annotations

import os

from flask import Flask, render_template, request

from frontend import client
from frontend.client import PAGE_SIZE, SearchError


def describe_filters(filters: dict) -> str:
    """Readable summary, e.g. ``name Charizard, type Fire, HP >= 150``."""
    parts: list[str] = []
    if filters.get("name"):
        parts.append(f"name {filters['name']}")
    types = [t for t in (filters.get("types") or []) if t]
    if types:
        parts.append("type " + " or ".join(types))
    parts.extend(_range("HP", filters.get("hp_min"), filters.get("hp_max")))
    if filters.get("rarity"):
        parts.append(f"rarity {filters['rarity']}")
    parts.extend(_range("year", filters.get("year_min"), filters.get("year_max")))
    return ", ".join(parts) if parts else "none (matching every card)"


def _range(label: str, low, high) -> list[str]:
    if low is not None and high is not None and low == high:
        return [f"{label} = {low}"]
    out = []
    if low is not None:
        out.append(f"{label} >= {low}")
    if high is not None:
        out.append(f"{label} <= {high}")
    return out


def card_view(card: dict) -> dict:
    """Display-ready values for one Card; missing values get friendly text."""
    image_url = card.get("image_url")
    if not (isinstance(image_url, str) and image_url.startswith(("http://", "https://"))):
        image_url = None
    hp = card.get("hp")
    year = card.get("release_year")
    types = card.get("types") or []
    return {
        "card_id": card.get("card_id", ""),
        "name": card.get("name", ""),
        "types": ", ".join(types) if types else "No type",
        "hp": f"{hp} HP" if hp is not None else "HP unknown",
        "rarity": card.get("rarity") or "Rarity unknown",
        "set_name": card.get("set_name", ""),
        "year": str(year) if year is not None else "Year unknown",
        "image_url": image_url,
    }


def _parse_offset(raw: str | None) -> int:
    try:
        return max(int(raw), 0) if raw is not None else 0
    except ValueError:
        return 0


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        query = request.args.get("q", "")
        offset = _parse_offset(request.args.get("offset"))
        context = {"query": query, "results": None, "error": None}

        if "q" in request.args:
            try:
                data = client.search(query, limit=PAGE_SIZE, offset=offset)
            except SearchError as exc:
                context["error"] = exc.message
            else:
                cards = [card_view(c) for c in data["cards"]]
                total = data["total"]
                context["results"] = {
                    "cards": cards,
                    "total": total,
                    "count_label": f"{total} result" + ("" if total == 1 else "s"),
                    "filters_text": describe_filters(data["filters"]),
                    "model_version": data.get("model_version"),
                    "first": offset + 1,
                    "last": offset + len(cards),
                    "prev_offset": max(offset - PAGE_SIZE, 0) if offset > 0 else None,
                    "next_offset": offset + PAGE_SIZE if offset + len(cards) < total else None,
                }
        return render_template("index.html", **context)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
