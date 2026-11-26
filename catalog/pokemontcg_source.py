"""Source adapter: Pokémon TCG API (pokemontcg.io) -> canonical Cards.

All field names specific to this source stay in this module. Everything it
produces goes through normalize_card, so the rest of the project only ever
sees the shared Card format.

Usage (from the repo root; needs internet):

    python -m catalog.pokemontcg_source --out data/cards.csv

An API key is optional. Without one the API allows fewer requests; with one,
set it in the environment (never commit it):

    PowerShell:  $env:POKEMONTCG_API_KEY = "your-key"
    macOS/Linux: export POKEMONTCG_API_KEY=your-key
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from catalog.cards import normalize_card, write_cards_csv

API_URL = "https://api.pokemontcg.io/v2/cards"
PAGE_SIZE = 250  # API maximum

# A Gen-1-heavy subset spanning five release years, so the year, rarity and
# HP filters have real variety while Pokémon names stay a manageable
# vocabulary for the query parser.
DEFAULT_SET_IDS = (
    "base1",   # Base, 1999
    "neo1",    # Neo Genesis, 2000
    "ex6",     # FireRed & LeafGreen, 2004
    "xy12",    # Evolutions, 2016
    "sv3pt5",  # 151, 2023
)

# Only Pokémon cards: Trainer and Energy cards have no name/type/HP to search.
KEEP_SUPERTYPE = "Pokémon"


# --------------------------------------------------------------------------
# Mapping (no network)
# --------------------------------------------------------------------------

def _release_year(release_date) -> int | None:
    """Source dates look like '1999/01/09'. Missing or malformed -> None."""
    if not isinstance(release_date, str) or len(release_date) < 4:
        return None
    year = release_date[:4]
    return int(year) if year.isdigit() else None


def map_source_card(raw: dict) -> dict:
    """Map one API card object (with its embedded `set`) to a validated Card."""
    source_set = raw.get("set") or {}
    images = raw.get("images") or {}
    return normalize_card({
        "card_id": raw.get("id"),
        "name": raw.get("name"),
        "types": raw.get("types"),          # absent -> []
        "hp": raw.get("hp"),                # API returns HP as a string
        "rarity": raw.get("rarity"),        # absent -> None
        "set_name": source_set.get("name"),
        "release_year": _release_year(source_set.get("releaseDate")),
        "image_url": images.get("small"),   # absent -> None
    })


# --------------------------------------------------------------------------
# API access
# --------------------------------------------------------------------------

def http_get_json(url: str, api_key: str | None = None, retries: int = 4,
                  timeout: float = 30.0) -> dict:
    """GET a JSON document, retrying on rate limits and server errors."""
    headers = {"Accept": "application/json", "User-Agent": "CardQuery-catalog"}
    if api_key:
        headers["X-Api-Key"] = api_key
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == retries:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries:
                raise
        time.sleep(2 ** attempt)  # 1s, 2s, 4s, 8s
    raise RuntimeError("unreachable")


def fetch_set_cards(set_id: str, fetch_json=http_get_json, api_key: str | None = None) -> list[dict]:
    """Fetch every card in one set, following pagination."""
    cards, page = [], 1
    while True:
        query = urllib.parse.urlencode({
            "q": f"set.id:{set_id}",
            "page": page,
            "pageSize": PAGE_SIZE,
            "orderBy": "number",
        })
        payload = fetch_json(f"{API_URL}?{query}", api_key=api_key)
        batch = payload.get("data", [])
        cards.extend(batch)
        total = payload.get("totalCount", len(cards))
        if not batch or len(cards) >= total:
            break
        page += 1
    if not cards:
        raise ValueError(f"API returned no cards for set {set_id!r}")
    return cards


def build_cards(set_ids=DEFAULT_SET_IDS, fetch_json=http_get_json,
                api_key: str | None = None) -> tuple[list[dict], list[dict]]:
    """Return (cards, raw_api_records) for the chosen sets, Pokémon only."""
    cards, raw_records = [], []
    for set_id in set_ids:
        for raw in fetch_set_cards(set_id, fetch_json=fetch_json, api_key=api_key):
            if raw.get("supertype") == KEEP_SUPERTYPE:
                raw_records.append(raw)
                cards.append(map_source_card(raw))
    return cards, raw_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build data/cards.csv from the Pokémon TCG API.")
    parser.add_argument("--out", default="data/cards.csv")
    parser.add_argument("--sets", nargs="+", default=list(DEFAULT_SET_IDS))
    parser.add_argument("--save-raw", metavar="PATH",
                        help="also save the raw API records (JSON) as a snapshot")
    args = parser.parse_args()

    api_key = os.environ.get("POKEMONTCG_API_KEY") or None
    cards, raw_records = build_cards(args.sets, api_key=api_key)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_cards_csv(cards, args.out)
    print(f"wrote {len(cards)} cards to {args.out}")

    if args.save_raw:
        snapshot = {
            "source": API_URL,
            "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "set_ids": args.sets,
            "data": raw_records,
        }
        Path(args.save_raw).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_raw).write_text(json.dumps(snapshot, ensure_ascii=False, indent=1), "utf-8")
        print(f"saved raw snapshot to {args.save_raw}")


if __name__ == "__main__":
    main()
