# Part 1 — Card catalog

Converts raw Pokémon TCG card metadata into the shared **Card** format and
hands it off as `data/cards.csv`. No other part needs to know the source
dataset's field names or formats.

## Public API (`catalog/cards.py`)

```python
from catalog.cards import normalize_card, load_cards, write_cards_csv

card = normalize_card(raw)                  # one record -> valid Card, or ValueError
write_cards_csv(cards, "data/cards.csv")    # validate, reject duplicate IDs, sort, write
cards = load_cards("data/cards.csv")        # read, validate, reject duplicates, sort by card_id
```

A Card has exactly eight keys:

| key | type |
|---|---|
| `card_id` | non-empty str, unique |
| `name` | non-empty str |
| `types` | list of str (never null; `[]` if unknown) |
| `hp` | int ≥ 0 or `None` |
| `rarity` | str or `None` |
| `set_name` | non-empty str |
| `release_year` | int 1..9999 or `None` |
| `image_url` | http(s) URL or `None` |

**Cleaning rules**
- `types` accepts a Python list or a JSON list string.
- Blank or absent nullable fields become `None`. Values are never invented.
- Numbers must be whole integers (int or digit string). Booleans, decimals,
  negatives and out-of-range years raise `ValueError`.
- Blank `card_id` / `name` / `set_name`, bad URLs, and unknown keys raise
  `ValueError`. Source-specific field names belong in the source adapter only.

**CSV handoff**
- Header, in this exact order: `card_id,name,types,hp,rarity,set_name,release_year,image_url`
- `types` is a JSON list in a quoted cell, e.g. `"[""Fire""]"`.
- Null cells are blank. The file never contains the text `None` or `null`.
- Rows are sorted by `card_id` (string order). Duplicate IDs are rejected on
  write and on load. UTF-8 encoding.

## Data source

- **Source:** the [Pokémon TCG API](https://pokemontcg.io) (`https://api.pokemontcg.io/v2/cards`),
  a free, community-run REST API. Its raw data lives in
  [PokemonTCG/pokemon-tcg-data](https://github.com/PokemonTCG/pokemon-tcg-data).
- **Fetched:** one request per page (250 cards) per set, `q=set.id:<set>`.
  An API key is optional; set `POKEMONTCG_API_KEY` in your environment for
  higher rate limits. Never commit the key.
- **Reproducibility:** the API always serves current data, so `data/cards.csv`
  is committed and is the version every other part uses. Rebuilding is only
  needed to change the subset. Use `--save-raw` to keep a dated snapshot of the
  raw API records.
- **Usage rights:** the data repository currently ships no LICENSE file, and its
  README states no license. Third-party API listings describe it as MIT, but we
  could not confirm that in the repo. We use this data only for a non-commercial
  course project (McGill INSY 684). Pokémon names and card content are
  trademarks of Nintendo / The Pokémon Company / Creatures / GAME FREAK; this
  project is not affiliated with or endorsed by them. Card images are linked
  by URL only and are not stored in this repo.

### Subset kept (517 cards)

We keep only **Pokémon** cards; Trainer and Energy cards have no name, type or
HP to search. We use five sets that span 1999–2023 and focus on Generation 1,
which keeps the name vocabulary manageable for the query parser.

| set id | set name | year | cards |
|---|---|---|---|
| base1 | Base | 1999 | 69 |
| neo1 | Neo Genesis | 2000 | 81 |
| ex6 | FireRed & LeafGreen | 2004 | 99 |
| xy12 | Evolutions | 2016 | 83 |
| sv3pt5 | 151 | 2023 | 185 |

HP ranges from 30 to 340, and there are 13 rarity labels.

### Source → Card mapping (`catalog/pokemontcg_source.py`)

| Card field | source field | cleaning |
|---|---|---|
| card_id | `id` | — |
| name | `name` | trimmed |
| types | `types` | absent → `[]` |
| hp | `hp` | string `"120"` → int `120`; absent → `None` |
| rarity | `rarity` | absent → `None` |
| set_name | `set.name` | — |
| release_year | `set.releaseDate` | `"1999/01/09"` → `1999` |
| image_url | `images.small` | absent → `None` |

## Commands

Python 3.11+. Run from the repo root.

```bash
pip install -r catalog/requirements.txt

# Tests (offline, no network or services needed)
python -m pytest tests/test_catalog.py -q

# Smoke test
python -c "from catalog.cards import load_cards; print(len(load_cards('data/cards.csv')), 'cards')"

# Rebuild data/cards.csv from the API (needs internet; only to change the subset)
python -m catalog.pokemontcg_source --out data/cards.csv
# optional: --sets base1 neo1 ex6 xy12 sv3pt5   --save-raw data/raw/api_snapshot.json
```

`data/cards.csv` is committed, so other parts never call the API.

## Tests

`tests/test_catalog.py` covers:
- the guide's example record normalizing to `DEMO_CARDS[0]`
- the `DEMO_CARDS` CSV round trip with no value changes
- a missing HP value, invalid numbers, and invalid required text
- duplicate IDs on write and on load
- the header, blank-null and quoted-types CSV format
- source mapping (string HP, date → year, missing values staying `None`)
- API pagination and Pokémon-only filtering, using a fake API (no network)
- a sanity check on the committed `data/cards.csv`

The tests import `DEMO_CARDS` from `shared.fixtures` when the integrator's
scaffold exists, and use an identical local copy otherwise.
