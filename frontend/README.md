# Part 4: Frontend and traffic simulator

Part 4 has two pieces:

- **Search page** (`frontend/`): a Flask page that sends the user's query to the API's `POST /search` and shows the matching cards and the interpreted filters. It talks to the backend only over HTTP.
- **Traffic simulator** (`monitoring/simulator.py`): sends unseen labelled queries in `normal` or `drift` mode. It records one TrafficLog per request, attaches the known labels, and stores the logs through Part 6's `save_traffic_logs`.

Neither piece needs the backend, parser or database source code. `frontend/fake_api.py` is a small stand-in API that uses only the standard library, for demos and tests.

## Setup

Python 3.11. Run these from the repo root.

```bash
python -m pip install -r frontend/requirements-dev.txt -r frontend/requirements-simulator.txt
python -m pytest -q tests/test_frontend.py tests/test_simulator.py
```

## Search page

```bash
# Terminal 1: fake API on :8000 (or start the real Part 3 API instead)
python -m frontend.fake_api

# Terminal 2: the page on http://localhost:5000
API_BASE_URL=http://localhost:8000 python -m frontend.app
```

| Setting | Default | Notes |
|---|---|---|
| `API_BASE_URL` | `http://localhost:8000` | Use `http://api:8000` in Docker Compose |
| `PORT` | `5000` | The page always binds to `0.0.0.0` |

**What it does:**

- The query box submits to `GET /?q=...`.
- The server sends `{"query", "limit": 20, "offset": 0}` to `API_BASE_URL/search` with a 10-second timeout.
- The page shows the result count, the interpreted filters, the model version and a card gallery. Cards without an image get a placeholder.
- *Previous* / *Next* links appear when there are more than 20 results.
- A loading indicator shows while a search is running.
- All query and card text is HTML-escaped by Jinja autoescaping. Only `http(s)` image URLs are rendered.

The golden query `fire charizards with at least 150 hp` shows:

```
1 result
Interpreted filters: name Charizard, type Fire, HP >= 150
Charizard | Fire | 180 HP | Rare
Demo A | 2020 | ID: demo-001
```

**Error states.** The query stays in the box in every case.

| Situation | Message shown |
|---|---|
| Blank query or over 500 characters | Checked before calling the API |
| 0 results | "No cards matched your search…" (the filters are still shown) |
| HTTP 422 | "We couldn't understand that search: …" with FastAPI's `detail` messages |
| HTTP 503 | "The search model is temporarily unavailable…" |
| Other HTTP error or unreadable response | "The search service ran into a problem (HTTP n)…" |
| Timeout (10 s) / connection error | "…took too long…" / "We couldn't reach the search service…" |

**Fake API triggers.** These let you see each error state by hand: `trigger 503`, `trigger 500` and `trigger slow`. `trigger slow` waits 12 seconds, so the page's 10-second timeout fires; set `FAKE_API_SLOW_SECONDS` to change the wait. Stop the fake API to see the connection error. Any other query is parsed by keywords and searched against the three demo cards, for example `pikachu cards` or `rare fire cards before 2019`.

`GET /health` on the page returns `{"status": "ok"}`, which the container healthcheck uses.

## Traffic simulator

Type each command on one line:

```bash
python -m monitoring.simulator --mode normal --count 100 --api-url http://localhost:8000
python -m monitoring.simulator --mode drift --count 100 --api-url http://localhost:8000
```

| Option | Default | |
|---|---|---|
| `--mode` | required | `normal` or `drift` |
| `--count` | `100` | Exact number of requests |
| `--api-url` | `$API_BASE_URL` or `http://localhost:8000` | |
| `--cases` | `data/unseen_<mode>.csv` | `query,filters_json` pool (Part 2 supplies the real pools) |
| `--database-url` | `$DATABASE_URL` | Required unless `--dry-run` |
| `--seed` | `42` | |
| `--jsonl PATH` | off | Optional extra export; PostgreSQL stays the main store |
| `--dry-run` | off | Keep logs in memory instead of PostgreSQL |

**How it runs:**

1. Loads the pool and shuffles it once with `random.Random(42)`.
2. Cycles through the shuffled pool to send exactly `count` requests.
3. Copies each selected case's label into `expected_filters`.
4. Calls `monitoring.persistence.save_traffic_logs(logs, DATABASE_URL)`, which is Part 6's adapter.

If that call fails, the command exits with code 1. It never falls back to file-only storage silently.

**Trying it before Parts 2 and 6 are merged.** The `tests/fixtures/unseen_*.csv` files are the committed placeholder pools:

```bash
python -m frontend.fake_api &
python -m monitoring.simulator --mode drift --count 100 --cases tests/fixtures/unseen_drift.csv --dry-run
```

**Python API:**

```python
from monitoring.simulator import run_traffic
logs = run_traffic("http://localhost:8000", ["fire charizards with at least 150 hp"], mode="normal")
```

`run_traffic` returns one TrafficLog per request, with these keys: `request_id` (a UUID4), `timestamp` (UTC ending in `Z`), `query`, `mode`, `status_code`, `latency_ms`, `model_version`, `predicted_filters` and `expected_filters`.

- `expected_filters` is always `None`; the CLI attaches the labels.
- Any non-200 response, or a 200 whose `filters` isn't a seven-key object, gets `predicted_filters=None`.
- A timeout or connection error gets `status_code=0`.
- Latency is measured for every request, including failures.
- An invalid `mode`, a blank `base_url` or a non-list `queries` raises `ValueError`.

**Placeholder drift pool.** It uses the same intents and labels as the normal pool, reworded so the text looks different: slang, typos, all caps and much longer sentences. The tests check that at least 30% of the drift pool's tokens never appear in the normal pool, and that drift queries are on average more than 1.5× as long. Part 2's real pools replace these, and they must not overlap with any training data. A test checks this against `data/query_examples.csv` once that file exists.

## Docker

The build context is the repo root.

```bash
docker build -f frontend/Dockerfile -t cardquery-frontend .
docker run --rm -p 5000:5000 -e API_BASE_URL=http://host.docker.internal:8000 cardquery-frontend

docker build -f frontend/Dockerfile.simulator -t cardquery-simulator .
docker run --rm -e API_BASE_URL=http://api:8000 -e DATABASE_URL=... cardquery-simulator --mode drift --count 100
```

`frontend/compose.part4.yml` is the Compose fragment for the integrator. Its build contexts are relative to `frontend/`, so change `context: ..` to `context: .` when moving them into the root compose file.

- The page runs under gunicorn on `0.0.0.0:5000`.
- The simulator is under the `traffic` profile, so `docker compose up` doesn't start it. Run it on demand with `docker compose run --rm simulator --mode drift --count 100`.
- The simulator image installs any `monitoring/requirements*.txt` from Part 6, so `psycopg` and similar packages come in automatically.

## Files

| File | Purpose |
|---|---|
| `frontend/app.py` | Flask app, filter summary, card display values |
| `frontend/client.py` | `POST /search` client; turns every failure into a user-facing `SearchError` |
| `frontend/templates/index.html`, `frontend/static/*` | Page, styles, loading indicator |
| `frontend/fake_api.py` | Stand-in API for demos and tests |
| `frontend/fixtures.py` | Page-3 shared examples. Imports `shared.fixtures` when the integrator adds it |
| `monitoring/simulator.py` | `run_traffic` and the CLI |
| `tests/test_frontend.py`, `tests/test_simulator.py` | Offline tests: 60 total, including invalid-input and real-HTTP cases |
| `tests/fixtures/unseen_{normal,drift}.csv` | Placeholder labelled pools |

## Notes for the integrator

- `monitoring/__init__.py` is an empty package marker. The `monitoring/` directory is shared by Parts 4, 6 and 7, and none of the guide's file lists includes this file.
- The simulator imports `save_traffic_logs` from `monitoring.persistence`. If Part 6 names that module differently, only `_default_saver()` in `simulator.py` needs to change.
