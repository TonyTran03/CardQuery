# CardQuery API

From the repository root with Python 3.11 or newer:

```bash
python -m pip install -r api/requirements.txt
python -m uvicorn api.main:app --port 8000
```

The API loads `data/cards.csv` and `artifacts/parser/model.joblib` once per
process. Open [Swagger UI](http://127.0.0.1:8000/docs) to try the routes. The
frontend and traffic simulator can use their existing default API URL
(`http://localhost:8000`).

| Method and route | Request body | Success response |
| --- | --- | --- |
| `POST /search` | `{query, limit?, offset?}` | `{filters, cards, total, model_version}` |
| `POST /parse` | `{query}` | `{filters, model_version}` |
| `POST /cards/search` | `{filters, limit?, offset?}` | `{cards, total, limit, offset}` |
| `GET /health` (liveness) | None | `{"status":"ok"}` |
| `GET /ready` (readiness) | None | `{"status":"ok"}` |
| `GET /metrics` | None | Prometheus text |

Try `{"query":"fire charizards with at least 150 hp","limit":1}` on
`POST /search`. The filters contain `name: "Charizard"`, `types: ["Fire"]`,
and `hp_min: 150`. The response includes one card and the total number of
matches before pagination. Defaults are `limit=20` and `offset=0`. Filter
search accepts all seven keys returned by `/parse`.

Invalid request bodies or contradictory constraints return HTTP 422. If the
model fails to load or predict, the model routes return HTTP 503 with
`{"detail":"model_unavailable"}`. The catalog must load before the API starts.
`/ready` returns HTTP 503 if the model fails to load or run its startup test
prediction. Docker uses `/ready` for its health check. The `/health` liveness
route and catalog-only search still work after a model startup failure.

The API sorts and indexes cards once at startup. Name, type, and rarity
filters narrow the cards checked per request. Numeric-only searches still scan
the sorted catalog to count exact matches.

Before prediction, the API normalizes `less than N hp` to `under N hp`, preserving
the strict upper bound `N-1`. After prediction, it checks Pokémon names against
the names actually mentioned in the request. Unknown searches such as `winston`
return a helpful HTTP 422 message. Type/rarity/HP/year-only queries keep the name
filter empty, so wording such as `water pokemon` cannot acquire an unrelated
Pokémon name. These checks live in `api/query_interpreter.py`; the registered
model weights and the team's training implementation are unchanged.

Set `CARDQUERY_MODEL_PATH`, `CARDQUERY_CARDS_PATH`, or
`CARDQUERY_MODEL_VERSION` before starting Uvicorn to override the defaults.
The default version label is `local-v1`. Set `CARDQUERY_API_MODE=stub` to
serve the three demo cards with the fixture parser. In PowerShell, use
`$env:CARDQUERY_API_MODE = "stub"` in the terminal running Uvicorn.

`create_app(cards, parse, model_version, instrument)` accepts a replacement
parser callable and metrics installer. Its fixture default exposes a process
gauge. Local and registry startup install real request counters and duration
histograms from `monitoring.metrics`, excluding `/metrics` itself.

The root Compose stack uses `CARDQUERY_API_MODE=registry`,
`MLFLOW_TRACKING_URI=http://mlflow:5000`, and `CARDQUERY_ACTIVE_MODEL_PATH`.
The adapter reads the bootstrap manifest, loads its exact registered pyfunc
version, predicts one query row, decodes `filters_json`, and validates Filters.
The response version comes from that registry manifest. Model initialization
failure keeps liveness/catalog searches available while readiness returns 503.
See the root README for full-stack startup; the fragment below runs only the
local saved-model API.

```bash
python -m pip install -r api/requirements-dev.txt
python -m pytest -q tests/test_api.py tests/test_search.py
docker compose -f api/compose.api.yml up --build
docker compose -f api/compose.api.yml down
```
