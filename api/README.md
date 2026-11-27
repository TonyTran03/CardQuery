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

Set `CARDQUERY_MODEL_PATH`, `CARDQUERY_CARDS_PATH`, or
`CARDQUERY_MODEL_VERSION` before starting Uvicorn to override the defaults.
The default version label is `local-v1`. Set `CARDQUERY_API_MODE=stub` to
serve the three demo cards with the fixture parser. In PowerShell, use
`$env:CARDQUERY_API_MODE = "stub"` in the terminal running Uvicorn.

`create_app(cards, parse, model_version, instrument)` accepts a replacement
parser callable and metrics installer. The default installer exposes a process
gauge. Pass the monitoring team's installer when request counters and latency
metrics are ready.

```bash
python -m pip install -r api/requirements-dev.txt
python -m pytest -q tests/test_api.py tests/test_search.py
docker compose -f api/compose.api.yml up --build
docker compose -f api/compose.api.yml down
```
