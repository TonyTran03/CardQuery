# Part 6 - Storage + evaluation (PostgreSQL, Evidently)

Owns: `monitoring/{drift,performance,persistence}.py`, `monitoring/sql/`, the reference/current
fixtures, `tests/test_monitoring.py`, and the PostgreSQL + drift-worker Compose fragment.

## Public functions

| Function | Module | Used by |
|---|---|---|
| `save_traffic_logs(rows, database_url)` | `persistence` | Part 4 simulator |
| `compute_drift(reference_rows, current_rows)` | `drift` | worker, tests |
| `compute_performance(rows)` | `performance` | worker, tests |
| `python -m monitoring.drift --reference ... --watch-seconds 60` | `drift` | Compose worker |

`save_traffic_logs` validates every row first, applies `schema.sql`, then inserts with
`ON CONFLICT (request_id) DO NOTHING`, so retries never duplicate rows. If PostgreSQL is
unreachable it raises `RuntimeError("PostgreSQL unavailable: ...")`; there is no file fallback.

## How it works

Every 60 s the worker reads the latest 100 rows from `prediction_logs` (ordered by
`timestamp, request_id`), waits until it has 100, then:

1. **Drift**: features are `query_length` and `token_count` only. Two-sample KS test at p < 0.05
   per feature; `drift_share = drifted / 2`; `drift_detected` when share >= 0.5. Needs >= 100
   reference and >= 100 current rows (else `ValueError`).
2. **Performance**: only HTTP 200 rows with valid expected + predicted Filters count
   (`n_labeled_successes`). `exact_match` = full-filter equality; `micro_f1` = slot-level F1.
   `label_coverage = n_labeled_successes / n_requests`. Both scores are `null` with no labeled rows.
3. Saves one `drift_runs` row. `run_id` hashes reference + current row contents, so an unchanged
   window is skipped.

Drift without labels cannot establish accuracy or concept drift; `exact_match` is what tells you
the model got worse.

## Handoff to Part 7 (Grafana)

Table `drift_runs`. Useful queries:

```sql
-- latest drift share (panel + alert)
SELECT window_end AS time, drift_share FROM drift_runs ORDER BY window_end DESC LIMIT 1;

-- exact match and coverage over time
SELECT window_end AS time,
       (model_metrics->>'exact_match')::float    AS exact_match,
       (model_metrics->>'label_coverage')::float AS label_coverage
FROM drift_runs ORDER BY window_end;
```

JSONB paths: `model_metrics->>'exact_match'`, `model_metrics->>'micro_f1'`,
`model_metrics->>'label_coverage'`. `exact_match` may be `NULL`; filter or coalesce in panels.
`drift_share` is a plain column. `drifted_columns` is a JSONB array.
`evidently_report` holds `{"drift": {...}, "performance": {...}}`; each side has an `engine` key.

Seed rows for building dashboards before live data exists: `monitoring/sql/seed.sql`
(12 normal runs, then 12 drift runs, relative to `now()`).

```bash
psql "$DATABASE_URL" -f monitoring/sql/schema.sql -f monitoring/sql/seed.sql
```

## Run

```bash
pip install -r monitoring/requirements.txt
python -m pytest tests/test_monitoring.py                 # offline
TEST_DATABASE_URL=postgresql://... python -m pytest tests/test_monitoring.py   # + live test
DATABASE_URL=postgresql://... python -m monitoring.drift \
    --reference data/reference_traffic.jsonl --once         # single window, then exit
docker compose -f docker-compose.yml -f docker-compose.monitoring-storage.yml up postgres drift-worker
```

## Engines

Evidently is required by default and explicitly selected in Compose. The `scipy` drift
engine and `fallback` accuracy engine are for isolated offline tests only. The real
Evidently test overrides those settings and runs when Evidently is installed.
Canonical label spelling is preserved, as required by the shared evaluation rules;
coordinate with the parser owner when the shared validator is published.

For isolated Docker testing, from the directory containing the Compose fragment:

```bash
export POSTGRES_PASSWORD=your_local_test_password
docker compose -f docker-compose.monitoring-storage.yml up -d --build
docker compose -f docker-compose.monitoring-storage.yml logs drift-worker
```

Use a fresh disposable database for the live test. Do not commit passwords.
The pinned dependencies and live services must be verified on Python 3.11.

## Fixtures

- `data/reference_traffic.jsonl`: 100 normal logs (frozen reference).
- `tests/fixtures/monitoring/normal_current.jsonl`: fresh normal window, no drift expected.
- `tests/fixtures/monitoring/shifted_current.jsonl`: long, wordy queries; drift expected.

These are synthetic stand-ins. Replace the reference with real normal-mode simulator output once
Part 4 produces it (keep 100 rows).
