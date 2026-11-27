# CardQuery

Search Pokemon trading cards in plain language, see interpreted filters, and
monitor predictions against unseen labeled traffic. The catalog, learned parser,
API, frontend, training/registry workflow, simulator, and PostgreSQL/Evidently
worker are integrated. **Grafana dashboards, Prometheus scraping, and alerts
remain pending.** API metrics and stored evaluation results are ready for them.

Part 5 comes from the team's merged [PR #9](https://github.com/TonyTran03/CardQuery/pull/9)
(commit `8e0857c`). The orchestration and bootstrap are retained, with small
dependency/provenance additions for the pretrained semantic parser. The root
Compose stack uses `api.bootstrap` to reuse the active model; it delegates to the
team's `training.bootstrap` for initial or explicit training. MLflow uses the same
training image. See [the semantic parser](parser/SEMANTIC.md) for model details.

## Start the application

Install Docker Desktop with its Linux engine running. From the repository root:

```powershell
docker compose up --build -d
```

The first start builds images, starts PostgreSQL and MLflow, runs a Prefect
training job, registers the selected parser, and publishes its exact version to
a shared manifest. Only then does the API become ready, followed by the frontend
and an initial 100-request normal simulator job. Training may take several
minutes. Later starts validate and reuse the pinned registered model, skipping
training. A new version is trained only for an empty installation or an explicit
`--retrain` command.

| Service | Address |
| --- | --- |
| Search page | [localhost:5000](http://localhost:5000) |
| API docs | [localhost:8000/docs](http://localhost:8000/docs) |
| Readiness | [localhost:8000/ready](http://localhost:8000/ready) |
| API metrics | [localhost:8000/metrics](http://localhost:8000/metrics) |
| MLflow runs and registry | [localhost:5001](http://localhost:5001) |

Try `fire charizards with at least 150 hp`. The frontend calls the API over HTTP;
the API loads the registered pyfunc and searches the real card catalog. The
parser supports trained names/types/rarities and the documented HP/year grammar.
Synthetic held-out scores do not establish unrestricted natural-language accuracy.

```powershell
docker compose ps -a
docker compose logs -f training api drift-worker
docker compose down
```

`down` stops/removes containers while preserving PostgreSQL, MLflow SQLite and
artifacts, and the active-model manifest on named volumes. `docker compose down -v`
also deletes those local records/models and causes training on the next startup.

The stack has local demo database defaults and binds web ports to localhost.
Optionally copy `.env.example` to `.env` and change its database password before
the first start. Use letters/numbers in this password because the stack constructs
its internal connection URL. PostgreSQL is accessible only inside Compose.

## Demonstrate normal traffic and drift

Wait for the initial simulator to complete, then evaluate its latest 100 logs:

```powershell
docker compose ps -a simulator
docker compose run --rm --no-deps drift-worker python -m monitoring.drift --reference data/reference_simulator.jsonl --once
docker compose run --rm --no-deps simulator --mode drift --count 100
docker compose run --rm --no-deps drift-worker python -m monitoring.drift --reference data/reference_simulator.jsonl --once
```

Confirm that the simulator shows `Exited (0)` before evaluating its initial
traffic. The long-running worker also evaluates new windows every 60 seconds, skipping
unchanged windows. The simulator stores predictions and independently supplied
labels in `prediction_logs`; the worker stores drift, exact match, micro-F1,
coverage, and real Evidently report snapshots in `drift_runs`. PostgreSQL is the
primary store, including when an optional JSONL export is requested.

The integrated worker uses `data/reference_simulator.jsonl`, a frozen capture of
100 real API predictions from the normal simulator pool. The original
`data/reference_traffic.jsonl` remains the standalone monitoring-test reference.
To deliberately replace the integrated reference after reviewing a new baseline,
run `python -m monitoring.reference --api-url http://localhost:8000`; ordinary
startup never recaptures it.

Inspect current results:

```powershell
docker compose exec postgres psql -U cardquery -d cardquery -c "SELECT window_end, drift_share, model_metrics FROM drift_runs ORDER BY window_end DESC LIMIT 5;"
```

Adjust the username/database arguments if customized. Send normal traffic again
with `docker compose run --rm --no-deps simulator --mode normal --count 100`. The dashboard
owner can use the live tables and `/metrics`; dashboard/alert demonstrations and
screenshots remain for the dashboard delivery.

To train and activate a new registered version:

```powershell
docker compose run --rm --no-deps training python -m api.bootstrap --retrain
docker compose restart api
```

## Tools and architecture

- scikit-learn: TF-IDF/logistic-regression parser with explicit numeric rules.
- Prefect: orchestrates candidate training and final evaluation locally.
- MLflow: tracks three validation candidates and a final run; registers the
  complete pyfunc bundle with source and pinned dependencies.
- FastAPI and Flask: serve model/search endpoints and the separate search page.
- PostgreSQL: stores traffic, known labels, and evaluation reports persistently.
- Evidently: computes text-feature KS drift and labeled full-filter accuracy.
- prometheus-client: emits bounded-label request counters and duration histograms.
- Docker Compose: orders readiness/bootstrap jobs and persists service state.
- GitHub Actions/GHCR: lint, ML tests, monitoring tests, container integration,
  and image publishing after successful main-branch CI.

Diagram source: [docs/architecture.mmd](docs/architecture.mmd).

## Development and verification

Use Python 3.11+ and a virtual environment. Application development:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check . --select E9,F63,F7,F82
$env:RUN_MLFLOW_INTEGRATION = "1"
$env:RUN_LOCAL_DEMO_TESTS = "1"
python -m pytest -q
```

The team's optional training test runs its Prefect/MLflow workflow with a fake
parser in a temporary SQLite registry and verifies registered predictions in a
fresh Python process. The container smoke test exercises the real trained parser.
The HTTP integration test exercises the real parser, API, frontend client, and
labeled simulator. Once Compose is running, verify containers and normal/drift
PostgreSQL snapshots with `python scripts/smoke_stack.py`.

Monitoring uses a separate numerical runtime for Evidently 0.4.40; install
`monitoring/requirements.txt` in a **separate virtual environment**, then run
`python -m pytest tests/test_monitoring.py -q`. Set `TEST_DATABASE_URL` to a
throwaway PostgreSQL database and apply `monitoring/sql/schema.sql` to enable the
live round-trip test. CI tests the application and monitoring environments separately.

For quick local search without MLflow, run `python -m uvicorn api.main:app --port
8000` and, in another terminal, `python -m frontend.app`. To use a local registry,
follow [training/README.md](training/README.md), set `CARDQUERY_API_MODE=registry`
and the same `MLFLOW_TRACKING_URI` before starting the API. Compose supplies
these settings automatically.

Use either Docker or the local Python servers. The local frontend checks for an
occupied port and exits with an explanation when another server is already there.

The quick local command uses the committed baseline artifact. The Compose app
uses the registered semantic model with pretrained MiniLM features.

PR checks are defined in `.github/workflows/ci.yml`. A repository administrator
must select **Lint and application tests**, **PostgreSQL and Evidently tests**, and
**Container integration** as required checks in the main branch protection/ruleset.
Pushing/merging and remote repository settings are unchanged by local integration.
Successful main-branch CI publishes five images to `ghcr.io/<owner>/<repo>-<service>`
with both `latest` and commit SHA tags. GHCR stores images; the app runs locally.

Component documentation: [catalog](catalog/README.md), [parser](parser/README.md),
[API](api/README.md), [frontend and traffic](frontend/README.md),
[training](training/README.md), and [monitoring](monitoring/README.md).
