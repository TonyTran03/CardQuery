# Dashboard integration handoff (pending component)

The application stack deliberately leaves Grafana, Prometheus scraping, and alert
provisioning to the remaining dashboard delivery. No placeholder dashboard or
synthetic alert is started. Add its Compose services to the root stack once ready.

## API metrics

Scrape `http://api:8000/metrics` from the Compose network. The installer is
`monitoring.metrics.instrument_api(app)`, already wired into local/registry mode.
Each app owns a separate registry, so tests and multiple app factories do not
double-register collectors. `/metrics` is excluded from counting; unknown paths
use `unmatched` to bound label cardinality.

- `api_requests_total{method,path,status}`: request counts, including 422/5xx.
- `api_request_duration_seconds{bucket,sum,count}`: histogram with `method,path`
  labels and buckets 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, +Inf.

Use `rate(api_requests_total[5m])` for throughput, filter `status=~"5.."` for errors,
and `histogram_quantile(0.95, sum by (le) (rate(api_request_duration_seconds_bucket[5m])))`
for p95. Scope/sum labels as appropriate for the supplied dashboard.

## PostgreSQL

Connect to `postgres:5432`, database/user from the stack's `.env` settings.
Use the same database as the worker. Do not load `monitoring/sql/seed.sql` into
the live demonstration: the simulator and worker produce real rows automatically.

```sql
SELECT window_end AS time,
       drift_share,
       (model_metrics->>'exact_match')::double precision AS exact_match,
       (model_metrics->>'micro_f1')::double precision AS micro_f1,
       (model_metrics->>'label_coverage')::double precision AS label_coverage
FROM drift_runs
ORDER BY window_end DESC
LIMIT 1;
```

`evidently_report->'drift'->'snapshot'` and
`evidently_report->'performance'->'snapshot'` store real report snapshots.
Null exact-match/F1 values mean no valid labeled successes. Coverage includes
failed/unlabeled requests in its denominator. Drift uses only query length and
token count against the frozen reference, never mode labels or predictions.

The planned alert is latest drift share >=0.5 for 60 seconds, evaluated every
30 seconds. An actual firing-alert demonstration and normal/drift screenshots
remain pending. Root README commands generate the two traffic conditions.
