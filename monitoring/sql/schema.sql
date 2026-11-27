-- Part 6 schema. Applied at startup by monitoring.persistence.apply_schema()
-- and also mounted into postgres /docker-entrypoint-initdb.d for first boot.

CREATE TABLE IF NOT EXISTS prediction_logs (
    request_id        TEXT PRIMARY KEY,
    timestamp         TIMESTAMPTZ NOT NULL,
    query             TEXT NOT NULL,
    mode              TEXT NOT NULL,
    status_code       INTEGER NOT NULL,
    latency_ms        DOUBLE PRECISION NOT NULL,
    model_version     TEXT,
    predicted_filters JSONB,
    expected_filters  JSONB
);

CREATE TABLE IF NOT EXISTS drift_runs (
    run_id            TEXT PRIMARY KEY,
    window_start      TIMESTAMPTZ NOT NULL,
    window_end        TIMESTAMPTZ NOT NULL,
    drift_detected    BOOLEAN NOT NULL,
    drift_share       DOUBLE PRECISION NOT NULL,
    drifted_columns   JSONB NOT NULL,
    model_metrics     JSONB NOT NULL,
    evidently_report  JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_prediction_logs_ts
    ON prediction_logs (timestamp, request_id);
CREATE INDEX IF NOT EXISTS idx_drift_runs_window_end
    ON drift_runs (window_end);
