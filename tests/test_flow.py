"""Unit and integration tests for the training workflow."""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from training import fake_parser
from training.flow import (
    load_data,
    select_best_model,
    split_data,
)

GOLDEN_QUERY = "fire charizards with at least 150 hp"

GOLDEN_FILTERS = {
    "name": "Charizard",
    "types": ["Fire"],
    "hp_min": 150,
    "hp_max": None,
    "rarity": None,
    "year_min": None,
    "year_max": None,
}


@pytest.fixture
def sample_rows():

    return [
        {
            "query": f"fire charizard at least 150 hp example {i}",
            "filters": GOLDEN_FILTERS.copy(),
        }
        for i in range(10)
    ]


@pytest.fixture
def sample_csv(tmp_path, sample_rows):

    path = tmp_path / "query_examples.csv"

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["query", "filters_json"],
        )

        writer.writeheader()

        for row in sample_rows:
            writer.writerow(
                {
                    "query": row["query"],
                    "filters_json": json.dumps(row["filters"]),
                }
            )

    return path


def test_load_data(sample_csv):

    rows = load_data.fn(str(sample_csv))

    assert len(rows) == 10
    assert isinstance(rows[0]["filters"], dict)
    assert set(rows[0]) == {"query", "filters"}


def test_invalid_csv(tmp_path):

    path = tmp_path / "invalid.csv"

    path.write_text(
        "query,label\npikachu,test\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_data.fn(str(path))


def test_split_sizes(sample_rows):

    train, validation, test = split_data.fn(sample_rows)

    assert len(train) == 6
    assert len(validation) == 2
    assert len(test) == 2


def test_split_reproducibility(sample_rows):

    first = split_data.fn(sample_rows)
    second = split_data.fn(sample_rows)

    assert first == second


def test_no_data_leakage(sample_rows):

    train, validation, test = split_data.fn(sample_rows)

    train_queries = {row["query"] for row in train}
    val_queries = {row["query"] for row in validation}
    test_queries = {row["query"] for row in test}

    assert train_queries.isdisjoint(val_queries)
    assert train_queries.isdisjoint(test_queries)
    assert val_queries.isdisjoint(test_queries)


def test_best_model_selection():

    results = [
        {
            "C": 0.1,
            "metrics": {
                "exact_match": 0.50,
                "micro_f1": 0.70,
            },
        },
        {
            "C": 1.0,
            "metrics": {
                "exact_match": 0.80,
                "micro_f1": 0.90,
            },
        },
        {
            "C": 10.0,
            "metrics": {
                "exact_match": 0.95,
                "micro_f1": 0.98,
            },
        },
    ]

    assert select_best_model.fn(results) == 10.0


def test_f1_tiebreaker():

    results = [
        {
            "C": 0.1,
            "metrics": {
                "exact_match": 0.90,
                "micro_f1": 0.92,
            },
        },
        {
            "C": 1.0,
            "metrics": {
                "exact_match": 0.90,
                "micro_f1": 0.95,
            },
        },
    ]

    assert select_best_model.fn(results) == 1.0


def test_smaller_c_tiebreaker():

    results = [
        {
            "C": 10.0,
            "metrics": {
                "exact_match": 0.95,
                "micro_f1": 0.98,
            },
        },
        {
            "C": 1.0,
            "metrics": {
                "exact_match": 0.95,
                "micro_f1": 0.98,
            },
        },
    ]

    assert select_best_model.fn(results) == 1.0


def test_fake_parser_save_load(tmp_path, sample_rows):

    model = fake_parser.train_parser(sample_rows)

    path = tmp_path / "model.joblib"

    fake_parser.save_parser(model, str(path))
    loaded = fake_parser.load_parser(str(path))

    assert loaded == model

    prediction = fake_parser.parse_query(
        loaded,
        sample_rows[0]["query"],
    )

    assert prediction == GOLDEN_FILTERS


def test_fake_parser_invalid_query(sample_rows):

    model = fake_parser.train_parser(sample_rows)

    with pytest.raises(ValueError):
        fake_parser.parse_query(model, "")


@pytest.mark.skipif(
    os.getenv("RUN_MLFLOW_INTEGRATION") != "1",
    reason="Enable local MLflow integration explicitly",
)
def test_training_registry_integration(tmp_path, sample_csv):

    repo_root = Path(__file__).resolve().parents[1]

    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"

    env = os.environ.copy()

    env["MLFLOW_TRACKING_URI"] = tracking_uri
    env["MLFLOW_REGISTRY_URI"] = tracking_uri
    env["CARDQUERY_PARSER_BACKEND"] = "fake"
    env["PYTHONPATH"] = str(repo_root)

    script = f"""
from training.flow import run_training
from mlflow import MlflowClient

report = run_training(
    {str(sample_csv)!r},
    experiment_name="test-query-parser",
)

assert set(report) == {{
    "run_id",
    "model_name",
    "model_version",
    "metrics",
}}

assert report["model_name"] == "pokemon-query-parser"
assert isinstance(report["model_version"], str)

assert set(report["metrics"]) == {{
    "exact_match",
    "micro_f1",
}}

client = MlflowClient()
experiment = client.get_experiment_by_name("test-query-parser")
runs = client.search_runs([experiment.experiment_id])

assert len(runs) == 4
assert all(run.info.status == "FINISHED" for run in runs)

print("Training and registration passed")
"""

    subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        check=True,
        timeout=300,
    )

    # Reload in a separate process outside the repository.
    env.pop("PYTHONPATH", None)
    env["CARDQUERY_PARSER_BACKEND"] = "real"

    reload_script = f"""
import json
import mlflow
import pandas as pd

model = mlflow.pyfunc.load_model(
    "models:/pokemon-query-parser/1"
)

queries = pd.DataFrame({{
    "query": [
        {GOLDEN_QUERY!r},
        "another test query",
    ]
}})

predictions = model.predict(queries)

assert list(predictions.columns) == ["filters_json"]
assert len(predictions) == 2

for prediction in predictions["filters_json"]:
    assert json.loads(prediction) == {GOLDEN_FILTERS!r}

print("Registered model reload passed")
"""

    subprocess.run(
        [sys.executable, "-c", reload_script],
        cwd=tmp_path,
        env=env,
        check=True,
        timeout=300,
    )
