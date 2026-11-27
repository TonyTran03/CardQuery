"""Prefect training workflow for CardQuery."""

import hashlib
import json
import os
import tempfile
from importlib.metadata import version
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from prefect import flow, task

from parser.data import (
    assert_disjoint,
    load_rows,
    majority_label,
    split_rows,
)
from parser.evaluation import score_predictions
from training.adapter import (
    PARSER_BACKEND,
    evaluate_parser,
    parse_query,
    save_parser,
    train_parser,
)
from training.mlflow_model import QueryParserModel

SEED = 42
CANDIDATES = [0.1, 1.0, 10.0]
MODEL_NAME = "pokemon-query-parser"


@task
def load_data(data_path):
    rows = load_rows(data_path)

    pools = []

    for filename in ["unseen_normal.csv", "unseen_drift.csv"]:
        path = Path(data_path).parent / filename

        if path.exists():
            pools.append(load_rows(path))

    assert_disjoint(rows, *pools)

    print(f"Loaded {len(rows)} queries")

    return rows


@task
def split_data(rows):
    train, validation, test = split_rows(rows, SEED)

    print(f"Training: {len(train)}")
    print(f"Validation: {len(validation)}")
    print(f"Testing: {len(test)}")

    return train, validation, test


@task
def train_candidate(train, validation, C, data_hash):

    with mlflow.start_run(run_name=f"candidate-C-{C}"):
        model = train_parser(train, seed=SEED, C=C)

        metrics = evaluate_parser(model, validation)

        mlflow.log_params(
            {
                "C": C,
                "seed": SEED,
                "parser_kind": model["parser_kind"],
                "ngram_range": "(1, 2)",
                "numeric_rules_version": model["numeric_rules_version"],
                "train_size": len(train),
                "validation_size": len(validation),
                "data_sha256": data_hash,
            }
        )

        mlflow.set_tag("stage", "validation")
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.joblib"
            save_parser(model, str(path))
            mlflow.log_artifact(str(path))

        return {
            "C": C,
            "metrics": metrics,
        }


@task
def select_best_model(results):

    best = min(
        results,
        key=lambda result: (
            -result["metrics"]["exact_match"],
            -result["metrics"]["micro_f1"],
            result["C"],
        ),
    )

    print(f"Selected C: {best['C']}")

    return best["C"]


@task
def train_final_model(train, validation, test, C, data_hash):

    fitted_rows = train + validation

    with mlflow.start_run(run_name="final-model") as run:
        model = train_parser(
            fitted_rows,
            seed=SEED,
            C=C,
        )

        metrics = evaluate_parser(model, test)

        baseline = majority_label(fitted_rows)

        baseline_metrics = score_predictions(
            [baseline] * len(test),
            [row["filters"] for row in test],
        )

        mlflow.log_params(
            {
                "C": C,
                "seed": SEED,
                "parser_kind": model["parser_kind"],
                "ngram_range": "(1, 2)",
                "numeric_rules_version": model["numeric_rules_version"],
                "train_size": len(train),
                "validation_size": len(validation),
                "test_size": len(test),
                "final_fit_size": len(fitted_rows),
                "data_sha256": data_hash,
            }
        )

        mlflow.set_tag("stage", "final_test")
        mlflow.log_metrics(metrics)

        mlflow.log_metrics(
            {
                "baseline_exact_match": baseline_metrics["exact_match"],
                "baseline_micro_f1": baseline_metrics["micro_f1"],
            }
        )

        if (
            PARSER_BACKEND == "real"
            and metrics["exact_match"] <= baseline_metrics["exact_match"]
        ):
            raise ValueError("Final model did not beat baseline")

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.joblib"

            save_parser(model, str(path))
            mlflow.log_artifact(str(path))

            input_example = pd.DataFrame(
                {"query": ["fire charizards with at least 150 hp"]}
            )

            output_example = pd.DataFrame(
                {
                    "filters_json": [
                        json.dumps(
                            parse_query(
                                model,
                                input_example["query"].iloc[0],
                            )
                        )
                    ]
                }
            )

            signature = infer_signature(
                input_example,
                output_example,
            )

            requirements = [
                f"{package}=={version(package)}"
                for package in [
                    "mlflow",
                    "pandas",
                    "numpy",
                    "scipy",
                    "scikit-learn",
                    "joblib",
                    "cloudpickle",
                    "pydantic",
                ]
            ]
            if model["parser_kind"] == "minilm-tfidf-logreg-hybrid-v2":
                requirements.extend(f"{package}=={version(package)}"
                                    for package in ["onnxruntime", "tokenizers", "rapidfuzz", "requests"])
                mlflow.log_params({"encoder_model": model["encoder"]["repository"],
                                   "encoder_revision": model["encoder"]["revision"],
                                   "semantic_features_version": model["semantic_features_version"]})

            model_info = mlflow.pyfunc.log_model(
                name="parser-model",
                python_model=QueryParserModel(backend=PARSER_BACKEND),
                artifacts={"parser": str(path)},
                code_paths=[
                    str(Path(__file__).resolve().parents[1] / folder)
                    for folder in ["parser", "shared", "training"]
                    if (Path(__file__).resolve().parents[1] / folder).is_dir()
                ],
                pip_requirements=requirements,
                signature=signature,
                input_example=input_example,
            )

            registered = mlflow.register_model(
                model_uri=model_info.model_uri,
                name=MODEL_NAME,
            )

        return {
            "run_id": run.info.run_id,
            "model_name": MODEL_NAME,
            "model_version": str(registered.version),
            "metrics": metrics,
        }


@flow(name="cardquery-training")
def run_training(
    data_path: str,
    experiment_name: str = "query-parser",
) -> dict:

    mlflow.set_tracking_uri(
        os.environ.get(
            "MLFLOW_TRACKING_URI",
            "http://127.0.0.1:5001",
        )
    )

    mlflow.set_experiment(experiment_name)

    data_hash = hashlib.sha256(Path(data_path).read_bytes()).hexdigest()

    rows = load_data(data_path)

    train, validation, test = split_data(rows)

    results = []

    for C in CANDIDATES:
        result = train_candidate(
            train,
            validation,
            C,
            data_hash,
        )
        results.append(result)

    best_C = select_best_model(results)

    report = train_final_model(
        train,
        validation,
        test,
        best_C,
        data_hash,
    )

    return report


if __name__ == "__main__":
    report = run_training("data/query_examples.csv")

    print(report)
