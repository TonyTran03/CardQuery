"""Train and register the model for application startup."""

import json
import os
from pathlib import Path

from training.adapter import PARSER_BACKEND
from training.flow import run_training


def main():

    if PARSER_BACKEND != "real":
        raise ValueError("Application bootstrap requires the real parser")

    data_path = os.getenv(
        "TRAINING_DATA_PATH",
        "data/query_examples.csv",
    )

    report = run_training(data_path)

    model_uri = f"models:/{report['model_name']}/{report['model_version']}"

    active_model = {
        "model_uri": model_uri,
        "model_version": report["model_version"],
    }

    output_path = Path("artifacts/active_model.json")
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(active_model, indent=2),
        encoding="utf-8",
    )

    print("Training bootstrap completed")
    print(json.dumps(active_model, indent=2))


if __name__ == "__main__":
    main()
