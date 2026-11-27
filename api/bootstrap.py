"""Application startup: reuse a healthy pinned model, train only when needed."""

import argparse
import json
from pathlib import Path

from api.model_loader import load_registered_parser

ACTIVE_MODEL_PATH = Path(__file__).resolve().parents[1] / "artifacts/active_model.json"


def train_model():
    from training.bootstrap import main

    main()


def ensure_model(*, retrain=False):
    if ACTIVE_MODEL_PATH.exists() and not retrain:
        _, version = load_registered_parser(ACTIVE_MODEL_PATH)
        print(f"Reusing registered model {version}; training skipped", flush=True)
    else:
        train_model()
        load_registered_parser(ACTIVE_MODEL_PATH)
    return json.loads(ACTIVE_MODEL_PATH.read_text(encoding="utf-8"))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--retrain", action="store_true", help="Train and activate a new registered version")
    args = cli.parse_args()
    print(json.dumps(ensure_model(retrain=args.retrain), indent=2))


if __name__ == "__main__":
    main()
