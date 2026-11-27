"""Capture a fixed normal reference from real API predictions, without DB writes."""

import argparse
import json
from pathlib import Path

from monitoring.simulator import REPO_ROOT, load_cases, simulate


def capture_reference(api_url: str, path: str) -> None:
    rows = simulate(api_url, load_cases(REPO_ROOT / "data/unseen_normal.csv"), "normal", 100)
    if any(row["status_code"] != 200 or row["predicted_filters"] is None for row in rows):
        raise RuntimeError("reference requires 100 successful real API predictions")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(target)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--api-url", default="http://localhost:8000")
    cli.add_argument("--out", default="data/reference_simulator.jsonl")
    args = cli.parse_args()
    capture_reference(args.api_url, args.out)


if __name__ == "__main__":
    main()
