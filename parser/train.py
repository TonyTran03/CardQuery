"""Local Part 2 training; Part 5 can call the public functions from its flow."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

from .data import assert_disjoint, label_json, load_rows, majority_label, split_rows
from .evaluation import evaluate_parser, score_predictions
from .parser import PARSER_KIND, save_parser, train_parser

ROOT = Path(__file__).resolve().parents[1]


def train(data_path, output, seed=42, pool_paths=()):
    rows = load_rows(data_path)
    pools = [load_rows(path) for path in pool_paths]
    assert_disjoint(rows, *pools)
    training, validation, test = split_rows(rows, seed)
    candidates = []
    for C in (0.1, 1.0, 10.0):
        model = train_parser(training, seed=seed, C=C)
        metrics = evaluate_parser(model, validation)
        candidate = {"C": C, "metrics": metrics}
        candidates.append(candidate)
        print(f"Validation C={C}: {json.dumps(metrics)}", flush=True)
    selected = min(candidates, key=lambda run: (-run["metrics"]["exact_match"], -run["metrics"]["micro_f1"], run["C"]))
    fitted_rows = training + validation
    model = train_parser(fitted_rows, seed=seed, C=selected["C"])
    # Final test is evaluated once after selection/refit, never for tuning.
    metrics = evaluate_parser(model, test)
    baseline = majority_label(fitted_rows)
    baseline_metrics = score_predictions([baseline] * len(test), [row["filters"] for row in test])
    if metrics["exact_match"] <= baseline_metrics["exact_match"]:
        raise ValueError("selected model did not beat the majority-label baseline; artifact was not saved")
    save_parser(model, str(output))
    data_hash = hashlib.sha256("\n".join(row["query"] + "\t" + label_json(row["filters"]) for row in rows).encode()).hexdigest()
    report = {
        "parser_kind": PARSER_KIND, "seed": seed, "selected_C": selected["C"],
        "ngram_range": [1, 2], "numeric_rules_version": model["numeric_rules_version"],
        "preprocessing_version": model["preprocessing_version"],
        "split_sizes": {"train": len(training), "validation": len(validation), "test": len(test), "final_fit": len(fitted_rows)},
        "validation_candidates": candidates, "test_metrics": metrics,
        "baseline_test_metrics": baseline_metrics, "baseline_filters": baseline,
        "data_sha256": data_hash, "sklearn_version": model["sklearn_version"],
        "model_sha256": hashlib.sha256(Path(output).read_bytes()).hexdigest(),
        "python_version": platform.python_version(), "vocabulary": model["vocabulary"],
        "fitted_query_sha256": sorted(hashlib.sha256(row["query"].encode()).hexdigest() for row in fitted_rows),
        "data_note": "Synthetic catalog-grounded queries; random held-out queries share generation templates. Not evidence of unrestricted natural-language accuracy.",
        "unseen_pool_checks": {Path(path).name: len(pool) for path, pool in zip(pool_paths, pools)},
    }
    report_path = Path(output).with_name("evaluation.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Test: {json.dumps(metrics)}; baseline: {json.dumps(baseline_metrics)}", flush=True)
    print(f"Saved {output} and {report_path}", flush=True)
    return report


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data", type=Path, default=ROOT / "data/query_examples.csv")
    cli.add_argument("--out", type=Path, default=ROOT / "artifacts/parser/model.joblib")
    cli.add_argument("--seed", type=int, default=42)
    cli.add_argument("--unseen", type=Path, nargs="*", default=[ROOT / "data/unseen_normal.csv", ROOT / "data/unseen_drift.csv"])
    args = cli.parse_args()
    train(args.data, args.out, args.seed, args.unseen)


if __name__ == "__main__":
    main()
