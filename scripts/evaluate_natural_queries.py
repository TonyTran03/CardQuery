"""Evaluate a registered candidate before activating it in the application."""

import argparse
import json
from pathlib import Path

from api.model_loader import load_registered_parser
from api.query_interpreter import QueryInterpreter, normalize_search_query
from catalog.cards import load_cards
from parser.contracts import InvalidQueryError

ROOT = Path(__file__).resolve().parents[1]


def evaluate(manifest, output):
    parse, model_version = load_registered_parser(manifest)
    interpreter = QueryInterpreter(load_cards(ROOT / "data/cards.csv"))
    cases = json.loads((ROOT / "tests/fixtures/natural_queries.json").read_text(encoding="utf-8"))
    results = []
    for case in cases:
        query = normalize_search_query(case["query"])
        try:
            prediction = interpreter.validate_prediction(query, parse(query))
            error = None
        except InvalidQueryError as exc:
            prediction, error = None, str(exc)
        results.append({**case, "predicted": prediction, "error": error,
                        "correct": prediction == case["filters"]})
    unknown = []
    for query in ["winston", "steven", "hello", "flanfy", "winston pokemon", "sandwich", "xyzzyblorp"]:
        try:
            normalized = normalize_search_query(query)
            prediction = interpreter.validate_prediction(normalized, parse(normalized))
            unknown.append({"query": query, "rejected": False, "predicted": prediction})
        except InvalidQueryError:
            unknown.append({"query": query, "rejected": True})
    report = {"model_version": model_version, "n_cases": len(cases),
              "exact_match": sum(row["correct"] for row in results) / len(results),
              "results": results, "unknown_queries": unknown,
              "note": "Small, separately authored regression set; not a broad general-language accuracy claim."}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Natural queries: {sum(r['correct'] for r in results)}/{len(cases)}; model {model_version}")
    for result in results:
        if not result["correct"]:
            print("MISMATCH", result["query"], "expected", result["filters"], "actual", result["predicted"], result["error"])
    print("Unknown queries rejected:", sum(row["rejected"] for row in unknown), "/", len(unknown))
    return report


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--manifest", required=True)
    cli.add_argument("--out", default=".integration-results/semantic-upgrade/natural_report.json")
    args = cli.parse_args()
    evaluate(args.manifest, args.out)
