"""Smoke test for the registered CardQuery model."""

import json

import mlflow
import pandas as pd

MODEL_URI = "models:/pokemon-query-parser/1"


def main():

    model = mlflow.pyfunc.load_model(MODEL_URI)

    queries = [
        "fire charizards with at least 150 hp",
        "pikachu",
        "water pokemon",
        "pokemon with under 100 hp",
        "rare pokemon from 2020",
    ]

    inputs = pd.DataFrame({"query": queries})
    predictions = model.predict(inputs)

    expected_keys = {
        "name",
        "types",
        "hp_min",
        "hp_max",
        "rarity",
        "year_min",
        "year_max",
    }

    assert list(predictions.columns) == ["filters_json"]
    assert len(predictions) == len(queries)

    for query, prediction in zip(queries, predictions["filters_json"]):
        filters = json.loads(prediction)

        assert set(filters.keys()) == expected_keys

        print(f"\nQuery: {query}")
        print(json.dumps(filters, indent=4))

    golden = json.loads(predictions["filters_json"].iloc[0])

    assert golden == {
        "name": "Charizard",
        "types": ["Fire"],
        "hp_min": 150,
        "hp_max": None,
        "rarity": None,
        "year_min": None,
        "year_max": None,
    }

    print("\nSmoke test passed successfully!")


if __name__ == "__main__":
    main()
