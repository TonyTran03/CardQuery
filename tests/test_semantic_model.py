"""Opt-in checks of the actual registered, self-contained pretrained model."""

import os

import pytest

pytestmark = pytest.mark.skipif(not os.getenv("CARDQUERY_SEMANTIC_MANIFEST"),
                                reason="needs a registered semantic candidate")


def test_registered_model_passes_natural_language_gate(tmp_path):
    from scripts.evaluate_natural_queries import evaluate

    report = evaluate(os.environ["CARDQUERY_SEMANTIC_MANIFEST"], tmp_path / "natural.json")
    assert report["exact_match"] >= 0.9
    assert all(row["rejected"] for row in report["unknown_queries"])
    more_than = next(row for row in report["results"] if row["query"] == "charizard more than 150 hp")
    assert more_than["correct"]


def test_encoder_assets_are_packaged_for_offline_inference(monkeypatch):
    import json
    from pathlib import Path

    import mlflow.pyfunc
    import pandas as pd

    manifest = json.loads(Path(os.environ["CARDQUERY_SEMANTIC_MANIFEST"]).read_text())
    model = mlflow.pyfunc.load_model(manifest["model_uri"])
    bundle = model.unwrap_python_model().model
    assert bundle["parser_kind"] == "minilm-tfidf-logreg-hybrid-v2"
    assert len(bundle["encoder"]["onnx"]) > 20_000_000

    def forbid_download():
        raise AssertionError("prediction must use the packaged encoder, without downloading weights")

    monkeypatch.setattr("parser.encoder.load_encoder_assets", forbid_download)
    result = model.predict(pd.DataFrame({"query": ["charizard more than 150 hp"]}))
    assert json.loads(result.iloc[0]["filters_json"])["hp_min"] == 151
