"""Load a pinned registered version and adapt it to the API parse callable."""

import json
import re
from pathlib import Path

from shared.contracts import validate_filters


def load_registered_parser(manifest_path):
    import mlflow.pyfunc
    import pandas as pd

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    version = manifest["model_version"]
    uri = manifest["model_uri"]
    if not isinstance(version, str) or not re.fullmatch(r"[1-9][0-9]*", version):
        raise ValueError("model_version must be a positive registry version string")
    if uri != f"models:/pokemon-query-parser/{version}":
        raise ValueError("model_uri must pin the declared registered model version")
    model = mlflow.pyfunc.load_model(uri)

    def parse(query):
        result = model.predict(pd.DataFrame({"query": [query]}))
        if list(result.columns) != ["filters_json"] or len(result) != 1:
            raise ValueError("registered model returned an invalid prediction shape")
        return validate_filters(json.loads(result.iloc[0]["filters_json"]))

    validate_filters(parse("cards"))
    return parse, version
