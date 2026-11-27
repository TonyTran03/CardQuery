"""MLflow inference wrapper for CardQuery."""

import json

import mlflow
import pandas as pd


class QueryParserModel(mlflow.pyfunc.PythonModel):
    def __init__(self, backend="real"):
        self.backend = backend

    def load_context(self, context):

        if self.backend == "fake":
            from training import fake_parser as parser
        else:
            import parser

        self.model = parser.load_parser(context.artifacts["parser"])

    def predict(
        self,
        context,
        model_input,
        params=None,
    ) -> pd.DataFrame:

        if self.backend == "fake":
            from training import fake_parser as parser
        else:
            import parser

        if list(model_input.columns) != ["query"]:
            raise ValueError("Expected a DataFrame with one column: query")

        predictions = [
            json.dumps(parser.parse_query(self.model, query))
            for query in model_input["query"]
        ]

        return pd.DataFrame({"filters_json": predictions})
