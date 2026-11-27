"""Fixture-mode adapters and the default Prometheus installer."""

from fastapi import FastAPI, Response
from frontend.fake_api import fake_parse as stub_parse
from frontend.fixtures import DEMO_CARDS


def stub_instrument_api(app: FastAPI) -> None:
    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(
            "# HELP cardquery_api_info CardQuery API process is running.\n"
            "# TYPE cardquery_api_info gauge\n"
            "cardquery_api_info 1\n",
            media_type="text/plain; version=0.0.4",
        )


__all__ = ["DEMO_CARDS", "stub_parse", "stub_instrument_api"]
