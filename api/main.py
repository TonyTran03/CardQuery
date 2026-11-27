"""CardQuery HTTP endpoints and local model wiring."""

from __future__ import annotations

import logging
import os
from functools import partial
from pathlib import Path
from typing import Annotated, Callable

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from api.stubs import DEMO_CARDS, stub_instrument_api, stub_parse
from api.query_interpreter import QueryInterpreter, normalize_search_query
from catalog.cards import load_cards
from parser import load_parser, parse_query
from parser.contracts import InvalidQueryError, validate_filters
from search.search import CardSearchIndex
from monitoring.metrics import instrument_api

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)


class RequestBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class QueryRequest(RequestBody):
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]


class PageRequest(RequestBody):
    limit: int = Field(20, ge=1, le=100)
    offset: int = Field(0, ge=0)


class SearchRequest(QueryRequest, PageRequest):
    pass


class Filters(RequestBody):
    name: str | None
    types: list[str]
    hp_min: int | None = Field(ge=0)
    hp_max: int | None = Field(ge=0)
    rarity: str | None
    year_min: int | None = Field(ge=1, le=9999)
    year_max: int | None = Field(ge=1, le=9999)

    @model_validator(mode="before")
    @classmethod
    def normalize(cls, values):
        return validate_filters(values)


class CardSearchRequest(PageRequest):
    filters: Filters


def create_app(
    cards: list[dict],
    parse: Callable[[str], dict] | None,
    model_version: str = "stub-v1",
    instrument: Callable[[FastAPI], None] = stub_instrument_api,
) -> FastAPI:
    catalog = CardSearchIndex(cards)
    interpreter = QueryInterpreter(cards)
    app = FastAPI(title="CardQuery API")

    def parse_filters(query: str) -> dict:
        if parse is None:
            raise HTTPException(status_code=503, detail="model_unavailable")
        try:
            normalized = normalize_search_query(query)
            filters = validate_filters(parse(normalized))
            return interpreter.validate_prediction(normalized, filters)
        except InvalidQueryError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("CardQuery model prediction failed")
            raise HTTPException(status_code=503, detail="model_unavailable") from exc

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready", responses={503: {"description": "Model unavailable"}})
    def ready() -> dict:
        if parse is None:
            raise HTTPException(status_code=503, detail="model_unavailable")
        return {"status": "ok"}

    @app.post("/parse")
    def parse_route(body: QueryRequest) -> dict:
        return {"filters": parse_filters(body.query), "model_version": model_version}

    @app.post("/cards/search")
    def filter_route(body: CardSearchRequest) -> dict:
        return catalog.search(body.filters.model_dump(), body.limit, body.offset)

    @app.post("/search")
    def search_route(body: SearchRequest) -> dict:
        filters = parse_filters(body.query)
        result = catalog.search(filters, body.limit, body.offset)
        return {
            "filters": filters,
            "cards": result["cards"],
            "total": result["total"],
            "model_version": model_version,
        }

    instrument(app)
    return app


def create_local_app() -> FastAPI:
    mode = os.environ.get("CARDQUERY_API_MODE", "local")
    if mode == "stub":
        return create_app(DEMO_CARDS, stub_parse)
    if mode not in {"local", "registry"}:
        raise ValueError("CARDQUERY_API_MODE must be 'local', 'registry', or 'stub'")

    cards_path = os.environ.get("CARDQUERY_CARDS_PATH", str(ROOT / "data/cards.csv"))
    model_path = os.environ.get("CARDQUERY_MODEL_PATH", str(ROOT / "artifacts/parser/model.joblib"))
    model_version = os.environ.get("CARDQUERY_MODEL_VERSION", "local-v1")
    cards = load_cards(cards_path)
    try:
        if mode == "registry":
            from api.model_loader import load_registered_parser

            parse, model_version = load_registered_parser(os.environ.get(
                "CARDQUERY_ACTIVE_MODEL_PATH", str(ROOT / "artifacts/active_model.json"),
            ))
        else:
            parse = partial(parse_query, load_parser(model_path))
        validate_filters(parse("cards"))
    except Exception:
        logger.exception("CardQuery model could not be initialized")
        return create_app(cards, None, model_version, instrument=instrument_api)
    return create_app(cards, parse, model_version, instrument=instrument_api)


app = create_local_app()
