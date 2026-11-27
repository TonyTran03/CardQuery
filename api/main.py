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
from catalog.cards import load_cards
from parser import load_parser, parse_query
from parser.contracts import InvalidQueryError, validate_filters
from search.search import search_cards

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
    app = FastAPI(title="CardQuery API")

    def parse_filters(query: str) -> dict:
        if parse is None:
            raise HTTPException(status_code=503, detail="model_unavailable")
        try:
            return validate_filters(parse(query))
        except InvalidQueryError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("CardQuery model prediction failed")
            raise HTTPException(status_code=503, detail="model_unavailable") from exc

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/parse")
    def parse_route(body: QueryRequest) -> dict:
        return {"filters": parse_filters(body.query), "model_version": model_version}

    @app.post("/cards/search")
    def filter_route(body: CardSearchRequest) -> dict:
        return search_cards(cards, body.filters.model_dump(), body.limit, body.offset)

    @app.post("/search")
    def search_route(body: SearchRequest) -> dict:
        filters = parse_filters(body.query)
        result = search_cards(cards, filters, body.limit, body.offset)
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
    if mode != "local":
        raise ValueError("CARDQUERY_API_MODE must be 'local' or 'stub'")

    cards_path = os.environ.get("CARDQUERY_CARDS_PATH", str(ROOT / "data/cards.csv"))
    model_path = os.environ.get("CARDQUERY_MODEL_PATH", str(ROOT / "artifacts/parser/model.joblib"))
    model_version = os.environ.get("CARDQUERY_MODEL_VERSION", "local-v1")
    cards = load_cards(cards_path)
    try:
        model = load_parser(model_path)
    except Exception:
        logger.exception("CardQuery model could not be loaded")
        return create_app(cards, None, model_version)
    return create_app(cards, partial(parse_query, model), model_version)


app = create_local_app()
