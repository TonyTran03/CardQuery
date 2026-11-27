"""Adapter connecting the training workflow to the parser."""

import os

PARSER_BACKEND = os.getenv(
    "CARDQUERY_PARSER_BACKEND",
    "real",
).lower()


if PARSER_BACKEND == "real":
    from parser import (
        evaluate_parser,
        load_parser,
        parse_query,
        save_parser,
        train_parser,
    )

elif PARSER_BACKEND == "fake":
    from training.fake_parser import (
        evaluate_parser,
        load_parser,
        parse_query,
        save_parser,
        train_parser,
    )

else:
    raise ValueError(f"Unknown parser backend: {PARSER_BACKEND}")


__all__ = [
    "evaluate_parser",
    "load_parser",
    "parse_query",
    "save_parser",
    "train_parser",
]
