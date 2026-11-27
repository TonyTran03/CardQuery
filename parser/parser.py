"""One shared TF-IDF vectorizer, logistic classifiers, and numeric rules."""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .contracts import normalize_query, validate_filters
from .data import prepare_rows
from .numeric_rules import NUMERIC_RULES_VERSION, parse_numeric

PARSER_KIND = "tfidf-logreg-hybrid-v1"
NONE_LABEL = "__NONE__"


def preprocess_query(query: str) -> str:
    """Preserve card variants that the default word tokenizer would conflate.

    Charizard ex and Charizard-EX are different catalog labels. Likewise M
    (Mega) and the Nidoran gender symbols must survive tokenization. These are
    text transformations, never category lookups or prediction overrides.
    """
    text = normalize_query(query).replace("♀", " female ").replace("♂", " male ")
    text = re.sub(r"-ex\b", " hyphenex ", text)
    text = re.sub(r"\bm\s+(?=[a-z])", "mega ", text)
    return " ".join(text.split())


def train_parser(rows: list[dict], seed: int = 42, C: float = 1.0, *, kind: str | None = None):
    selected_kind = kind or os.getenv("CARDQUERY_PARSER_KIND", "baseline")
    if selected_kind == "semantic":
        from .semantic import train_semantic_parser

        return train_semantic_parser(rows, seed, C)
    if selected_kind != "baseline":
        raise ValueError("CARDQUERY_PARSER_KIND must be baseline or semantic")
    if isinstance(C, bool) or not isinstance(C, (int, float)) or not math.isfinite(C) or C <= 0:
        raise ValueError("C must be a finite positive number")
    if type(seed) is not int or not 0 <= seed < 2 ** 32:
        raise ValueError("seed must be an integer in 0..2**32-1")
    rows = prepare_rows(rows)
    labels = [row["filters"] for row in rows]
    vocabulary = {
        "names": sorted({label["name"] for label in labels if label["name"] is not None}),
        "rarities": sorted({label["rarity"] for label in labels if label["rarity"] is not None}),
        "types": sorted({t for label in labels for t in label["types"]}, key=str.casefold),
    }
    targets = {field: [label[field] or NONE_LABEL for label in labels]
               for field in ("name", "rarity")}
    for field, values in targets.items():
        if NONE_LABEL in vocabulary["names" if field == "name" else "rarities"]:
            raise ValueError(f"{NONE_LABEL} is reserved for a missing {field}")
        if len(set(values)) < 2 or NONE_LABEL not in values:
            raise ValueError(f"{field} classifier needs at least two classes, including no-{field} examples")
    if not vocabulary["types"]:
        raise ValueError("type classifiers need examples with a type and without that type")
    type_targets = {t: [int(t in label["types"]) for label in labels] for t in vocabulary["types"]}
    for card_type, values in type_targets.items():
        if len(set(values)) < 2:
            raise ValueError(f"type {card_type!r} needs both present and absent examples")
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
    features = vectorizer.fit_transform([preprocess_query(row["query"]) for row in rows])

    def fit(values):
        return LogisticRegression(C=C, max_iter=1000, random_state=seed).fit(features, values)

    return {
        "parser_kind": PARSER_KIND,
        "numeric_rules_version": NUMERIC_RULES_VERSION,
        "preprocessing_version": "v1",
        "vectorizer": vectorizer,
        "name_classifier": fit(targets["name"]),
        "rarity_classifier": fit(targets["rarity"]),
        "type_classifiers": {t: fit(values) for t, values in type_targets.items()},
        "vocabulary": vocabulary,
        "seed": seed, "C": float(C), "n_training_rows": len(rows),
        "sklearn_version": sklearn.__version__,
    }


def validate_bundle(model):
    if isinstance(model, dict) and model.get("parser_kind") == "minilm-tfidf-logreg-hybrid-v2":
        required = {"vectorizer", "name_classifier", "rarity_classifier", "type_classifiers", "vocabulary", "encoder"}
        if not required <= model.keys():
            raise ValueError("incomplete semantic parser bundle")
        if model.get("numeric_rules_version") != "v2" or model.get("semantic_features_version") != "v1":
            raise ValueError("unsupported semantic parser version")
        from .encoder import FILES, REVISION
        import hashlib

        assets = model.get("encoder", {})
        if assets.get("revision") != REVISION or any(
            not isinstance(assets.get(key), bytes) or hashlib.sha256(assets[key]).hexdigest() != info[2]
            for key, info in FILES.items()
        ):
            raise ValueError("invalid or incomplete pretrained encoder assets")
        if model.get("sklearn_version") != sklearn.__version__:
            raise ValueError("semantic model scikit-learn version does not match runtime")
        return model
    if not isinstance(model, dict) or model.get("parser_kind") != PARSER_KIND:
        raise ValueError(f"expected a {PARSER_KIND} model bundle")
    if model.get("numeric_rules_version") != NUMERIC_RULES_VERSION or model.get("preprocessing_version") != "v1":
        raise ValueError("model rule/preprocessing version is not supported")
    required = {"vectorizer", "name_classifier", "rarity_classifier", "type_classifiers", "vocabulary"}
    if not required <= model.keys():
        raise ValueError("incomplete parser bundle")
    if model.get("sklearn_version") != sklearn.__version__:
        raise ValueError(f"model uses scikit-learn {model.get('sklearn_version')}; installed version is {sklearn.__version__}; retrain or install pinned dependencies")
    return model


def parse_queries(model, queries: list[str]) -> list[dict]:
    """Batch inference used by evaluation; the public one-query API stays simple."""
    validate_bundle(model)
    if model["parser_kind"] == "minilm-tfidf-logreg-hybrid-v2":
        from .semantic import parse_semantic_queries

        return parse_semantic_queries(model, queries)
    if not queries:
        return []
    numbers = [parse_numeric(query) for query in queries]
    features = model["vectorizer"].transform([preprocess_query(query) for query in queries])
    names = model["name_classifier"].predict(features)
    rarities = model["rarity_classifier"].predict(features)
    probabilities = {}
    for card_type, classifier in model["type_classifiers"].items():
        positive = list(classifier.classes_).index(1)
        probabilities[card_type] = classifier.predict_proba(features)[:, positive]
    return [validate_filters({
        "name": None if names[i] == NONE_LABEL else str(names[i]),
        "types": sorted([t for t, values in probabilities.items() if values[i] >= 0.5], key=str.casefold),
        "rarity": None if rarities[i] == NONE_LABEL else str(rarities[i]),
        **numbers[i],
    }) for i in range(len(queries))]


def parse_query(model, query: str) -> dict:
    return parse_queries(model, [query])[0]


def save_parser(model, path: str) -> None:
    validate_bundle(model)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination, compress=3)


def load_parser(path: str):
    """Load only trusted local artifacts: joblib uses Python pickle internally."""
    return validate_bundle(joblib.load(path))
