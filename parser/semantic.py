"""Catalog-grounded hybrid parser with learned pretrained MiniLM features."""

from __future__ import annotations

import re
from functools import lru_cache

from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression

from .contracts import InvalidQueryError, empty_filters, normalize_query, validate_filters
from .data import prepare_rows
from .encoder import get_encoder, load_encoder_assets
from .numeric_v2 import VERSION, parse_numeric_v2

KIND = "minilm-tfidf-logreg-hybrid-v2"
NONE = "__NONE__"


class EntityResolver:
    def __init__(self, names):
        from .parser import preprocess_query

        self.patterns = [(name, re.compile(r"(?<!\w)" + re.escape(preprocess_query(name)) + r"s?(?!\w)"))
                         for name in names]
        self.simple_names = {name.lower(): name for name in names if name.isalpha() and len(name) >= 4}

    def resolve(self, query):
        from .parser import preprocess_query

        text = preprocess_query(query)
        matches = sorted([(m.start(), m.end(), name) for name, pattern in self.patterns
                          for m in pattern.finditer(text)], key=lambda m: (-(m[1] - m[0]), m[0]))
        selected = []
        for match in matches:
            if not any(match[0] < end and match[1] > start for start, end, _ in selected):
                selected.append(match)
        if len({name for _, _, name in selected}) > 1:
            raise InvalidQueryError("Search for one Pokémon name at a time")
        if selected:
            for start, end, _ in sorted(selected, reverse=True):
                text = text[:start] + " pokemon " + text[end:]
            return selected[0][2], " ".join(text.split())

        # A conservative, character-based entity correction. Semantic similarity
        # must never map arbitrary people/words to Pokémon names.
        from rapidfuzz import fuzz, process

        for token in re.findall(r"\b[a-z]{4,}\b", text):
            best = process.extract(token, list(self.simple_names), scorer=fuzz.ratio, limit=2)
            if len(best) == 2 and best[0][1] >= 90 and best[0][1] - best[1][1] >= 8:
                name = self.simple_names[best[0][0]]
                return name, re.sub(r"\b" + re.escape(token) + r"\b", "pokemon", text)
        return None, text


@lru_cache(maxsize=8)
def resolver(names):
    return EntityResolver(names)


def semantic_texts(model, queries):
    entity_resolver = resolver(tuple(model["vocabulary"]["names"]))
    return [re.sub(r"\d+(?:\.\d+)?", "number", entity_resolver.resolve(query)[1]) for query in queries]


def features(model, queries):
    texts = semantic_texts(model, queries)
    words = model["vectorizer"].transform(texts)
    embeddings = get_encoder(model["encoder"]).encode(texts)
    return hstack([words, csr_matrix(embeddings * 3.0)], format="csr")


def train_semantic_parser(rows, seed=42, C=1.0):
    from .parser import train_parser

    rows = prepare_rows(rows)
    # Preserve the baseline's shared TF-IDF representation and name head.
    model = train_parser(rows, seed=seed, C=C, kind="baseline")
    model.update(parser_kind=KIND, numeric_rules_version=VERSION,
                 encoder=load_encoder_assets(), semantic_features_version="v1")
    x = features(model, [row["query"] for row in rows])

    def fit(targets):
        return LogisticRegression(C=C, max_iter=1000, random_state=seed).fit(x, targets)

    model["rarity_classifier"] = fit([row["filters"]["rarity"] or NONE for row in rows])
    model["type_classifiers"] = {kind: fit([int(kind in row["filters"]["types"]) for row in rows])
                                 for kind in model["vocabulary"]["types"]}
    return model


def _literal_categories(text, values):
    matches = []
    for value in sorted(values, key=len, reverse=True):
        pattern = r"(?<!\w)" + re.escape(value.lower()) + r"(?!\w)"
        match = re.search(pattern, text)
        if match and not any(match.start() < end and match.end() > start for start, end, _ in matches):
            matches.append((match.start(), match.end(), value))
    return [value for _, _, value in matches]


def parse_semantic_queries(model, queries):
    if not queries:
        return []
    normalized = [normalize_query(query) for query in queries]
    numeric = [parse_numeric_v2(query) for query in normalized]
    entity_resolver = resolver(tuple(model["vocabulary"]["names"]))
    entities = [entity_resolver.resolve(query)[0] for query in normalized]
    x = features(model, normalized)
    rarity = model["rarity_classifier"].predict(x)
    probabilities = {kind: classifier.predict_proba(x)[:, list(classifier.classes_).index(1)]
                     for kind, classifier in model["type_classifiers"].items()}
    output = []
    for i, query in enumerate(normalized):
        # Exact category mentions anchor labels; pretrained features handle the
        # paraphrases for which no category word appears verbatim.
        literal_types = _literal_categories(query, model["vocabulary"]["types"])
        literal_rarity = _literal_categories(query, model["vocabulary"]["rarities"])
        result = {**empty_filters(), **numeric[i], "name": entities[i],
                  "types": literal_types or [kind for kind, p in probabilities.items() if p[i] >= 0.5],
                  "rarity": literal_rarity[0] if literal_rarity else (None if rarity[i] == NONE else str(rarity[i]))}
        output.append(validate_filters(result))
    return output
