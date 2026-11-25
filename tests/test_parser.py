"""Part 2 offline ML checks. No APIs, credentials, or downloaded models."""

import copy
import itertools
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pytest

from catalog.cards import load_cards
from parser import evaluate_parser, load_parser, parse_query, save_parser, train_parser
from parser.contracts import FILTER_FIELDS, InvalidQueryError, empty_filters, validate_filters
from parser.data import assert_disjoint, load_rows, majority_label, prepare_rows, split_rows, write_rows
from parser.demo_server import demo_cases, preview_cards
from parser.evaluation import score_predictions
from parser.numeric_rules import parse_numeric
from parser.parser import NONE_LABEL, parse_queries, preprocess_query

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def model():
    return load_parser(str(ROOT / "artifacts/parser/model.joblib"))


@pytest.fixture(scope="module")
def dataset():
    return load_rows(ROOT / "data/query_examples.csv")


def tiny_rows():
    rows = []
    for name, types, rarity in itertools.product([None, "Charizard", "Pikachu"], [[], ["Fire"], ["Water"], ["Fire", "Water"]], [None, "Rare"]):
        filters = {**empty_filters(), "name": name, "types": types, "rarity": rarity}
        query = " ".join([rarity or "", " or ".join(types), name or "", "cards"])
        rows.append({"query": query, "filters": filters})
    return rows


def test_catalog_vocabulary_is_used_in_labels_and_bundle(model, dataset):
    cards = load_cards(str(ROOT / "data/cards.csv"))
    assert model["vocabulary"]["names"] == sorted({card["name"] for card in cards})
    assert model["vocabulary"]["rarities"] == sorted({card["rarity"] for card in cards if card["rarity"]})
    assert set(model["vocabulary"]["types"]) == {t for card in cards for t in card["types"]}
    assert len(dataset) >= 100
    for row in dataset:
        assert set(row) == {"query", "filters"}
        assert validate_filters(row["filters"]) == row["filters"]
        for field, value in parse_numeric(row["query"]).items():
            assert row["filters"][field] == value


def test_unseen_pools_are_disjoint_and_deliberately_shifted(dataset):
    normal = load_rows(ROOT / "data/unseen_normal.csv")
    drift = load_rows(ROOT / "data/unseen_drift.csv")
    assert_disjoint(dataset, normal, drift)
    assert len(normal) >= 100 and len(drift) >= 100
    assert np.mean([len(row["query"].split()) for row in drift]) > 2 * np.mean([len(row["query"].split()) for row in normal])


@pytest.mark.parametrize("test", [case for case in demo_cases() if "expected" in case])
def test_hand_authored_examples(model, test):
    assert parse_query(model, test["query"]) == test["expected"]


def test_case_whitespace_and_output_types(model):
    filters = parse_query(model, "   FIRE   CHARIZARDS with AT LEAST 150 HP  ")
    assert filters == demo_cases()[0]["expected"]
    assert set(filters) == set(FILTER_FIELDS)
    assert type(filters["hp_min"]) is int
    assert isinstance(filters["types"], list)


@pytest.mark.parametrize("query,name", [
    ("Charizard ex cards", "Charizard ex"), ("Charizard-EX cards", "Charizard-EX"),
    ("M Charizard-EX cards", "M Charizard-EX"), ("Nidoran ♀ cards", "Nidoran ♀"),
    ("Nidoran ♂ cards", "Nidoran ♂"),
])
def test_variant_preprocessing_preserves_catalog_distinctions(model, query, name):
    assert parse_query(model, query)["name"] == name


def test_preprocessing_is_shared_and_vectorizer_only_fits_training():
    rows = tiny_rows()
    original = copy.deepcopy(rows)
    model = train_parser(rows, C=10)
    assert rows == original
    vocabulary = model["vectorizer"].vocabulary_.copy()
    parse_query(model, "unseenmagicword cards")
    assert "unseenmagicword" not in vocabulary
    assert model["vectorizer"].vocabulary_ == vocabulary
    assert model["vectorizer"].ngram_range == (1, 2)
    assert model["vectorizer"].lowercase is True
    assert preprocess_query("  M Charizard-EX ♀ ") == "mega charizard hyphenex female"


def test_type_models_have_independent_binary_targets(model):
    assert set(model["type_classifiers"]) == set(model["vocabulary"]["types"])
    for classifier in model["type_classifiers"].values():
        assert list(classifier.classes_) == [0, 1]
        assert classifier.C == model["C"] and classifier.max_iter == 1000
    assert NONE_LABEL in model["name_classifier"].classes_
    assert NONE_LABEL in model["rarity_classifier"].classes_


def test_split_matches_the_exact_team_algorithm(dataset):
    train, validation, test = split_rows(dataset)
    expected = sorted(dataset, key=lambda row: row["query"])
    random.Random(42).shuffle(expected)
    size = int(np.ceil(0.2 * len(dataset)))
    assert test == expected[:size]
    assert validation == expected[size:2 * size]
    assert train == expected[2 * size:]
    assert_disjoint(train, validation, test)


def test_selected_model_beats_majority_on_held_out_data(model, dataset):
    train, validation, held_out = split_rows(dataset)
    metrics = evaluate_parser(model, held_out)
    baseline = majority_label(train + validation)
    baseline_metrics = score_predictions([baseline] * len(held_out), [row["filters"] for row in held_out])
    assert metrics["exact_match"] > baseline_metrics["exact_match"]
    report = json.loads((ROOT / "artifacts/parser/evaluation.json").read_text())
    assert metrics == report["test_metrics"]
    assert baseline_metrics == report["baseline_test_metrics"]
    assert model["n_training_rows"] == len(train + validation)


def test_refitting_is_deterministic():
    first, second = train_parser(tiny_rows(), seed=42), train_parser(tiny_rows(), seed=42)
    np.testing.assert_array_equal(first["name_classifier"].coef_, second["name_classifier"].coef_)
    assert parse_query(first, "fire cards") == parse_query(second, "fire cards")


def test_save_reload_preserves_batch_predictions(model, tmp_path):
    path = tmp_path / "nested/model.joblib"
    queries = [case["query"] for case in demo_cases() if "expected" in case]
    before = parse_queries(model, queries)
    save_parser(model, str(path))
    assert parse_queries(load_parser(str(path)), queries) == before


def test_invalid_or_incompatible_artifacts_are_rejected(model, tmp_path):
    path = tmp_path / "bad.joblib"
    for bad in [{}, {**model, "sklearn_version": "0.0.0"}, {**model, "numeric_rules_version": "v999"}]:
        joblib.dump(bad, path)
        with pytest.raises(ValueError):
            load_parser(str(path))


@pytest.mark.parametrize("query", [None, 10, True, "", "   ", "x" * 501])
def test_invalid_query_raises_clear_error(model, query):
    with pytest.raises(InvalidQueryError):
        parse_query(model, query)


@pytest.mark.parametrize("query,expected", [
    ("at least 150 hp", {"hp_min": 150}), ("over 150 hp", {"hp_min": 151}),
    ("at most 150 hp", {"hp_max": 150}), ("under 150 hp", {"hp_max": 149}),
    ("between 60 and 120 hp", {"hp_min": 60, "hp_max": 120}),
    ("after 2020", {"year_min": 2021}), ("before 2020", {"year_max": 2019}),
    ("from 2020", {"year_min": 2020, "year_max": 2020}),
    ("between 1999 and 2023", {"year_min": 1999, "year_max": 2023}),
    ("AT LEAST 0 HP from 0001", {"hp_min": 0, "year_min": 1, "year_max": 1}),
    ("at least 100 hp and at least 150 hp and under 250 hp", {"hp_min": 150, "hp_max": 249}),
    ("after 1999 before 2024", {"year_min": 2000, "year_max": 2023}),
    ("between 50 and 100 hp between 1999 and 2023", {"hp_min": 50, "hp_max": 100, "year_min": 1999, "year_max": 2023}),
    ("over 173 hp", {"hp_min": 174}),
    ("plain cards", {}),
])
def test_numeric_rules(query, expected):
    assert parse_numeric(query) == {**dict.fromkeys(["hp_min", "hp_max", "year_min", "year_max"]), **expected}


@pytest.mark.parametrize("query", [
    "under 0 hp", "at least -1 hp", "over 2.5 hp", "between 100 and 50 hp",
    "at least 200 hp and under 100 hp", "after 9999", "before 0001", "from 0000",
    "from 10000", "after -2020", "from 2020.5", "between 2024 and 1999",
    "after 2020 before 2000", "between -30 and 100 hp", "from 20",
])
def test_invalid_recognized_constraints_raise(query):
    with pytest.raises(InvalidQueryError):
        parse_numeric(query)


@pytest.mark.parametrize("updates", [
    {"hp_min": True}, {"hp_min": "150"}, {"hp_min": 1.0}, {"hp_max": -1},
    {"year_min": 0}, {"year_max": 10000}, {"types": None}, {"types": [""]},
    {"name": ""}, {"rarity": 8}, {"hp_min": 150, "hp_max": 100}, {"extra": 1},
])
def test_invalid_filter_labels_are_rejected(updates):
    with pytest.raises(ValueError):
        validate_filters({**empty_filters(), **updates})


def test_missing_filter_keys_are_rejected():
    with pytest.raises(ValueError):
        validate_filters({"name": None})


def test_type_canonicalization_trims_deduplicates_and_sorts():
    assert validate_filters({**empty_filters(), "types": [" Water ", "Fire", "fire"]})["types"] == ["Fire", "Water"]


def test_conflicting_duplicates_rejected_identical_ones_deduplicated():
    row = {"query": "  FIRE   cards ", "filters": {**empty_filters(), "types": ["Fire"]}}
    same = {"query": "fire cards", "filters": {**empty_filters(), "types": ["fire"]}}
    assert len(prepare_rows([row, same])) == 1
    with pytest.raises(ValueError, match="conflicting"):
        prepare_rows([row, {"query": "fire cards", "filters": empty_filters()}])


def test_csv_roundtrip_and_header_validation(tmp_path):
    path = tmp_path / "queries.csv"
    rows = tiny_rows()
    write_rows(path, rows)
    assert load_rows(path) == prepare_rows(rows)
    path.write_text("query,filters\nhello,{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        load_rows(path)


def test_pool_overlap_rejected():
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint(tiny_rows(), tiny_rows())


@pytest.mark.parametrize("C", [0, -1, True, "1", float("nan"), float("inf")])
def test_invalid_regularization_rejected(C):
    with pytest.raises(ValueError, match="C"):
        train_parser(tiny_rows(), C=C)


def test_missing_classifier_classes_raise_informative_errors():
    with pytest.raises(ValueError, match="name classifier"):
        train_parser([{"query": "cards", "filters": empty_filters()}])
    rows = tiny_rows()
    for row in rows:
        row["filters"]["rarity"] = None
    with pytest.raises(ValueError, match="rarity classifier"):
        train_parser(rows)
    rows = tiny_rows()
    for row in rows:
        row["filters"]["types"] = ["Fire"]
    with pytest.raises(ValueError, match="present and absent"):
        train_parser(rows)


def test_evaluation_slot_counts_and_empty_slots():
    correct = {**empty_filters(), "name": "Charizard", "types": ["Fire"], "hp_min": 150}
    wrong = {**correct, "name": "Pikachu"}
    assert score_predictions([wrong, empty_filters()], [correct, empty_filters()]) == {"exact_match": 0.5, "micro_f1": 2 / 3}
    assert score_predictions([empty_filters()], [empty_filters()]) == {"exact_match": 1.0, "micro_f1": 1.0}
    with pytest.raises(ValueError):
        score_predictions([], [])


def test_majority_baseline_tie_is_lexicographic():
    labels = [row["filters"] for row in tiny_rows()]
    rows = [{"query": str(i), "filters": label} for i, label in enumerate(labels)]
    expected = min(json.dumps(label, sort_keys=True) for label in labels)
    assert json.dumps(majority_label(rows), sort_keys=True) == expected


def test_demo_preview_has_expected_golden_card_and_null_handling(model):
    cards = [
        {"card_id": "demo-003", "name": "Charizard", "types": ["Fire"], "hp": 120, "rarity": "Rare", "set_name": "Demo C", "release_year": 2016, "image_url": None},
        {"card_id": "demo-001", "name": "Charizard", "types": ["Fire"], "hp": 180, "rarity": "Rare", "set_name": "Demo A", "release_year": 2020, "image_url": None},
        {"card_id": "missing", "name": "Charizard", "types": ["Fire"], "hp": None, "rarity": None, "set_name": "Demo", "release_year": None, "image_url": None},
    ]
    result = preview_cards(cards, parse_query(model, demo_cases()[0]["query"]))
    assert result["total"] == 1 and result["cards"][0]["card_id"] == "demo-001"
    assert preview_cards(cards, empty_filters())["total"] == 3
