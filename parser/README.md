# Part 2 - Learned query parser

One shared **TF-IDF** vectorizer (`lowercase=True, ngram_range=(1, 2)`) feeds:

- one logistic-regression classifier for the canonical card name or `__NONE__`;
- one classifier for canonical rarity or `__NONE__`;
- one independent binary classifier per type, included at `P(1) >= 0.5`.

Explicit versioned rules extract HP and year bounds. There is no pretrained
model, external inference service, API key, or hard-coded prediction for the
golden query. Classifiers use `LogisticRegression(C=C, max_iter=1000,
random_state=seed)`.

## Setup and first run

Run from the repository root using Python 3.11+:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r parser/requirements-dev.txt
python -m parser.demo_server
```

Open **http://127.0.0.1:8765**. A real saved model handles every request. The page
shows interpreted filters, raw JSON, model inference time, saved evaluation
scores, and up to eight matching catalog records. It includes eight independently
specified sanity checks, loading/error states, and the supported vocabulary.
All query and card text is rendered as text, not HTML. It does not fetch card
images automatically; the image links open the catalog's original URLs.

`parser/demo/index.html` is a diagnostic HTML client. It needs
`parser/demo_server.py` to execute Python; double-clicking the HTML alone is not
enough. The server binds to **127.0.0.1**, requires no web framework, and is for
local testing. Stop with Ctrl+C. To change the port or artifact:

```bash
python -m parser.demo_server --port 8766 --model artifacts/parser/model.joblib
```

The test harness uses `/demo/info` and `/demo/parse`; it does not occupy or
implement the agreed Part 3 production endpoints. Its small catalog preview is
diagnostic and can be discarded when the production search/API/UI arrives.

For a library-only install, omit test dependencies:

```bash
python -m pip install .
```

The joblib artifact and CSV data remain explicit file handoffs in the repository;
they are not downloaded automatically or embedded in the package wheel.

## Public API / handoff to Parts 3 and 5

```python
from parser import (
    train_parser, parse_query, save_parser, load_parser, evaluate_parser,
)
from parser.data import load_rows

model = load_parser("artifacts/parser/model.joblib")
filters = parse_query(model, "fire charizards with at least 150 hp")
assert filters == {
    "name": "Charizard", "types": ["Fire"],
    "hp_min": 150, "hp_max": None,
    "rarity": None, "year_min": None, "year_max": None,
}

# Rows in memory contain `filters`, not `filters_json`.
rows = load_rows("data/query_examples.csv")
model = train_parser(rows, seed=42, C=10.0)
save_parser(model, "artifacts/parser/experimental.joblib")
reloaded = load_parser("artifacts/parser/experimental.joblib")
assert parse_query(reloaded, "fire cards") == parse_query(model, "fire cards")
```

`evaluate_parser(model, labeled_rows)` returns **exactly** `exact_match` and
`micro_f1`. Use the split below for measured performance rather than evaluating
on rows used in the example training call. Part 5 can forward these five
functions from `training/adapter.py` and package the same joblib bundle in MLflow.

Part 3 can bind the loaded model once:

```python
parse = lambda query: parse_query(model, query)
```

Filters always have exactly seven keys. Scalars are strings, Python integers, or
`None`; `types` is a sorted list, never `None`. Missing output values are not
inferred from catalog metadata. Empty text, non-string text, queries longer than
500 trimmed characters, and invalid recognized numeric constraints raise
`InvalidQueryError`, a `ValueError` subclass.

**Shared scaffold handoff:** `shared/contracts.py` is not in this checkout.
`parser/contracts.py` uses its `InvalidQueryError` when it becomes available,
and otherwise defines a local compatible subclass. It validates the agreed
seven-key format locally without creating integrator-owned files. The integrator
should add its shared module before wiring the API's 422 handler; load failures
and unexpected prediction failures should map to the API's 503 response.

## Supported catalog vocabulary and text

The committed CSV supplies **all 250 distinct names, all 13 rarities, and all 11
types** in Part 1's `data/cards.csv`, including variants such as `Charizard ex`,
`Charizard-EX`, and `M Charizard-EX`. Labels are the catalog's canonical spelling.
The HTML page and `model["vocabulary"]` expose the full list. Only Pokémon cards
are included, matching Part 1's choice.

Preprocessing lowercases and collapses whitespace at both training and inference.
Before default word tokenization it replaces gender symbols with `female`/`male`,
`-EX` with `hyphenex`, and standalone `M` with `mega`. This preserves distinctions
the default TF-IDF tokenizer would lose; it does not look up names or override
predictions. Plain-name plurals such as `charizards` are represented in training.

Examples:

```text
fire charizards with at least 150 hp
show me Pikachu cards
water or fire cards
Rare Holo Charizard cards
M Charizard-EX cards
Nidoran ♀ cards
cards with between 60 and 120 hp
cards between 1999 and 2016
```

Types use OR within the list. Active filter fields use AND in the diagnostic
preview. A learned name may still produce zero results with impossible HP,
rarity, or year combinations; correct filtering does not guarantee a match.

## Numeric grammar, version v1

Matching is case-insensitive. Use digits and a space before `hp`.

| Phrase | Inclusive output |
| --- | --- |
| `at least N hp` | `hp_min=N` |
| `over N hp` | `hp_min=N+1` |
| `at most N hp` | `hp_max=N` |
| `under N hp` | `hp_max=N-1` |
| `between N and M hp` | `hp_min=N, hp_max=M` |
| `after YYYY` | `year_min=YYYY+1` |
| `before YYYY` | `year_max=YYYY-1` |
| `from YYYY` | both year bounds equal `YYYY` |
| `between YYYY and YYYY` without `hp` | both year bounds, inclusive |

Repeated constraints intersect: `at least 100 hp and at least 150 hp` yields
`hp_min=150`. Reversed ranges, empty intersections, negative/decimal HP, invalid
year width, years outside 1..9999, `under 0 hp`, `after 9999`, and `before 0001`
raise `InvalidQueryError`. Numbers not seen in training still work with supported
grammar. A phrase with no supported numeric constraint leaves those bounds null.

Out of scope: spelled-out numbers, `150+ hp`, `hp >= 150`, `released in 2020`,
negation, more than one named Pokémon, unknown Pokémon, set-name filters, and
arbitrary paraphrases. The model can misclassify unfamiliar wording into an
existing category; it does not have a calibrated unknown-name/rejection model.

## Data, training, and evaluation

Each CSV has exactly `query,filters_json` as its columns:

- `data/query_examples.csv`: **19,196** distinct synthetic labeled queries.
- `data/unseen_normal.csv`: **160** separately generated, labeled normal queries.
- `data/unseen_drift.csv`: **160** labeled queries with deliberately longer text
  and more tokens for Part 4's traffic generator / Part 6's drift evaluation.

All canonical categorical values come from the committed catalog. Name/type/rarity
combinations for named queries use real card records. Category presence is varied
independently, and type-only, rarity-only, empty, and multi-type cases are added.
Numeric labels come from explicit generation choices, **never predictions or the
numeric parser**. Numeric combinations can intentionally match no card. The
normal/drift pools are mutually disjoint and absent from the complete modeling
CSV, not just from one split.

The source recipe is committed in `parser/build_examples.py`:

```bash
# Optional: regenerate the three CSVs from the committed catalog, seed 42
python -m parser.build_examples

# Compare three settings, select on validation, refit, then test and save
python -m parser.train
```

The split follows the team rules: normalize query text; deduplicate equal
labels; reject conflicting duplicate labels; sort by normalized query; shuffle
with `random.Random(42)`. First `ceil(0.2*N)` rows are final test, the next same
size are validation, and the remainder are training. This produces:

| Split | Rows |
| --- | ---: |
| Training (candidate fits) | 11,516 |
| Validation (selection only) | 3,840 |
| Final test | 3,840 |
| Train + validation (saved model fit) | 15,356 |

Candidates use C=0.1, 1.0, 10.0. Selection sorts by validation exact match, then
micro-F1, then smaller C. The selected C=10 model is refitted on train+validation;
final test is scored after selection, without using it to tune the model.
Verification tests recompute the frozen artifact's scores, without changing it.

The saved model scored **99.974% test exact match** and **99.997% slot micro-F1**.
The complete-label majority baseline scored **0.026% exact match**. The baseline
uses the most frequent complete label from fitted rows and breaks ties by
lexicographic sorted-key JSON. Micro-F1 counts each non-null scalar and each type
as a separate `(field, value)` slot, sums TP/FP/FN across queries, and returns 1.0
when the denominator is zero.

**Interpretation:** these are synthetic, template-generated examples; random
splits share templates. The high scores mainly show consistent learning of the
generated vocabulary and grammar. They do not establish broad language accuracy,
robustness to unknown names, or performance under drift. Collect independently
human-written labels to evaluate those claims. The unseen traffic pools are
reserved for teammates and are not used to select the shipped model.

`artifacts/parser/evaluation.json` records candidate scores, test/baseline scores,
split sizes, vocabulary, dependency versions, data/model hashes, and normalized
fitted-query hashes. `artifacts/parser/model.joblib` contains the fitted
vectorizer, all classifiers, vocabulary, seed/C, parser kind, and rule/preprocessing
versions. Keep source code and pinned dependencies alongside it. Only load trusted
joblib artifacts, since joblib uses Python pickle internally.

## Validation

```bash
# All offline catalog + parser checks; local HTTP tests skipped by default
python -m pytest -q

# Include real local HTTP integration tests (no external services needed)
RUN_LOCAL_DEMO_TESTS=1 python -m pytest -q

# One-line smoke test
python -c 'from parser import load_parser, parse_query; print(parse_query(load_parser("artifacts/parser/model.joblib"), "fire charizards with at least 150 hp"))'
```

Tests cover label schema and invalid values, duplicate handling, exact split
membership, preprocessing, fitting vocabulary on training text only, independent
type models, deterministic fitting, golden and variant queries, numeric boundary
and contradiction cases, metric arithmetic, majority baseline comparison,
serialization round trips, and the diagnostic HTTP routes.

Dependencies are pinned in `parser/requirements.txt` (runtime) and
`parser/requirements-dev.txt` (tests); `pyproject.toml` makes the library installable.
The shipped artifact was trained and verified with Python 3.14.0 and scikit-learn
1.8.0. Other Python versions supported by the pins can retrain with the same
command; loading with a different scikit-learn version fails clearly.
