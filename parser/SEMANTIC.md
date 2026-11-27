# Pretrained semantic parser

The upgraded parser combines a frozen MiniLM sentence encoder with supervised
TF-IDF/logistic-regression heads, catalog-grounded names, and explicit numerical
constraints. The model kind is `minilm-tfidf-logreg-hybrid-v2`; the original
`tfidf-logreg-hybrid-v1` artifact remains loadable for comparison and rollback.

## Pretrained representation

The encoder is [Sentence Transformers' all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
using [Xenova's quantized ONNX export](https://huggingface.co/Xenova/all-MiniLM-L6-v2).
It produces 384-dimensional, normalized sentence vectors. `parser/encoder.py`
pins revision `cb3d680149bf9a3209564e1b27ab3bb355b65707`, verifies SHA-256 hashes
for both the tokenizer and model graph, and runs ONNX Runtime on the CPU. The
upstream models use the Apache-2.0 license.

The encoder is frozen. We train logistic-regression heads on combined TF-IDF and
pretrained embedding features for types and rarity. Pokémon names are masked in
those features, so requesting a Charizard does not imply requesting Fire cards.
Name resolution is grounded in the catalog, preserves EX/Mega/gender variants,
and allows conservative spelling corrections with a minimum similarity and gap.
Unrecognized searches are rejected by the API instead of becoming default names.

Numeric comparisons use rules, because similar sentence embeddings alone do not
guarantee correct inequality semantics. Examples:

| Input | Inclusive filter |
| --- | --- |
| `more than 150 hp`, `HP > 150` | minimum 151 |
| `at least 150 hit points`, `HP >= 150` | minimum 150 |
| `less than 150 hp`, `HP < 150` | maximum 149 |
| `no more than 150 hp`, `HP <= 150` | maximum 150 |
| `150+ hp` | minimum 150 |
| `between 120 and 180 health points` | 120 through 180 |
| `released since 2016` | year 2016 or later |

## Training and serving

`python -m parser.build_semantic_examples` reproduces 23,397 synthetic labeled
queries. Labels come from recipe choices, never predictions. The original 19,196
examples and both unseen traffic pools are preserved. The semantic training CSV
is disjoint from the normal/drift pools. Additional templates cover natural
comparisons and type descriptions such as electric, plant, and flame abilities.

The team's Prefect flow still compares C=0.1/1/10 on validation, refits the selected
setting, evaluates the final test once, checks the majority baseline, and registers
the model in MLflow. Set `CARDQUERY_PARSER_KIND=semantic` and
`TRAINING_DATA_PATH=data/query_examples_semantic.csv`; root Compose supplies both.
The only workflow additions package the semantic dependencies and log encoder
revision metadata. The registered artifact contains the encoder weights and
tokenizer alongside the trained heads; prediction needs no model download,
external inference service, API key, GPU, or separately hosted language model.
The first training run needs network access to fetch the pinned public encoder;
the registered artifact then supports inference without that download.

## Evaluation

`tests/fixtures/natural_queries.json` contains 40 separately authored labeled
examples, not used to fit models or choose C. The prior deployed API scored
13/40 exact matches; the deployed version 4 scores 40/40, with all seven
unknown-name probes rejected. The synthetic final test scores 99.295% exact
match and 99.897% micro-F1 on the expanded dataset. The complete per-query
results and model provenance are in
[`artifacts/parser/semantic_evaluation.json`](../artifacts/parser/semantic_evaluation.json).
`scripts/evaluate_natural_queries.py` evaluates a candidate
manifest without activating it. The deployment gate requires at least 90% on
this small set, correct strict HP bounds for the reported query, and rejection
of the unknown-name probes. Scores on these checks do not establish arbitrary
natural-language accuracy.

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5001"
python -m scripts.evaluate_natural_queries --manifest artifacts/active_model.json
```

`tests/test_numeric_v2.py` runs offline. Set `CARDQUERY_SEMANTIC_MANIFEST` to a
registered candidate manifest to enable the actual model/asset integration tests.
For native Windows training, use `python -X utf8` or `PYTHONIOENCODING=utf-8` so
MLflow's console links can be printed. Docker uses UTF-8 by default.
