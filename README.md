# CardQuery

Search Pokémon cards with plain-language queries. This checkout contains the
Part 1 catalog and the Part 2 learned query parser.

## Try the model

From the repository root, with Python 3.11 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r parser/requirements-dev.txt
python -m parser.demo_server
```

Open **http://127.0.0.1:8765**. The saved model is included, so training is not
needed to try it. Enter a request, inspect the seven predicted filters, and
preview matching cards. The **Run checks** button tests eight known cases.
Press Ctrl+C in the terminal to stop.

The HTML file is `parser/demo/index.html`. Serve it with the command above:
opening it directly cannot run the Python model. This is a Part 2 development
tester; the team's production API and frontend remain separate components.

```bash
# Offline tests
python -m pytest -q

# Also run the local HTTP integration tests
RUN_LOCAL_DEMO_TESTS=1 python -m pytest -q

# Rebuild the parser, comparing C=0.1, 1, and 10
python -m parser.train
```

- [Parser documentation and integration contract](parser/README.md)
- [Catalog documentation and data source](catalog/README.md)

The parser learns only the catalog vocabulary. Training examples are synthetic;
high held-out scores on these templates do not establish accuracy on arbitrary
natural language. See the parser documentation for the supported numeric grammar.
