"""Verify a running Compose stack, including normal and drift DB snapshots."""

import json
import subprocess
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def compose(*args):
    subprocess.run(["docker", "compose", *args], check=True)


def main():
    query = "fire charizards with at least 150 hp"
    with urlopen("http://localhost:8000/ready", timeout=10) as response:
        assert response.status == 200
    request = Request("http://localhost:8000/search",
                      data=json.dumps({"query": query}).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=15) as response:
        result = json.load(response)
    assert result["filters"]["name"] == "Charizard" and result["total"] > 0
    assert result["model_version"].isdigit()
    invalid = Request("http://localhost:8000/parse",
                      data=json.dumps({"query": "under 0 hp"}).encode(),
                      headers={"Content-Type": "application/json"})
    try:
        urlopen(invalid, timeout=15)
    except HTTPError as error:
        assert error.code == 422, error.read().decode()
    else:
        raise AssertionError("Contradictory numeric queries must return HTTP 422")
    with urlopen("http://localhost:5000/?" + urlencode({"q": query}), timeout=15) as response:
        assert "Charizard" in response.read().decode()
    with urlopen("http://localhost:8000/metrics", timeout=10) as response:
        assert "api_request_duration_seconds_bucket" in response.read().decode()
    cases = json.loads((Path(__file__).resolve().parents[1] / "tests/fixtures/natural_queries.json").read_text(encoding="utf-8"))
    matched = 0
    for case in cases:
        request = Request("http://localhost:8000/parse", data=json.dumps({"query": case["query"]}).encode(),
                          headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=15) as response:
                matched += json.load(response)["filters"] == case["filters"]
        except HTTPError:
            pass
    assert matched / len(cases) >= 0.9, f"Natural-language regression: {matched}/{len(cases)}"
    print(f"Natural-language regression: {matched}/{len(cases)}")
    for mode in ("normal", "drift"):
        compose("run", "--rm", "--no-deps", "simulator", "--mode", mode, "--count", "100")
        compose("run", "--rm", "--no-deps", "drift-worker", "python", "-m", "monitoring.drift",
                "--reference", "data/reference_simulator.jsonl", "--once")
        check = f'''
import os
from monitoring.persistence import _connect
with _connect(os.environ['DATABASE_URL']) as conn:
    row = conn.execute('SELECT drift_detected, model_metrics, evidently_report FROM drift_runs ORDER BY window_end DESC LIMIT 1').fetchone()
    assert row is not None
    assert row[0] == {mode == 'drift'!r}, row
    assert row[1]['n_requests'] == 100 and row[1]['label_coverage'] == 1.0, row[1]
    assert row[2]['drift']['engine'] == row[2]['performance']['engine'] == 'evidently'
    print('Verified {mode}:', row[0], row[1])
'''
        compose("exec", "-T", "drift-worker", "python", "-c", check)
    print("Stack smoke test passed: registry, API, frontend, PostgreSQL, Evidently, and drift.")


if __name__ == "__main__":
    main()
