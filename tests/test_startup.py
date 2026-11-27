"""Startup regressions: cached models and duplicate frontend listeners."""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from api import bootstrap


@pytest.fixture
def startup(monkeypatch, tmp_path):
    path = tmp_path / "active_model.json"
    manifest = {"model_uri": "models:/pokemon-query-parser/4", "model_version": "4"}
    calls = []

    def train():
        calls.append("train")
        path.write_text(json.dumps(manifest))

    def load(target):
        calls.append("load")
        assert target == path and path.exists()
        return lambda query: {}, "4"

    monkeypatch.setattr(bootstrap, "ACTIVE_MODEL_PATH", path)
    monkeypatch.setattr(bootstrap, "train_model", train)
    monkeypatch.setattr(bootstrap, "load_registered_parser", load)
    return path, manifest, calls


def test_restart_reuses_registered_model_without_training(startup):
    path, manifest, calls = startup
    path.write_text(json.dumps(manifest))
    assert bootstrap.ensure_model() == manifest
    assert calls == ["load"]


def test_empty_installation_trains_and_validates_model(startup):
    _, manifest, calls = startup
    assert bootstrap.ensure_model() == manifest
    assert calls == ["train", "load"]


def test_explicit_retrain_does_not_reuse_old_version(startup):
    path, manifest, calls = startup
    path.write_text(json.dumps(manifest))
    assert bootstrap.ensure_model(retrain=True) == manifest
    assert calls == ["train", "load"]


def test_broken_cached_model_fails_visibly_instead_of_silently_training(startup, monkeypatch):
    path, manifest, calls = startup
    path.write_text(json.dumps(manifest))

    def fail(target):
        raise ValueError("registered model is missing")

    monkeypatch.setattr(bootstrap, "load_registered_parser", fail)
    with pytest.raises(ValueError, match="registered model is missing"):
        bootstrap.ensure_model()
    assert calls == []


def test_local_frontend_refuses_a_port_already_in_use():
    with socket.socket() as listener:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        result = subprocess.run([sys.executable, "-m", "frontend.app"],
                                cwd=Path(__file__).resolve().parents[1],
                                env={**os.environ, "PORT": str(listener.getsockname()[1])},
                                capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert "already in use" in result.stdout + result.stderr
