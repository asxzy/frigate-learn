"""Webapp tests (require the `web` extra: fastapi + uvicorn)."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from frigate_learn.config import build_config
from frigate_learn.webapp.app import create_app


def make_client(tmp_path) -> TestClient:
    cfg = build_config({}, tmp_path)
    return TestClient(create_app(cfg))


def test_health(tmp_path):
    c = make_client(tmp_path)
    r = c.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_serves_spa(tmp_path):
    c = make_client(tmp_path)
    r = c.get("/")
    assert r.status_code == 200
    assert "frigate-learn" in r.text


def test_app_uses_default_config_without_arg(tmp_path, monkeypatch):
    monkeypatch.setenv("FRIGATE_LEARN_CONFIG", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("data:\n  root: data\n", encoding="utf-8")
    client = TestClient(create_app())
    assert client.get("/api/health").status_code == 200