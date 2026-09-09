"""Config loading tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from frigate_learn.config import (
    AppConfig,
    FrigateSettings,
    build_config,
    load_config,
)
from conftest import make_config_file


def test_defaults():
    cfg = AppConfig()
    assert cfg.frigate.base_url == "http://frigate:8971"
    assert cfg.collection.severity == ["detection", "alert"]
    assert cfg.collection.concurrency == 8
    assert cfg.evaluation.golden_dataset == "golden-v001"
    assert len(cfg.classes) >= 8
    assert "person" in cfg.classes


def test_build_config_maps_sections():
    raw = {
        "classes": ["person", "car"],
        "frigate": {"base_url": "http://cam:8971/", "token": "t"},
        "collection": {"cameras": "front,gate", "severity": ["detection"], "concurrency": 3},
        "data": {"root": "var"},
    }
    cfg = build_config(raw, base_dir=Path("/tmp/x"))
    assert cfg.frigate.base_url == "http://cam:8971"  # trailing slash stripped
    assert cfg.frigate.token == "t"
    assert cfg.collection.cameras == ["front", "gate"]
    assert cfg.collection.severity == ["detection"]
    assert cfg.collection.concurrency == 3
    assert cfg.data.root == "var"
    assert cfg.base_dir == Path("/tmp/x").resolve()


def test_interpolation_and_anchoring(tmp_path, monkeypatch):
    body = """
frigate:
  base_url: "http://frigate:8971"
  token: "${FRIGATE_TOKEN:-fallback}"
  timeout_seconds: 12
collection:
  severity: ["detection"]
data:
  root: "data"
"""
    cfg_path = tmp_path / "config.yaml"
    make_config_file(cfg_path, body)
    monkeypatch.setenv("FRIGATE_TOKEN", "secret")
    cfg = load_config(cfg_path)
    assert cfg.frigate.token == "secret"
    assert cfg.frigate.timeout_seconds == 12.0
    assert cfg.collection.severity == ["detection"]
    # relative paths anchored to the config file's directory
    assert cfg.base_dir == tmp_path.resolve()
    assert cfg.database_path() == tmp_path.resolve() / "data" / "frigate_learn.db"


def test_env_token_default_when_unset(tmp_path):
    body = 'frigate:\n  token: "${FRIGATE_TOKEN:-fallback}"\n'
    cfg_path = tmp_path / "config.yaml"
    make_config_file(cfg_path, body)
    cfg = load_config(cfg_path, overrides={"FRIGATE_TOKEN": ""})
    assert cfg.frigate.token == "fallback"


def test_dotenv_is_used(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text('FRIGATE_TOKEN=fromdotenv\n', encoding="utf-8")
    cfg_path = tmp_path / "config.yaml"
    make_config_file(cfg_path, 'frigate:\n  token: "${FRIGATE_TOKEN}"\n')
    cfg = load_config(cfg_path, env_file=env_file)
    assert cfg.frigate.token == "fromdotenv"


def test_missing_config_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no config.example.yaml fallback here
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_frigate_defaults_dataclass():
    f = FrigateSettings()
    assert f.completed_events_only is True
    assert f.max_retries == 3