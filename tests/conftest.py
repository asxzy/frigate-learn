"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from frigate_learn.config import AppConfig
from frigate_learn.db import Database


@pytest.fixture
def config(tmp_path) -> AppConfig:
    cfg = AppConfig(base_dir=tmp_path)
    cfg.frigate.base_url = "http://frigate:8971"
    cfg.data.root = "data"
    cfg.data.images = "images"
    cfg.data.database = "frigate_learn.db"
    cfg.collection.concurrency = 2
    return cfg


@pytest.fixture
def db(config: AppConfig) -> Database:
    database = Database(config.database_path())
    database.init()
    yield database
    database.dispose()


def make_config_file(path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")