"""FastAPI application factory for the webapp dashboard."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from ..config import AppConfig, load_config
from ..db import Database
from . import api


def create_app(config: AppConfig | None = None) -> FastAPI:
    """Create the webapp: API routes plus the static SPA shell over a fresh DB."""
    if config is None:
        config = load_config()
    db = Database(config.database_path())
    db.init()

    app = FastAPI(title="frigate-learn", version="0.2.0")
    app.state.config = config
    app.state.db = db
    api.register_routes(app)
    static_dir = Path(__file__).parent / "static"
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    return app


__all__ = ["create_app"]