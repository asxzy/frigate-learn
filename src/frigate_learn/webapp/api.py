"""Webapp API routes."""

from __future__ import annotations

from fastapi import FastAPI


def register_routes(app: FastAPI) -> None:
    """Attach the API routers to the app."""

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}