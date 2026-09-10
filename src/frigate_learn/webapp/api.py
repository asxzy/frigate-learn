from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from ..inspect.triage import set_sample_quality, validate_quality
from . import queries
from .jobs import JobManager, JobRunningError
from .serving import resolve_image, resolve_thumb


class JobRunRequest(BaseModel):
    steps: list[str]
    dry_run: bool = Field(default=False, strict=True)


class QualityRequest(BaseModel):
    quality: str


def register_routes(app: FastAPI) -> None:
    config = app.state.config
    db = app.state.db
    jobs: JobManager = app.state.jobs

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/overview")
    def overview() -> dict:
        return queries.overview(config, db)

    @app.get("/api/benchmark")
    def benchmark() -> dict:
        return queries.benchmark(config)

    @app.get("/api/deployments")
    def deployments() -> dict:
        return queries.deployments(db)

    @app.get("/api/quality")
    def quality() -> dict:
        return queries.quality_counts(db)

    @app.get("/api/datasets")
    def datasets() -> dict:
        return queries.datasets(config)

    @app.get("/api/training/list")
    def training_list() -> dict:
        return queries.training_index(config)

    @app.get("/api/training/{run}", response_model=None)
    def training_run(run: str):
        result = queries.training_run(config, run)
        if result is None:
            return JSONResponse({"detail": "unknown training run"}, status_code=404)
        return result

    @app.get("/api/samples")
    def samples(
        status: str | None = None,
        quality: str | None = None,
        camera: str | None = None,
        verified: int | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        return queries.samples(
            db, status=status, quality=quality, camera=camera,
            verified=verified, limit=limit, offset=offset,
        )

    @app.get("/api/samples/{sample_id}", response_model=None)
    def sample_detail(sample_id: str):
        result = queries.sample_detail(db, sample_id)
        if result is None:
            return JSONResponse({"detail": "unknown sample"}, status_code=404)
        return result

    @app.get("/api/jobs")
    def jobs_recent(limit: int = 20) -> list[dict]:
        return jobs.recent(limit=limit)

    @app.get("/api/jobs/{job_id}", response_model=None)
    def jobs_get(job_id: str):
        result = jobs.get(job_id)
        if result is None:
            return JSONResponse({"detail": "unknown job"}, status_code=404)
        return result

    @app.post("/api/jobs/run")
    async def jobs_run(body: JobRunRequest):
        try:
            job_id = jobs.start(body.steps, dry_run=body.dry_run)
        except JobRunningError:
            return JSONResponse(
                {"detail": "a pipeline job is already running"}, status_code=409,
            )
        except ValueError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=422)
        return JSONResponse({"job_id": job_id}, status_code=202)

    @app.post("/api/samples/{sample_id}/quality")
    async def sample_quality(sample_id: str, body: QualityRequest):
        try:
            validate_quality(body.quality)
        except (ValueError, TypeError) as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)
        updated = set_sample_quality(config, db, sample_id, body.quality)
        if not updated:
            return JSONResponse({"detail": "unknown sample"}, status_code=404)
        return queries.sample_detail(db, sample_id)

    @app.get("/images/{sample_id}", response_model=None)
    def serve_image(sample_id: str):
        path = resolve_image(config, db, sample_id)
        if path is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        data = path.read_bytes()
        return Response(content=data, media_type="image/jpeg",
                        headers={"Content-Length": str(len(data))})

    @app.get("/images/{sample_id}/thumb", response_model=None)
    def serve_thumb(sample_id: str):
        path = resolve_thumb(config, db, sample_id)
        if path is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        data = path.read_bytes()
        return Response(content=data, media_type="image/jpeg",
                        headers={"Content-Length": str(len(data))})
