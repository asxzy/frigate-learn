# Webapp: pipeline control panel & visualization dashboard

Date: 2026-09-09

## Goal

A local webapp for the `frigate-learn` pipeline that supports the operator's full
workflow: **view** pipeline health and decisions, **trigger** stages, **triage**
samples, and **approve/deny** gate outcomes. The four decision views, in
priority order: benchmark/gate winner, data quality & verify backlog, dataset/
golden coverage, and training curves.

## Constraints

- Runs on the dev machine only (this Mac), single user, bound to `127.0.0.1`.
- No auth, no CORS (same-origin SPA).
- Pure-Python repo: the only new dependency is `fastapi` (+ `uvicorn`). Charts
  come from a vendored `uPlot` bundle; no CDN, no Node toolchain.
- The webapp reuses the existing library directly — no subprocess-to-CLI. The
  same code paths the CLI drives (`run_pipeline`, `Trainer`, `DatasetBuilder`,
  `triage`, `record_deployment`) are called in-process.
- One `uvicorn` worker. Pipeline stages run in a background thread so long
  stages (verify) never block the event loop.

## Stack

- Backend: FastAPI + SQLAlchemy (existing). New: `src/frigate_learn/webapp/`.
- Frontend: no-build vanilla JS SPA served as static files from FastAPI, with a
  vendored `uPlot.bundle.js`.
- Entry point: `frigate-learn web` (new CLI command).

## Architecture

```
src/frigate_learn/webapp/
  app.py            # FastAPI factory: mounts static SPA + API routes (single uvicorn worker)
  api.py            # JSON endpoints: reads (dashboard queries) + writes (actions)
  jobs.py           # background job runner: runs stage functions in a thread, records to `jobs`
  queries.py        # SQLAlchemy/SQLite aggregations used by the dashboard
  serving.py        # image serving: original snapshot + thumbnail endpoints
  static/           # SPA: index.html, app.js, styles.css, uPlot.bundle.js (vendored)
tests/test_webapp.py
```

Action model: `POST` endpoints enqueue work on a background thread via
`jobs.py`. Reads hit `queries.py` / filesystem artifacts directly.

## Views

### 1. Overview / Pipeline
- Pipeline stage strip (collect → verify → build → train → benchmark → gate →
  deploy) with per-stage status.
- Recent `jobs` ledger.
- Disk-space warning.
- Per-stage "run" buttons.

### 2. Benchmark & Gate
- **Candidate table** from `benchmark-results.json` / `deployments`: mAP50,
  recall, latency, small-object recall, PASS/FAIL per rule, baseline.
- **Pareto scatter**: x = latency (ms), y = mAP50, bubble = params. Baseline +
  candidates; PASS region shaded.
- **Gate history**: `deployments` ledger timeline (verdict, reasons, deployed,
  timestamp).
- **Deploy** button → `deploy.hailo.deploy` (dry-run by default).

### 3. Data Quality & Verify Backlog
- KPI cards: total, verified, unverified, degenerate-box count, VLM failures,
  collection failures.
- Bar charts: per camera, per label, quality verdicts, verified-vs-unverified.
- Verify backlog list (unverified samples, filterable, paged); collection
  failures shown with reasons.

### 4. Dataset & Golden Coverage
- Dataset versions from `build.json`: totals, splits, per-class counts.
- Golden dataset stats: image count, per-class box counts.
- Pool-to-next-build readiness.

### 5. Training Curves
- Each `data/training/<model>-<dataset>/results.csv` as line charts (loss,
  mAP50 over epochs).

### 6. Triage / Image Review
- Filters (camera/label/quality) → thumbnail grid → lightbox with image + SVG
  overlays (Frigate vs VLM boxes, toggleable) → quality buttons (useful/bad/
  duplicate/ignore) reusing `triage.py`.

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/overview` | pipeline status, per-stage result, jobs ledger, disk usage |
| `GET /api/benchmark` | parsed results + `deployments` ledger |
| `GET /api/quality` | aggregations + verify backlog (filtered, paged) |
| `GET /api/datasets` | dataset versions, golden stats |
| `GET /api/training/list` | training run dirs |
| `GET /api/training/{run}` | `results.csv` lines |
| `GET /api/samples?filter=...` | sample list for triage (paged) |
| `GET /api/samples/{id}` | one sample + annotations |
| `GET /images/{sample_id}` | original snapshot bytes |
| `GET /images/{id}/thumb` | small JPEG |
| `POST /api/jobs/run` | `{steps, dry_run}` → enqueue pipeline |
| `POST /api/samples/{id}/quality` | `{quality}` → `triage.set_sample_quality` |
| `GET /api/jobs/{id}` | live status + log tail (polling) |

## Data flow

- Reads use the existing `Database`/`SQLAlchemy` layer. No schema drift.
- Filesystem artifacts (`benchmark-results.json`, `build.json`, `results.csv`,
  `golden/`) read in place — the webapp never caches or re-derives.
- Overlay geometry: boxes are normalized (0–1); browser SVG scales to the
  displayed image. Frigate boxes from `samples.frigate_*`, VLM from
  `annotations` (`verified=1`).

## Error handling

- Job validation: reject unknown steps; reject a second concurrent pipeline run
  (401) — matches cron semantics.
- On startup, mark any stale `running` job as failed ("terminated by server
  restart").
- Image 404s → JSON error, UI placeholder. Thumbnails generated lazily into
  `data/previews`.
- Job output: bounded log tail (~200 lines) in `jobs.metadata_json`.

## Testing

- `queries.py` aggregation unit tests on a temp SQLite with real migrations.
- FastAPI `TestClient` for read endpoints, triage write, job valid/reject,
  stale-run marking.
- Overlay normalization function test.
- Job runner tested with a fake stage function (no Frigate/torch in tests).
- All 202 existing tests must stay green.