# Frigate Learn — agent guide

Active-learning / auto-training pipeline for a Frigate NVR (0.18.x) deployment.
Collects real camera data over Frigate's HTTP API, verifies detections with a
VLM, builds frozen YOLO dataset versions, fine-tunes small detector candidates
(`ml` extra), benchmarks/gates them against an immutable golden dataset, and
deploys winners to Hailo-8. A local FastAPI dashboard (`web` extra) is the
control panel.

## What is implemented

- **Pipeline phases (P0-P13):** golden dataset, Frigate 0.18 collector →
  SQLite, quality triage, perceptual-hash dedup, VLM verification, versioned
  dataset builder (deterministic splits + manifest), candidate benchmark on the
  golden dataset, YOLO fine-tuning, VLM teacher, missed-object discovery, COCO
  external import, deployment gate (vs. baseline, `deployments` ledger), Hailo
  deploy (ONNX → HEF), and the `frigate-learn run` orchestrator.
- **Webapp (`frigate-learn web`):** FastAPI backend (`webapp/{app,api,queries,
  serving,jobs}.py`) + no-build vanilla-JS SPA (`webapp/static/`). 22 endpoints
  under `/api/*`, seven views (Overview, Benchmark,
  Quality, Audit, Datasets, Training, Triage lightbox), pipeline stages run
  in a background daemon thread (`jobs.py` JobManager, single job at a time,
  `jobs` SQLite ledger, bounded log tail). Localhost-only, no auth/CORS.

## Commands

```bash
.venv/bin/pip install -e ".[dev]"        # dev: pytest, respx, ruff etc.
.venv/bin/pip install -e ".[dev,web]"    # + fastapi/uvicorn (dashboard)
.venv/bin/pip install -e ".[ml]"         # + torch/ultralytics (train/benchmark/verify)
.venv/bin/frigate-learn --config config.yaml db init   # SQLite + migrations
.venv/bin/frigate-learn web --host 127.0.0.1 --port 8080
.venv/bin/python -m pytest               # full suite (444 tests)
.venv/bin/python -m pytest -p no:warnings 2>&1 | tail -1   # pass count (warnings hide the summary line)
.venv/bin/python -m pytest tests/test_webapp.py -v
```

## Conventions

- **No comments in code.** The repo generically uses `# ---` section banners,
  but the webapp directive is strict: zero comments in Python/JS/HTML/CSS.
  Do not add comments anywhere in `webapp/` or `static/`.
- **CLI is click** (`cli.py`), subcommands: collect, inspect, triage, verify,
  discover, dataset, import-coco, benchmark, train, gate, deploy, run, status,
  db, web, audit (`audit-dataset` is an alias for `audit run`; dataset audit /
  pseudo-labeling, see `audit/` and `docs/audit-pipeline.md`).
- **SQLite via SQLAlchemy**, WAL mode; schema migrations run through
  `frigate-learn db init` (see `db.py`, `models.py`). Tests build fresh DBs in
  `tmp_path` and dispose sessions.
- **Webapp optionality:** endpoints/CLI that need FastAPI are guarded by
  `pytest.importorskip("fastapi")` at module top of `tests/test_webapp.py`, and
  `frigate_learn.webapp` imports happen lazily inside `create_app`/`web`
  command. Optional-heavy stages (`train`, `benchmark`, `deploy`, `verify`)
  also lazy-import and raise clear `RuntimeError` hints when extras are missing.
- **Frigate specifics live only in `frigate/`** — URLs/response shapes nowhere
  else. Secrets come from env / `.env`; the webapp never reads `.env` or the
  VLM key.
- **Pipeline stages** are `collect → verify → build → train → benchmark → gate
  → deploy`, each reporting `executed|skipped|failed` with a reason. Stages
  honour `automation.enable`/`till`, `--steps`, `--until`, `--keep-going`,
  `--days`. `run_pipeline` in `run.py` is the single entry point the CLI and
  webapp jobs reuse.
- **Structured logging:** `logutil` `key=value` formatters (`info`, `warn`,
  `err`, `success`); the webapp JobManager captures them for the running-job
  log tail.
- **Never commit:** `.DS_Store`, `yolov8*.pt`, `yolov9*.pt`, `yolo11*.pt`, or
  any `data/` contents (gitignored).