# Notes — Agent Context

This file exists so agents can quickly restore project context across sessions.
Last updated: 2026-09-15

## What This Project Is

Active-learning / auto-training pipeline for a Frigate NVR (0.18.x) deployment.
Collects real camera data over Frigate HTTP API, verifies detections with a VLM,
builds frozen YOLO dataset versions, fine-tunes small detector candidates,
benchmarks/gates against an immutable golden dataset, and deploys winners to Hailo-8.

**Core constraint:** Nothing CPU/GPU-heavy runs on the Frigate VM (Intel N150 + Hailo-8, ~20-30 cameras). Training, VLM, and benchmarking run on a separate machine that talks to Frigate only over HTTP.

## At a Glance

- **v0.2.0** | MIT | author: asxzy
- **363 tests passing** (7.8s)
- **48 Python source files, ~7900 LOC**
- **3 SQLite migrations**
- **16 FastAPI endpoints + 6-view vanilla-JS SPA**
- **7 pipeline stages:** collect → verify → build → train → benchmark → gate → deploy
- **Branch:** `master`, 29 commits ahead of origin

## Architecture

```
CLI (click: cli.py)
  ├── config.yaml → AppConfig dataclasses (config.py)
  ├── SQLite via SQLAlchemy 2.x, WAL mode (db.py, models.py)
  ├── Pipeline stages orchestrated by run.py
  ├── Webapp (FastAPI + vanilla-JS SPA)
  └── Data lives under data/{images,datasets,golden,training,models,previews}
```

### Pipeline Flow

```
Frigate VM (N150 + Hailo-8)
  ↕ HTTP (collector only)
Training Machine
  collect (reviews→events→snapshots→SQLite)
  verify  (VLM annotation via OpenAI-compatible endpoint)
  build   (frozen dataset version with deterministic splits)
  train   (YOLO fine-tune, before/after golden evaluation)
  benchmark (candidate evaluation + Pareto frontier)
  gate    (latency/accuracy envelope vs baseline)
  deploy  (ONNX export → Hailo HEF compile → manifest)
```

## Source Layout

```
src/frigate_learn/
├── __init__.py          (11)  v0.2.0, exports AppConfig/load_config
├── cli.py               (1006) All subcommands (db, collect, status, inspect, triage, verify, dataset, discover, import-coco, benchmark, train, gate, deploy, run, web)
├── config.py            (421) AppConfig + 9 settings dataclasses, YAML + env interpolation
├── db.py                (153) Database class, migration runner, schema_migrations table
├── models.py            (127) ORM: Sample, Annotation, Job, CollectionFailure, DiscoveryWindow, Deployment
├── logutil.py            (49) Structured key=value logging
├── times.py              (93) CLI time-arg parsing ("7 days ago", ISO, etc.)
├── run.py               (307) PIPELINE orchestrator, StepReport, next_build_version

├── collection/
│   ├── collector.py     (502) Collector: reviews→events→snapshots→Samples, ThreadPoolExecutor, dedup, region crop
│   ├── dedup.py          (59) dHash perceptual hashing + Hamming distance
│   ├── discover.py      (151) P9: motion windows without collected evidence
│   ├── external.py      (182) P10: COCO import → external samples
│   └── sampling.py       (45) P3: temporal sampling within events

├── annotation/
│   ├── schema.py        (170) VLMObject, VLMValidationError, JSON schema validation, extract_json_from_response
│   ├── verifier.py      (242) P4/P8: batches samples through VLMProvider, writes verified annotations, drops failures to quality=bad
│   └── vlm.py           (169) VLMProvider protocol + OpenAICompatibleProvider (base64 JPEG → chat completions)

├── dataset/
│   ├── builder.py       (256) P5: versioned YOLO dataset builder (deterministic splits, manifest, build.json)
│   ├── manifest.py       (68) JSONL manifest append-only by sample_id
│   ├── splits.py         (68) Deterministic split via SHA-256 hash of sample_id
│   └── yolo.py          (141) YOLO label format: parse, write, bbox conversion

├── evaluation/
│   ├── backends.py       (72) UltralyticsBackend (lazy YOLO import)
│   ├── benchmark.py     (265) P6: benchmark_candidate, evaluate_backend, Pareto frontier, CandidateResult
│   ├── gate.py          (167) P11: deployment envelope (latency, CPU, recall/mAP50 deltas vs baseline)
│   ├── golden.py        (229) P0: GoldenDataset (create/load/validate, append-only manifest)
│   └── metrics.py       (348) P0: DetectionMetrics, IoU, AP50, mAP50-95, small_object_recall

├── training/
│   ├── candidates.py     (55) Candidate registry: yolov8n/s/m/l, yolo11n/s/m, yolov9t/s
│   └── trainer.py       (238) P7: wraps ultralytics YOLO.train, dry_run mode, before_after golden eval

├── deploy/
│   └── hailo.py         (201) P12: export_onnx, compile_hailo (dry-run placeholder), deploy()

├── inspect/
│   ├── report.py        (187) P2: HTML report generator
│   └── triage.py         (96) P2: quality verdict (useful|bad|duplicate|ignore)

├── frigate/
│   ├── client.py        (472) FrigateClient: httpx, retry/backoff, reviews, events, snapshots, motion
│   ├── events.py         (93) Event dataclass + parse_event (box normalized to xyxy)
│   ├── reviews.py       (126) Review dataclass, parse_review, extract_event_ids
│   ├── snapshots.py      (66) URL/param builders for clean/annotated/region-crop
│   └── motion.py         (40) MotionBucket + parse_motion_activity

├── webapp/
│   ├── app.py            (32) FastAPI factory: create_app(config)
│   ├── api.py           (133) 16 endpoints under /api/*
│   ├── jobs.py          (209) JobManager: daemon thread pipeline runner, log tail capture
│   ├── queries.py       (498) Read-side queries: overview, benchmark, datasets, training, samples
│   ├── serving.py        (59) Image/thumb resolution + lazy thumbnail generation
│   └── static/                Vanilla-JS SPA (app.js ~927 lines, styles.css ~393 lines, vendor/uPlot)

└── migrations/
    ├── 0001_initial.sql (72)  samples, annotations, jobs, collection_failures
    ├── 0002_multiframe_phash.sql (47)  frame_index, perceptual_hash, verified, discovery_windows, deployments
    ├── 0003_review_sync.sql (14)  samples.frigate_reviewed, samples.reviewed_at
    ├── 0004_multi_object.sql (*)  annotations.event_id
    └── 0005_drop_review.sql (*)  review columns removed
```

## Data Layout (runtime)

```
data/
├── frigate_learn.db              SQLite (WAL)
├── images/<YYYYMMDD>/<camera>/<sample_id>.jpg
├── previews/                     HTML reports, thumbnails
├── datasets/<version>/           images/, labels/, dataset.yaml, build.json, manifest.jsonl
├── golden/<version>/             immutable benchmark set
├── training/<model-tag>/         results.csv, weights/best.pt, before_after.json
├── models/<model>/               ONNX, HEF, manifest.json, frigate-detector.yml
└── benchmark-results.json
```

## Key Conventions

- **No comments in code** (except `# ---` section banners in non-webapp; webapp is strict zero-comment)
- **CLI is click**, lazy imports for heavy deps (torch/ultralytics/fastapi)
- **SQLite via SQLAlchemy 2.x**, WAL mode, explicit migrations via `db init`
- **Frigate-specifics live ONLY in `frigate/`** — secrets from env/.env
- **Structured logging:** `logutil` key=value, captured by webapp JobManager
- **Optional-heavy stages** (train, benchmark, deploy, verify) lazy-import and raise clear RuntimeError hints
- **Never commit:** `.DS_Store`, `*.pt`, `data/` contents

## Commands

```bash
.venv/bin/pip install -e ".[dev]"        # dev: pytest, respx, ruff
.venv/bin/pip install -e ".[dev,web]"    # + fastapi/uvicorn
.venv/bin/pip install -e ".[ml]"         # + torch/ultralytics
.venv/bin/frigate-learn db init          # SQLite + migrations
.venv/bin/frigate-learn web              # dashboard at 127.0.0.1:8080
.venv/bin/python -m pytest               # full suite
.venv/bin/python -m pytest tests/test_webapp.py -v
```

## Git Status (as of 2026-09-11)

Uncommitted changes (in progress):
- `deploy/hailo.py`: imgsz parameter plumbing (proper `int | None`, added to manifest, config snippet)
- `evaluation/backends.py`: Minor adjustment
- `test_deploy.py`: Expanded deploy tests (+47 lines)
- `README.md`: Documentation update (+18 lines)
