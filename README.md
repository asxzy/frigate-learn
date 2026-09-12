# Frigate Learn

Active-learning / auto-training pipeline for a Frigate NVR deployment.

The system collects real camera data from Frigate (0.18.x) over its HTTP API,
builds a reproducible, deduplicated dataset from that data, uses a vision
language model (VLM) to verify / correct Frigate detections, trains smaller
detector models intended for Hailo-8 deployment, and only deploys a new model
when it meets configurable accuracy / performance criteria — always measured
against an **immutable golden dataset**.

The current production detector (`YOLOv8l`) is used only as an expensive teacher
on difficult cases. The point is the accuracy/speed tradeoff: get most of the
accuracy at a fraction of the inference cost.

**Deployment constraints**

- Frigate runs on a shared VM (Intel N150 + Hailo-8, ~20-30 cameras). CPU is
  constrained.
- **Nothing CPU- or GPU-heavy from this pipeline may run on the Frigate VM.**
  Training and VLM processing run on a separate machine that only talks to
  Frigate over HTTP.
- Frigate remains the real-time production system. This pipeline never touches
  `frigate.db`, never scrapes the web UI, and (in this version) does not use
  MQTT.

## Roadmap

The whole pipeline is implemented; `run` chains the stages end to end.

| Phase | Deliverable | Status |
|---|---|---|
| P0 | Golden dataset representation + metric interfaces + scaffolding | ✅ implemented |
| P1 | Frigate 0.18 HTTP collector (reviews → events → clean snapshots → SQLite), CLI | ✅ implemented |
| P2 | Dataset inspector + quality triage (useful/bad/duplicate/ignore) | ✅ implemented |
| P3 | Perceptual hashing + temporal sampling + dedup | ✅ implemented |
| P4 | VLM-assisted verification (strict JSON schema) | ✅ implemented |
| P5 | Versioned YOLO dataset builder (+ deterministic splits, manifest) | ✅ implemented |
| P6 | Candidate benchmark against the golden dataset | ✅ implemented (needs `ml` extra) |
| P7 | Candidate training (YOLO fine-tunes) | ✅ implemented (needs `ml` extra) |
| P8 | VLM teacher (forced re-verify replaces prior VLM annotations) | ✅ implemented |
| P9 | Missed-object discovery (motion windows without collected evidence) | ✅ implemented |
| P10 | External data import (COCO annotations + images) | ✅ implemented |
| P11 | Deployment gate (candidate vs. baseline deltas, deployment ledger) | ✅ implemented |
| P12 | Hailo deploy (ONNX export + HEF compile, dry-run artifacts) | ✅ implemented |
| P13 | Automation (`frigate-learn run` orchestrated stages + cron docs) | ✅ implemented |

`data/`, `previews/`, `datasets/`, `golden/` under this repo are gitignored;
source images and SQLite are runtime data.

## Install

```bash
cd frigate-learn
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Run

```bash
cp .env.example .env          # fill in FRIGATE_TOKEN etc.
cp config.example.yaml config.yaml

# initialize the SQLite database (applies migrations)
.venv/bin/frigate-learn --config config.yaml db init

# collect the last 7 days of review/event data
.venv/bin/frigate-learn --config config.yaml collect --from "7 days ago"

# camera/time-scoped collection
.venv/bin/frigate-learn collect --camera driveway --from "2026-09-01" --to "2026-09-08"

# what have we got?
.venv/bin/frigate-learn status
.venv/bin/frigate-learn db stats

# inspect + triage collected samples (HTML report per camera/day)
.venv/bin/frigate-learn inspect --out previews/report
.venv/bin/frigate-learn triage set s00f… --quality useful
.venv/bin/frigate-learn triage batch --quality bad --camera front --limit 50

# VLM verification (requires a reachable openai-compatible endpoint: config vlm.*)
# writes verified=1 annotations. Field-tested against a local gateway serving
# Qwen3.8-27B: with the strict system prompt the model emits schema-valid JSON,
# but it is slow (~20–180 s/image) and occasionally drifts to pixel bboxes.
# VLM failures are DROPPED (sample quality -> bad): they are excluded from
# builds and never re-submitted (rescue later with `triage set`). Every verify
# run therefore terminates with samples either verified or dropped.
.venv/bin/frigate-learn verify

# missed-object discovery: motion buckets with no collected sample
.venv/bin/frigate-learn discover --days 1

# import external training data (COCO JSON + images → external samples)
.venv/bin/frigate-learn import-coco data/external/coco.json data/external/images

# build a frozen dataset version, then evaluate candidates / gate / deploy
.venv/bin/frigate-learn dataset build v003
.venv/bin/frigate-learn benchmark          # needs the `ml` extra (ultralytics)
.venv/bin/frigate-learn gate               # vs. deployment.baseline, writes ledger
.venv/bin/frigate-learn train v003 --real  # needs the `ml` extra
.venv/bin/frigate-learn deploy --real      # ONNX export + Hailo HEF (dry-run by default)

# the whole pipeline in one shot (stages/minimal deps are skipped cleanly)
.venv/bin/frigate-learn run --days 1
```

`collect` is idempotent: re-running the same range produces no duplicate
samples (dedup by Frigate `event_id`, with SHA-256 image hashing + optional
dHash near-duplicate suppression).

## Webapp

A local dashboard (`frigate-learn web`) is the pipeline's control panel: it
reads the same SQLite DB and config the CLI uses, and lets you inspect state,
launch pipeline stages, and triage samples from the browser — no shell needed.

```bash
# one-time extra (fastapi + uvicorn)
.venv/bin/pip install -e ".[web]"

# run the dashboard, then open http://127.0.0.1:8080 in a browser
.venv/bin/frigate-learn web --host 127.0.0.1 --port 8080
```

The SPA has no build step (plain HTML/JS with a vendored uPlot) and six views:

- **Overview** — headline counts (samples, cameras, disk free, VLM status),
  quality verdict totals, next build version, recent `jobs`, and the
  pipeline-stage strip.
- **Benchmark** — candidate-vs-baseline metrics, a latency vs mAP50 scatter,
  the deployments ledger, and the **Re-benchmark + gate** button.
- **Quality** — verdict totals, verified boxes per class, and the review
  backlog of unverified samples.
- **Datasets** — dataset version table (train/val splits, verified counts) and
  golden-dataset status.
- **Training** — pick a training run: best/latest mAP50, weights presence, and
  mAP50/recall epoch curves.
- **Triage** — filter the sample grid (quality/status/camera/verified); click a
  thumbnail for a lightbox with detection + Frigate bbox overlays.

There are two POST actions: the **Re-benchmark + gate** button runs the
`benchmark` and `gate` stages, and the Triage lightbox sets a sample's quality
verdict (`useful` / `bad` / `duplicate` / `ignore`).

Constraints, plainly:

- Localhost-only and single-user: no auth, no CORS. Bind to `127.0.0.1` and
  don't expose it on a network.
- Pipeline stages run **in the same process as the server** (a background
  thread). Stopping the server stops a running stage; a restart marks the stale
  job `failed`.
- Every run is recorded in the `jobs` table and shown in Overview; only one
  pipeline job runs at a time.

## Deploy decision: real benchmark + gate

The deploy/no-deploy call is made by **benchmark → gate**. `gate` only becomes
meaningful once there are real numbers (the gate reads the *last*
`data/benchmark-results.json`), so the flow is:

```bash
# 1. install the ML extra (torch + ultralytics) — one-time, on the training host
.venv/bin/pip install -e ".[ml]"

# 2. benchmark the baseline AND the candidates in ONE run: the gate compares
#    candidates against the baseline by name, so both must land in the same
#    results file. yolov8l is the configured baseline (deployment.baseline).
#    Available candidates: yolov8{n,s,m,l}, yolo11{n,s,m}, yolov9{t,s}.
.venv/bin/frigate-learn benchmark \
    --candidate yolov8l --candidate yolov8n --candidate yolov8s

# 3. gate the results: PASS/FAIL per candidate vs. the deployment envelope,
#    recorded in the deployments ledger
.venv/bin/frigate-learn gate
```

All candidates are **COCO-pretrained checkpoints** (ultralytics downloads the
`.pt` on first use) fine-tuned on your collected dataset — no public image
dataset is needed. The pretrained weights carry the backbone knowledge; the
frozen dataset versions provide the task-specific fine-tuning. If accuracy is
short, grow/verify the collected data — adding COCO images is not a lever.

What each step produces, measured on the **immutable golden dataset**
(`data/golden/golden-v001/`, seeded from VLM-verified samples):

- **Accuracy** — COCO-style `mAP50`, `mAP50-95`, precision, recall, F1,
  small-object recall, FP rate; per class and overall.
- **Speed** — mean per-image latency (30 repeats after 3 warmups) and
  throughput FPS; on this box latency is measured in the untrained ultralytics
  runtime — real Hailo latency is only known after `deploy --real` on the Hailo
  host.

A candidate **PASSes** (`evaluation/gate.py`) when it stays inside the
`deployment:` envelope in `config.yaml`:

- latency ≤ `max_latency_ms` (20 ms)  — with no baseline this is the only check
- recall ≥ baseline − `min_recall_delta` (−0.01)
- mAP50 ≥ baseline − `min_map50_delta` (−0.01)
- CPU ≤ `max_cpu_percent` (80 %, unmeasured → skipped)

The `gate` run itself prints `[PASS]`/`[FAIL]` per candidate with the reasons;
each verdict is also persisted in the `deployments` ledger table
(`model_name`, `version`, `verdict`, `metrics_json`, `reasons_json`). A PASS is
the hand-off for `train <model> <version> --real` + `deploy <model> <best.pt>
--real` on the Hailo/training host.

> **Caveat:** golden-v001 is only 26 images — fine as a regression tripwire, too
> small for a statistically decisive mAP. Grow it by curating more verified
> samples before trusting the verdict on its own.

## Pipeline

`frigate-learn run` executes the configured stages in order:

```
collect → verify → build → train → benchmark → gate → deploy
```

Each stage reports `executed | skipped | failed` (a skipped stage explains
why — missing VLM config, ML extra not installed, no new dataset, golden
missing, nothing trained, etc.), so a nightly cron can be launched
unconditionally while humans only read the summary. Stages honour
`automation.enable` / `automation.till`, `--steps`, `--until`, `--keep-going`
and `--days` (see `--help`). Dry-run is the default for `train`/`benchmark`/
`deploy` so smoke runs never touch real artifacts without `--real`.

## Layout

```
src/frigate_learn/
├── cli.py                 # `frigate-learn` command (collect, inspect, triage, verify,
│                          #   discover, dataset, import-coco, benchmark, train, gate,
│                          #   deploy, run, status, db)
├── run.py                 # P13: `run` pipeline orchestrator (collect…deploy)
├── config.py              # YAML config + env interpolation
├── db.py                  # SQLite + migration runner
├── models.py              # SQLAlchemy ORM models
├── times.py               # "7 days ago" → timestamp
├── logutil.py             # key=value structured logging helpers
├── frigate/               # Frigate 0.18 HTTP adapter (API specifics live HERE)
│   ├── client.py          # transport, auth, retries
│   ├── reviews.py         # review model + parsing + event extraction
│   ├── events.py          # event model + parsing
│   ├── snapshots.py       # snapshot / preview URL + download helpers
│   └── motion.py          # motion-activity model + parsing (P9 consumer)
├── collection/
│   ├── collector.py       # review → event → snapshot → SQLite pipeline
│   ├── sampling.py        # P3: temporal sampling within an event
│   ├── dedup.py           # P3: SHA-256 + dHash near-duplicate suppression
│   ├── discover.py        # P9: motion windows without collected evidence
│   └── external.py        # P10: COCO import as external samples
├── inspect/
│   ├── report.py          # P2: HTML/exif report grouped by camera/day
│   └── triage.py          # P2: per-sample / batch quality verdicts
├── dataset/
│   ├── yolo.py            # YOLO-labelfmt: normalization + validation
│   ├── manifest.py        # JSONL manifest reader/writer
│   ├── splits.py          # deterministic splits
│   └── builder.py         # P5: versioned dataset builder
├── annotation/
│   ├── schema.py          # P4: strict VLM JSON schema + tolerant extractor
│   ├── vlm.py             # P4: openai-compatible provider (retries + retry on bad JSON)
│   └── verifier.py        # P4/P8: verification pipeline → verified annotations
├── evaluation/
│   ├── metrics.py         # detector-agnostic metrics (precision/recall/F1/mAP…)
│   ├── golden.py          # immutable golden dataset representation
│   ├── benchmark.py       # P6: candidate benchmark runner
│   ├── gate.py            # P11: evaluate_gate + deployments ledger
│   └── backends.py        # P6: ultralytics/onnx backend wrapper
├── training/
│   ├── trainer.py         # P7: YOLO fine-tuning (real run needs `ml` extra)
│   └── candidates.py      # P7: candidate table (name → weights)
├── deploy/
│   └── hailo.py           # P12: ONNX export + HEF compile + Frigate detector YAML
└── webapp/                # local control-panel dashboard (`frigate-learn web`)
    ├── app.py             # FastAPI app factory + static mount
    ├── api.py             # HTTP surface: 16 endpoints under /api/*
    ├── queries.py         # read-side queries (overview, benchmark, datasets, …)
    ├── serving.py         # image/thumb resolution + on-demand thumbnail cache
    ├── jobs.py            # JobManager: pipeline stages in a background thread (jobs ledger)
    └── static/            # no-build SPA: index.html, app.js, styles.css, vendor/uPlot
```

## Design rules

- **Frigate specifics stay inside `frigate/`.** Nothing else in the codebase
  knows the URL layout or response shapes.
- **The collector owns its state.** Frigate's `has_been_reviewed` and review
  thumbnails are never used as pipeline state.
- **One review ≠ one object.** Review items are expanded to their underlying
  event IDs; each event becomes (at most) one sample.
- **Training images are clean snapshots.** The annotated snapshot is only
  optionally stored for debugging, never used for training.
- **The golden dataset is immutable.** Later collection/training jobs can read
  it but never overwrite it.
- **Secrets come from env / `.env`** (e.g. `FRIGATE_TOKEN`), never from the
  config file or the repo.

## Frigate 0.18 API deviations (isolated in `frigate/`)

The collection plan assumed `GET /api/events/{id}/snapshot-clean.webp`. Frigate
0.18 actually serves clean snapshots at
`GET /api/events/{id}/snapshot.jpg?bbox=0&timestamp=0` (JPEG). The adapter
implements that; everything above the adapter is unaffected. Other specifics:

- Review listing is `GET /api/review` (singular) and returns a plain array —
  pagination is windowed with the `before` cursor inside `client.py`.
- Review `data.detections` is a list of event-ID strings (labels in
  `data.objects`). The parser also accepts the older dict-shaped form.
- List endpoints return ISO-8601 timestamps, single-item endpoints return unix
  floats; `parse_ts()` handles both.

## Tests

```bash
.venv/bin/pytest
```

All tests use mocked HTTP (`respx`); no live Frigate instance required.

## Security notes

- Never print, log, or commit the Frigate token.
- Never put the token in dataset metadata, manifest files, or URLs.
- The config file resolution does `$VAR` / `${VAR}` / `${VAR:-default}`
  interpolation from the environment, so secrets stay out of YAML.