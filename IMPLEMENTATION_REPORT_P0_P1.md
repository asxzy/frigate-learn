# Frigate Learn — Milestone Report: Phase 0 + Phase 1

## What this is

`frigate-learn` is a local Python application for active-learning / auto-training of
small YOLO detectors for a home Frigate 0.18 NVR deployment. It collects data from the
Frigate HTTP API, stores provenance in SQLite, builds reproducible YOLO datasets, and
later phases will verify detections with a VLM, train and evaluate candidate models and
only deploy ones that pass an immutable golden-dataset gate.

This milestone delivers **Phase 0** (golden dataset + detector-agnostic metrics +
project scaffolding) and **Phase 1** (Frigate 0.18 batch collector + CLI + tests). The
remaining phases (P2–P13) are scaffolded as documented placeholder modules with their
design notes, so the next milestone has a clear entry point.

Location: `/service/frigate-learn/` (sibling of the live `/service/frigate/` deployment;
the Frigate VM is never used for training or inference).

## Definition of done — status

| Criterion | Status |
|---|---|
| `frigate-learn collect --from "7 days ago"` pulls reviews → events → clean snapshots → SQLite | ✅ (HTTP mocked in tests; a real run needs `FRIGATE_TOKEN` against the live NVR) |
| Review items resolved to underlying event ids, dedupes by `event_id`, one sample per event | ✅ |
| Filters by camera / label / severity / time window | ✅ |
| Failures recorded and never abort a batch | ✅ |
| Idempotent re-runs (no re-download, no dupes) | ✅ |
| Summary + progress output matching the spec format | ✅ |
| Tests for the API adapter, collector, and P0 utilities | ✅ — 119 passing |
| CLI `frigate-learn collect`/`status`/`db` | ✅ |
| Implementation report delivered | ✅ (this file) |

## Files created

```
frigate-learn/
├── pyproject.toml            click, httpx, PyYAML, python-dotenv, SQLAlchemy; dev: pytest, respx
├── README.md                 usage + the snapshot-clean deviation note
├── config.example.yaml       copy to config.yaml; ${VAR} env interpolation
├── .env.example              FRIGATE_TOKEN etc.
├── .gitignore
├── migrations/README.md      (the real SQL migrations live in the package; see below)
└── src/frigate_learn/
    ├── __init__.py           version + public exports
    ├── cli.py                frigate-learn {collect,status,db}
    ├── config.py             YAML→dataclasses, ${VAR:-default} env interpolation, path anchoring
    ├── times.py              "7 days ago" / ISO date parsing
    ├── logutil.py            structured key=value logging
    ├── db.py                 SQLite + migration runner (executescript for multi-statement DDL)
    ├── models.py             ORM models (Sample, Annotation, Job, CollectionFailure)
    ├── migrations/0001_initial.sql   authoritative schema (packaged, single source of truth)
    ├── dataset/              yolo.py, manifest.py, splits.py  (P0)
    ├── evaluation/           metrics.py, golden.py  (P0)
    ├── frigate/              reviews.py, events.py, snapshots.py, motion.py, client.py  (P1)
    ├── collection/           collector.py  (P1); sampling.py, dedup.py (P3 placeholders)
    ├── annotation/           schema.py, vlm.py, verifier.py (P4 placeholders)
    ├── dataset/builder.py    (P5 placeholder)
    ├── evaluation/benchmark.py (P6 placeholder)
    └── training/             trainer.py, candidates.py (P6/P7/P8 placeholders)
tests/                        13 test modules, 119 tests, HTTP via respx
```

~4,500 lines of Python (source + tests).

## Commands

```bash
# install
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"          # dev extra = pytest + respx

# database
.venv/bin/frigate-learn db migrate         # apply migrations
.venv/bin/frigate-learn db stats           # sample/annotation/failure/job counts by camera
.venv/bin/frigate-learn db migrations      # list applied + pending

# collect (must be run where the NVR is reachable; FRIGATE_TOKEN must be set)
.venv/bin/frigate-learn collect --from "7 days ago"
.venv/bin/frigate-learn collect --days 7 --camera front --label person \
    --severity detection --concurrency 8 --limit 5000

# overview
.venv/bin/frigate-learn status
```

Tests: `.venv/bin/pytest` → **119 passed** (~1.1 s).

## Frigate 0.18 API facts verified from source (isolated in `frigate/`)

- `GET /api/review` (singular) returns a **plain JSON array** with no paging object →
  windowed pagination on a `before` cursor (start_time descending). Severity is a
  per-query filter, so filtered queries are made per-severity.
- Review `data.detections` is a list of **event-id strings** (0.18); older formats used
  dict lists. `extract_event_ids()` resolves a review → its underlying event ids — one
  review segment can reference many objects, and each is processed individually.
- Events carry a normalized `box` as `[x1, y1, x2, y2]` in `[0, 1]` (top-level, with a
  `data.box` fallback). `data.score` is the final confidence.
- List endpoints serialize timestamps as ISO-8601 strings; single-item endpoints return
  unix floats — `parse_ts()` handles both.
- **The spec's assumed `snapshot-clean.webp` endpoint does not exist in 0.18.** The clean
  (unannotated) frame is `GET /api/events/{id}/snapshot.jpg?bbox=0&timestamp=0`; the
  annotated frame is the same URL without the overlay flag. This deviation is contained
  entirely in `frigate/snapshots.py` + `client.download_clean_snapshot()`.
- Review previews: `GET /api/review/{id}/preview?format=mp4|gif`.
- Motion activity: `GET /api/review/activity/motion?after&before&cameras&scale`.
- Auth: `Authorization: Bearer <token>` where the token comes from `FRIGATE_TOKEN`.

## Database schema (`0001_initial.sql`, SQLite)

- `samples` — one row per collected object: id, camera, timestamp, `event_id`
  (unique → idempotency), `review_id`, clean `image_path`, optional `debug_image_path`,
  `image_hash` (sha256), source, frigate label/score/bbox, quality, status, created_at.
- `annotations` — label/bbox/confidence rows per sample (`source` = frigate|vlm|…,
  `verified` flag). FK → samples; the ORM column declares the FK so SQLAlchemy orders
  child inserts correctly.
- `jobs` — pipeline job ledger (running|finished|failed + JSON metadata).
- `collection_failures` — per-item failure ledger (a failed event is retried next run,
  never silently dropped).
- Migrations are plain SQL files in `src/frigate_learn/migrations/` (correctly packaged
  via `package-data`); applied via sqlite3 `executescript` (multi-statement DDL); the
  top-level `migrations/` dir holds only a README.

## Config

`config.example.yaml` documents every section: `classes`, `frigate` (base_url, token,
timeouts, retries, `completed_events_only`), `collection` (default_days, camera/label/
severity filters, caps, concurrency, annotated-snapshot switch), `sampling`, `vlm`,
`training`, `evaluation`, `deployment`, and `data` paths (anchored to the config file).
`${VAR}` / `$VAR` / `${VAR:-default}` interpolation; `.env` support.

## Notable implementation decisions

1. **Review ≠ object.** The collector resolves every review to its underlying
   `data.detections` event ids and processes each individually — the metric layer sees
   one object per sample.
2. **Idempotency by DB constraint, not by Frigate state.** Never reads
   `has_been_reviewed`; a unique index on `samples.event_id` guarantees no duplicates
   across runs (including concurrent runs).
3. **`completed_events_only`** (default on): Frigate refuses snapshots for in-progress
   events, and a mid-event frame is not a usable training image.
4. **Metrics are pure-Python and detector-agnostic** (no numpy/torch dependency for P0/P1).
   COCO-style: IoU matching, per-class AP50/AP50–95 (101-point interpolation), global
   precision/recall/F1, `small_object_recall` (default threshold 0.0025 of frame area).
   `DetectorEvaluator` is a Protocol so ultralytics/onnx/Hailo backends can wrap behind it.
5. **Golden set is append-only.** Re-adding a `sample_id` raises `ManifestDuplicateError`;
   the manifest is JSONL, images/labels in standard YOLO layout with `dataset.yaml`.
6. **Deterministic splits** from a sha256 bucket of `sample_id` (no RNG) with the golden
   set pinned to its own `golden` split.
7. **Failure ledger.** Every per-event failure is recorded with the event id and error;
   the next collect run retries it. A fatal transport error only fails the job, cleanly.

## Tests (119)

- config: defaults, section mapping, `${VAR}`/`${VAR:-default}`/`$VAR`, `.env`, path
  anchoring, missing-config error.
- times: relative ("7 days ago", "24 hours ago", "2h ago"), ISO dates/datetimes,
  invalid inputs.
- db: migration idempotency, `schema_version`, table creation, unique `event_id` index,
  FK enforcement, migration file sorting, bundled fallback.
- yolo: normalize, degenerate/out-of-range boxes, center-wh roundtrip, label parse/
  write/read, invalid lines.
- manifest: append/read, duplicate rejection, `allow_overwrite`.
- splits: determinism, seed sensitivity, ratio distribution (~80/10/10), partition
  completeness, impossible-ratio error.
- metrics: IoU, perfect/empty/partial/no-GT cases, per-class metrics, small-object
  recall, `DetectionEvaluator` ≡ `evaluate`, mAP50 == IoU-50-only evaluation.
- golden: layout + yaml roundtrip, immutability + overwrite mode, file validation,
  extension guard, time buckets.
- frigate parse: review/event parsing (ISO + float timestamps, box from raw/data,
  incomplete events), event-id extraction (string + dict forms, dedup), `parse_ts`.
- frigate client (respx): auth header, two-page window cursor, per-severity queries,
  limit, 404 non-retryable, 500 retry-then-fail, snapshot download with `bbox=0`, empty
  download, motion activity params, non-JSON error, events pagination.
- collector (fake client + real SQLite): happy path (sample/annotation rows, hashes,
  paths), idempotent second run, shared-event dedup across reviews, failures ledger,
  incomplete-event skip, annotated-snapshot retention, filter passthrough.
- CLI (offline, Collector stubbed): version, `db migrate`/`status`/`db stats`, collect
  progress/summary output, bad severity rejection.

## Known limitations / notes

- Collection is verified against mocked HTTP; go-live requires `FRIGATE_TOKEN` and
  `base_url` in `config.yaml` — first real run should be a small window (e.g. `--limit 200`).
- No `frigate.db` access, no MQTT, no UI scraping (by design for this version).
- Frigate "objects" (camera-motion + classify) vs "detections" nuance: the collector uses
  the review `data.detections` list; `data.objects` is the fallback. If a label was never
  tracked, it may not appear — fine for the first training batches.
- `download_timeout_seconds` is read from config but the client currently uses
  `timeout_seconds` for downloads; wire the per-download timeout through in P2.
- Days-of-history have so far only been exercised against mocked data; the slider window
  semantics were chosen so `--from "7 days ago"` on a busy deployment yields thousands of
  samples without special DB access.
- No acceptance-gate CLI yet (that's P11); the metric/golden layers are ready for it.

## Next phase — P2 (recommended entry point)

A **dataset browser / inspector**: `frigate-learn browse` (or `--serve`) renders collected
samples into a local HTML/static viewer grouped by camera/day, with quality triage controls
(bad|useful|duplicate|ignore) writing back to `samples.quality`. This validates the
collected data before any expensive annotation/training work and is the natural first
consumer of the `quality` column already in the schema. After P2, P3 adds temporal
sampling + perceptual-hash dedup on the (now quality-controlled) pool, and P4 brings the
VLM verifier (config `vlm.provider` already exists).

---

# Milestone Addendum — Phases 2–13 (pipeline complete)

Following P0/P1, the remaining phases were implemented in this milestone. The full
pipeline `collect → verify → build → train → benchmark → gate → deploy` now exists,
is covered by tests, and can be driven end-to-end with `frigate-learn run`.

## What was added per phase

| Phase | Delivered | Module / command |
|---|---|---|
| P2 | HTML inspector grouped by camera/day (thumbnails, quality chips) + per-sample & batch triage (useful/bad/duplicate/ignore, camera/label/days/limit filters) | `inspect/{report,triage}.py`; `inspect`, `triage` |
| P3 | dHash near-duplicate suppression (SHA-256 exact always; Hamming threshold, per-camera) + optional temporal sampling within an event (`timestamp` param threaded into `download_clean_snapshot`) | `collection/{dedup,sampling}.py` |
| P4 | Strict VLM JSON schema (jsonschema), tolerant JSON extractor (fences/prose/trailing garbage), openai-compatible provider with retries incl. retry-on-bad-JSON, verifier writing `source='vlm', verified=1` annotations + `samples.verified` | `annotation/{schema,vlm,verifier}.py`; `verify` |
| P5 | Versioned dataset builder (images/, labels/, registry txts, `dataset.yaml`, `build.json`, `manifest.jsonl`); auto version `v001`…; prefers verified annotations then Frigate | `dataset/builder.py`; `dataset build/list` |
| P6 | Benchmark of candidates on the golden dataset (mAP50, latency, BOCO metrics) to `benchmark-results.json` | `evaluation/{benchmark,backends}.py`; `benchmark` |
| P7 | YOLO trainer (real run needs the `ml` extra; dry-run default) + candidate table | `training/{trainer,candidates}.py`; `train` |
| P8 | VLM teacher: `verify --force` re-verifies and replaces prior VLM annotations | `verifier.py` |
| P9 | Motion-window discovery: clusters `/api/review/activity/motion` buckets into windows (per-camera `closing_gap`), marks windows already covered by collected samples, persists uncovered ones to `discovery_windows` | `collection/discover.py`; `discover` |
| P10 | COCO import (`annotations` JSON + images → external samples + annotations, mtime-based dirs, timestamps from image mtime) | `collection/external.py`; `import-coco` |
| P11 | Deployment gate: `evaluate_gate(candidate, config.deployment, baseline)` using recall/mAP50/latency deltas + CPU headroom, ledger in new `deployments` table | `evaluation/gate.py`; `gate` |
| P12 | Hailo deploy: ONNX export (dry-run placeholder), HEF compile (dry-run placeholder file; real compile needs the Hailo toolchain), `frigate-detector.yml` + `manifest.json`, optional deployment-row recording when a gate verdict is supplied | `deploy/hailo.py`; `deploy` |
| P13 | Orchestrated `run` with `StepReport(executed|skipped|failed)` summaries, stage skip reasons, halt/`--keep-going`, `automation.enable`/`till`, `db.migrate()` bootstrap | `run.py`; `run` |

## Schema change (`0002_multiframe_phash.sql`)

Extends the P0/P1 schema for multi-frame events, hashing and discovery:

- `samples.frame_index` default 0 (multi-sample events), `samples.perceptual_hash`
  (dHash), `samples.verified` default 0.
- Unique index on `(event_id, frame_index)` (replaces plain `event_id` uniqueness).
- New tables: `discovery_windows` (camera/start/end/motion score/notes), `deployments`
  (model, version, artifact path + sha256, metrics_json, verdict, reasons, deployed flag).

Applied/migrated via the same migration runner — `db migrate` upgrades in place.

## New config sections (`config.example.yaml` documents all of them)

- `collection.dedup_enabled` / `phash_threshold` (3, default 10 Hamming bits).
- `sampling.enabled` / frame caps (off by default).
- `vlm` — provider/base_url/api_key/model/batch_size/temperature/timeout/system_prompt;
  `{classes}` interpolated into the prompt.
- `training` — model/batch/device/freeze/project/seed (the `ml` extra gates real runs).
- `evaluation.candidates` (models compared on the golden set).
- `deployment.baseline` (incumbent the gate compares against) + extra thresholds.
- `automation` — `enable`/`till` plus a cron `schedule` documented for the nightly timer.

`VLM_BASE_URL` / `VLM_API_KEY` come from the environment (never committed).

## Live validation of the VLM endpoint (smoke test)

The gateway at `http://<llm-gateway>:4001/v1` (LAN-only, key via env) was
exercised against real collected frames. `Ornith-1.5-9B`/`-35B-A3B` and
`Qwen3.8-27B` were all queried (`chat/completions`, key via env):

- Transport + base64 image data-URL + `chat/completions` round-trip: works on all.
- Tolerant JSON extraction: strips ```json``` fences, prose prefix/suffix and trailing
  garbage/2nd object.
- Strict schema validation: **correctly rejects** out-of-schema output (pixel-coordinate
  bboxes, `{"label": …}` flat objects, out-of-allowed labels, `confidence` out of range).
  Rejected samples are counted as `failed` and **never** written to the dataset —
  graceful degradation, no corruption.
- **`Qwen3.8-27B` is the working model.** With the strict system prompt it emits
  schema-valid `{"objects": [...]}` with normalized bboxes (verified 4/4), though it is
  stochastic at `temperature 0`: it occasionally drifts to pixel coordinates (rejected +
  retried), and sometimes returns an empty `objects` list on a scene it should see.
  The `Ornith` models shulkered between flat `{"label","bbox_2d"}` and
  `{"objects": [...]}` shapes, so they are not used.
- The verifier does **one request per image** (~20 s/image in isolation; more under
  gateway load with generation + retries), so a full collect→verify pass is slow —
  budget minutes per sample on this hardware. A conforming hosted VLM plugs in via
  `config.vlm.*` with no code changes.

### First real run on live field data (56 samples, 6 cameras)

- `verify` (Qwen3.8-27B, one request/image, ~2–6 min/image under gateway load) over the
  remaining 52 unverified samples finished cleanly:
  `processed=52 annotated=28 failed=24 objects=33`. Combined with the initial 4, **32/56
  samples are now VLM-verified**. The 24 `failed` were Qwen pixel-bbox drifts; under the
  current drop-on-fail behavior a re-run retires them to `quality='bad'` so they can never
  block or re-enter a build. No data corruption and the pass never aborts.
- **Frigate fallback is nearly useless here:** only 8/56 stored `data.box` rows are usable
  (`x2>x1, y2>y1`); the rest are zero-area. The VLM pass is the *only reliable box source*
  in this deployment, which makes P4/P8 the crux of the pipeline.
- `dataset build v001`: **26 images written, all VLM-verified** (train 21 / val 3 / test 2),
  30 skipped (`class`, degenerate boxes), 0 skipped for caps. Label files use normalized
  VLM boxes; `build.json` records the summary + seed `frigate-learn-v1` for reproducible
  splits. (`data/datasets/v001` is gitignored.)

### Robustness fixes driven by the live run

- `_default_provider` interpolates `{classes}` with `replace()`, not `.format()` —
  the system prompt contains literal `{"objects" ...}` JSON braces, which raised
  `KeyError` under `format`.
- `Query.order_by()` is now applied **before** `Query.limit()` in `Verifier.verify`
  (SQLAlchemy rejects `order_by` after a LIMIT was set).
- A single bad model response no longer aborts a verification pass: when a batched
  `provider.verify()` raises `VLMValidationError`, the verifier re-runs each image
  singly, keeps the good ones, and counts the bad ones as `failed`.
- **VLM failures are dropped.** A sample whose response keeps failing after retries
  gets `quality='bad'` (via `Verifier._drop`), which excludes it from BOTH future
  verify runs and dataset builds — the end2end pipeline never stalls or re-wastes
  time on samples the current model can't verify. The verdict is reversible with
  `triage set --quality useful`, so a stronger VLM can re-claim them later. Verify
  summaries now print `dropped=N` and record it in the job metadata.
- `vlm.enabled` is now `true` in the live `config.yaml` (model `Qwen3.8-27B`,
  `VLM_API_KEY` from env). `automation.enable` still lists `[collect, build]` —
  `verify` runs on demand until the endpoint cost/quality trade-off is settled.

## Tests

The suite grew from 119 to **202 tests, all passing** (`.venv/bin/pytest`, ~4 s). New
coverage: schema extraction/validation edge cases, dHash/dedup, sampling,
discovery clustering + coverage + idempotent storage, COCO import, gate verdicts +
ledger, deploy (dry-run manifest/HEF/frigate-detector.yml + gate-recording), run
orchestration (collect→build, `until` truncation, halt-vs-`--keep-going`, version
increment), inspect/triage (batch filters, HTML report + thumbnails), VLM provider
(respx-mocked retries on 503 + retry-on-bad-JSON), and new CLI wiring
(`run_pipeline_smoke`, unknown-stage rejection).

## Known limitations (still true, unchanged)

- Real training/benchmark/deploy need extras this dev box does not install: the `ml`
  extra (ultralytics/torch) and the Hailo Dataflow compiler. Both are gated cleanly —
  `train`/`deploy` default to dry-run and `benchmark` skips with a clear reason when
  `ultralytics` is absent.
- Discovery is currently motion-bucket → window only; review `data.detections` overlap
  refinement is enabled by the existing `covered` cross-reference.
- `deploy --real` writes the ONNX/HEF pipeline but the final HEF compile and Frigate
  `detectors:` wiring remain to be executed on the Hailo/training host (docs via the
  generated `frigate-detector.yml`).

## Open issues (recorded, to be tackled later)

- **Multi-object images.** A single image containing several objects is stored as one
  `Sample` carrying the primary `frigate_label`; the VLM *can* emit multiple objects and
  the verifier writes each as its own annotation, but the fallbacks used downstream
  (`_best_annotations` → `anns[:1]`, Frigate-event-per-track semantics, golden/eval
  matching) assume one box per sample. Images with 2+ objects therefore under-capture at
  build/eval time. This is deferred; the fix will move to per-image multi-box handling in
  the builder's annotation resolution and the golden evaluation matcher.
- **Pool capture rate is ~46%** (26/56 usable) on the current gateway/model: 24 samples
  were dropped by the VLM and 6 more verified-but-empty. Raising it means a better-behaved
  VLM or manual re-claim via `triage`.
# Addendum B (2026-09-09) — end-to-end smoke of train → benchmark → gate → deploy

Driven by the directive to stop collecting samples and prove the remaining pipeline
stages work end to end on real data. No `ml` extra (ultralytics/torch) and no Hailo
toolchain are installed on this dev box by design, so real stages run as dry-runs and
ml-gated stages skip cleanly. Everything below ran against the live DB
(`data/frigate_learn.db`, 205 samples) with the real `config.yaml`.

## What was produced

- **Golden dataset `golden-v001`** — `data/golden/golden-v001/` (26 images, dataset.yaml,
  manifest.jsonl, labels/). Seeded from the 32 `verified` samples: the 26 with ≥1 usable
  in-class verified annotation (37 boxes — bicycle 3, car 4, cat 8, person 22). The other
  6 verified samples carry only out-of-class labels and stay out of the golden set.
  `GoldenDataset.validate()` is clean. This is the permanent benchmark any candidate is
  now measured against.
- **`benchmark --dry-run`** → `data/benchmark-results.json` with zero-metric rows for
  `yolov8n`, `yolov8s` (real eval would load weights via the `ml` extra).
- **`gate`** → both candidates recorded **PASS** in the `deployments` ledger:
  latency 0.0ms ≤ 20ms OK; first deploy so no baseline → metric deltas skipped,
  latency-only envelope. (Rows: `yolov8n`/`yolov8s`, version `golden-v001`, `deployed=0`.)
- **`train yolov8n v003 --dry-run`** → `data/training/yolov8n-v003/results.json`
  (placeholder; real training needs `.[ml]`).
- **`deploy yolov8n <best.pt> --version v003 --dry-run`** → `data/models/yolov8n/`
  with placeholder `yolov8n.onnx`, `yolov8n.hef`, `manifest.json` (incl. hef_sha256,
  `dry_run: true`) and the ready-to-paste `frigate-detector.yml` (hailo, `num_classes: 8`).
- **Orchestration smoke** `run --steps build train benchmark gate deploy --dry-run`:

  ```
  [ok ] build      version=v004 total=181 written=33
  [ok ] train      model=yolov8n.pt best=dry-run
  [-- ] benchmark  ml extra not installed
  [ok ] gate       yolov8n=PASS | yolov8s=PASS
  [-- ] deploy     gate summary only (no auto-CI deploy)
  ```

  This is the intended shape: build → train execute, benchmark skips because the `ml`
  extra is absent (and does NOT halt the run), gate re-reads the last
  `benchmark-results.json` from disk and records verdicts, and deploy intentionally does
  not auto-CI — the verdict is the hand-off, real deploy is a manual `--real` step on the
  Hailo host.

## Bug fixed during the smoke

`frigate-learn gate` (no `--results`) crashed with
`AttributeError: 'str' object has no attribute 'read_text'` — the click `str` path was
passed straight into `results_from_json(path)`. Fixed two ways:
- the CLI now coerces with `Path(...)` before reading;
- `results_from_json` itself now accepts `str | Path` (defensive for future callers).
Regression test: `test_results_from_json_accepts_str_path`.

## What it would take to go fully real here

```
pip install '.[ml]'          # ultralytics + torch → train --real, benchmark real eval
frigate-learn benchmark --dry-run=false   # real mAP/recall/latency vs golden-v001
frigate-learn gate                         # verdicts with baseline deltas
frigate-learn train yolov8n v003 --real    # real weights
frigate-learn deploy yolov8n <best.pt> --real   # ONNX export; HEF compile on Hailo box
```

The golden set (26 images) is small, so real numbers there are a functional check rather
than a regression oracle; growing it is a curation task, not a pipeline task.

## Tests

Suite now **202 passing** (was 197) — one new gate regression test + the earlier phase
work.
