# Verified-only training with honest splits and a fixed dashboard metric curve

Date: 2026-09-10

## Problem

1. **Training includes unverified data.** `DatasetBuilder.build()` defaults to
   `verified_only=False`, and `_best_annotations` falls back to the original
   Frigate detection when no verified VLM/human annotation exists
   (`builder.py:219`). The pipeline `build` stage therefore produces datasets
   (weakly) labeled by Frigate, contradicting the crop-mode design where the
   VLM is the sole label source.
2. **Validation is in-sample.** `dataset.yaml` maps `train: images` and
   `val: images` to the same directory, so ultralytics validates on the
   training set. The reported "val mAP50 0.962" for v003 is measured on
   training images. The real splits (train=30 / val=4 / test=8) exist only as
   `.txt` files ultralytics never reads.
3. **No before/after comparison.** The effect of fine-tuning is not surfaced:
   pretrained vs fine-tuned metrics on a fixed reference set.
4. **Dashboard metric curve breaks.** `_read_csv` in `webapp/queries.py`
   returns `float("nan")`/`float("inf")` for such cells; Starlette serializes
   JSON with `allow_nan=False`, so a single NaN epoch in a training
   `results.csv` makes `/api/training/<run>` return **500** and the Training
   view (mAP50/recall epoch curve) collapses to an error box. Additionally,
   empty cells reach the JS as `null` and are coerced via `Number(null) = 0`,
   producing false zero-spikes in the curve; the curve is also not bridged
   across missing epochs.

## Goals

- Training runs only on **verified** images, guaranteed at the source (build).
- Validation runs on a **true holdout** (the val split), never training images.
- **Before/after** metrics (pretrained → fine-tuned) are reported by the train
  stage and visible in the pipeline benchmark/gate flow.
- The dashboard epoch curve never 500s and renders connected lines across
  missing epochs.

## Design

### 1. Verified-only build + real splits (in `dataset/builder.py`, `cli.py`)

- `DatasetBuilder.build(..., verified_only=True)` becomes the default. The
  unverified Frigate-label fallback is opt-in, reachable only explicitly
  (`verified_only=False`, new CLI flag `--include-unverified`). The pipeline
  `build` stage (`run.py::_step_build`) stays verified-only with no change.
- Dataset layout moves to ultralytics' split-dir convention:

  ```
  datasets/<version>/
  ├── images/{train,val,test}/<sample_id>.jpg
  ├── labels/{train,val,test}/<sample_id>.txt
  ├── dataset.yaml        # train: images/train, val: images/val, test: images/test
  ├── train.txt|val.txt|test.txt   # informational, mirror the split layout
  ├── build.json
  └── manifest.jsonl      # "image" entries now under images/<split>/
  ```

  Ultralytics locates labels by swapping `images` → `labels` with the same
  subdir, so `train: images/train` / `val: images/val` yields honest train/val
  separation. `.txt` split lists are kept (informational) but not wired into
  the yaml. Rationale: checked `check_det_dataset` in ultralytics 8.4 — list
  files resolve incorrectly (entries expanded to per-character globs); split
  directories behave identically to the currently-working `train: images`.
- `build.json`/`BuildSummary.split_counts` unchanged in meaning. Manifest
  "image" values update to the split path. Existing builder/webapp tests that
  assume a flat `images/` + `labels/` layout are updated.

### 2. Train self-reports before/after on the golden set (in `training/trainer.py`)

- After a successful real training run, evaluate two backends on the golden
  dataset (same `GoldenDataset`/`golden_to_examples`/`benchmark_candidate`/
  `UltralyticsBackend` machinery already used by the benchmark stage, at
  `config.training.image_size`):
  - **before**: the candidate's initial weights (the `.pt` the run started
    from),
  - **after**: `weights/best.pt` of the run.
- Print `before → after` (mAP50, recall, latency) and write
  `before_after.json` into the run dir alongside `results.csv`:
  `{"before": {...}, "after": {...}, "delta": {...}}`.
- Graceful when the golden set is missing or has no examples (message +
  `before_after.json` absent; training is not considered failed).

### 3. Benchmark + gate include the latest trained model (in `run.py`, `evaluation/benchmark.py`, `cli.py benchmark`)

- The benchmark step/candidate list additionally includes the **latest trained
  run** discovered under `<data>/training/` (highest `results.csv` mtime with
  an existing `weights/best.pt`), named `<model>-<dataset>` (e.g.
  `yolov8n-v003`). It runs in addition to the configured pretrained candidates.
  Works for the pipeline (`run_pipeline`), the standalone `benchmark` command,
  and the webapp "Re-benchmark + gate" job.
- Because results include both the pretrained candidate (before) and the
  trained model (after), the existing gate evaluates the trained model against
  the baseline `yolov8l` and records a `deployments` ledger row — reporting
  before/after deltas in the benchmark view and ledger.

### 4. Fix the dashboard metric curve (in `webapp/queries.py`, `webapp/static/app.js`)

- `_to_float_or_str`: return `None` when the parsed float is not finite
  (`math.isfinite`). This eliminates `nan`/`inf` from API payloads → no more
  500 from Starlette `allow_nan=False`. (`training_index`/`training_run`
  already filter `None` for best/latest mAP50.)
- `renderTrainingDetail` epoch mapping: treat `null`/`""`/non-numeric as
  missing → push `null` (never `Number(null)=0`).
- `renderLineChart` line series get `spanGaps: true` so the mAP50 and recall
  curves stay connected across missing epochs.
- No comments added to `webapp/` (AGENTS webapp directive).

## Non-goals

- No changes to the golden dataset's own layout or the deploy/Hailo path.
- No JS unit-test framework is added; the curve fix is verified by (a) Python
  tests asserting `/api/training/<run>` returns rows with `None` for
  `nan`/`inf` cells (the 500 root cause) and (b) manual render in the existing
  headless jsdom harness used during diagnosis.

## Testing

1. **builder:** default `verified_only=True` (unverified sample without VLM
   annotations is skipped); `--include-unverified`/`verified_only=False`
   restores the Frigate fallback; dataset.yaml points at split dirs; files land
   under `images/<split>/` and `labels/<split>/`; manifest image path updated;
   split txt files mirror the layout.
2. **trainer:** before/after evaluated and written on a golden fixture
   (backends isolated with a fake `ModelBackend`); graceful when golden
   missing/empty.
3. **benchmark discovery:** latest trained run added as an extra candidate;
   missing training dir no-op; name `<model>-<dataset>`.
4. **webapp queries:** `nan`/`inf`/`""` cells in a pathological `results.csv`
   yield `None` (and `/api/training/<run>` returns 200 with no NaN tokens).
5. Full suite: `.venv/bin/python -m pytest -p no:warnings`
   (must stay green; confirms webapp importorskip guards hold).