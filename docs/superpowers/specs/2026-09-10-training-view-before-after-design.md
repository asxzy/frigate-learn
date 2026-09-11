# Training view shows golden before/after alongside a pinned epoch curve

Date: 2026-09-10

## Problem

1. **Misleading epoch-1 metrics.** `train yolov8n` fine-tunes the COCO-pretrained
   `yolov8n.pt` (80 classes) on an 8-class dataset (`trainer.py:176`). The
   80-class detection head cannot transfer to 8 classes, so ultralytics
   re-initializes it; the pretrained backbone is kept. The Train-view table and
   curve therefore show a *fresh head* learning during training (epoch-1 val
   mAP50 ≈ 0.028, climbing to 0.83), which reads as "a pretrained model losing
   performance", even though the pretrained model's real quality only appears
   in the golden-set **before** metric (0.333) that the dashboard never shows.
2. **Before/after/during are not together.** The pipeline already computes
   `before_after.json` (pretrained vs fine-tuned on the golden set) but nothing
   renders it. Users end up comparing the epoch curve (val set, during training)
   against the pretrained/golden numbers they expect, and the two look
   contradictory.
3. **Epoch curve can render off the plot box.** The metric curve uses uPlot's
   auto y-scale (`scales.y = {}`), which is the only pin-point where a run that
   overflows its bounds (bad/legacy data, degenerate defaults) can draw lines
   outside the plot area.

## Goals

- The Training run detail shows **before** (pretrained, golden-set) and **after**
  (fine-tuned, golden-set) mAP50 explicitly, right where the during-training
  epoch curve lives, so the pretrained baseline is never "missing".
- The during-training curve (val set) is annotated so it cannot be confused with
  the golden-set numbers.
- The epoch curve y-axis is pinned to `[0, 1]`; mAP50/recall lines cannot leave
  the plot box regardless of data quirks.
- Runs without `before_after.json` (v001/v003) degrade to "not recorded"
  without errors.

## Design

### 1. Backend: `before_after` in the run detail (`webapp/queries.py`)

`training_run()` gains one read-only field. The run dir already resolves via
`config.resolve(data.root, TRAINING_DIR, run)`; `before_after.json` sits beside
`results.csv`.

- If `before_after.json` is a parseable JSON object, return an *extracted*
  shape — never the raw file:
  `{"before": {"map50": f, "recall": f, "latency_ms": f}, "after": {...}}` with
  finite numbers only (non-finite or missing members dropped to `None`).
- Else return `"before_after": null`.
- No caching; artifacts read in place, consistent with the rest of `queries.py`.

### 2. Frontend: cards + reference lines + pinned scale (`webapp/static/app.js`)

In `renderTrainingDetail()`:

- **Cards.** After the existing "Best mAP50" / "Latest mAP50" / "Weights" cards,
  add "Golden before" and "Golden after" (values from `data.before_after`,
  formatted with `num()`, "—" when null), plus a sub `"pretrained · 0.333"` /
  `"fine-tuned · 0.333"` hint, and a "Δ +0.000" delta when both are present.
- **Curve data.** Keep `xs/map/rec` as-is. When `before_after` is present, add
  two constant series: `before` (`xs` filled with `before.map50`) and `after`
  (`xs` filled with `after.map50`).
- **Pinned y-scale.** `renderLineChart()` becomes
  `scales: { x: { time: false }, y: { auto: false, range: [0, 1] } }` so the
  0–1 metric domain is fixed and deterministic. `renderScatter()` (benchmark
  mAP50 vs latency) is left on auto-scale — its y is mAP50 too, but it lives on
  a different axis pairing; it is out of scope for pinning.
- **Reference series styling.** before/after series render dashed
  (`dash: [6, 4]`), muted colors (e.g. `#8b949e` before, `#d29922` after),
  `spanGaps: true`, `points: { show: false }`, widths 1, so the two 0–1 epoch
  curves stay visually dominant. Legend shows all series (cursor values
  included); that is informative, not clutter.
- **Caption.** "Epoch curves — val set per epoch · dashed = golden-set
  before/after mAP50". A single clarifying line, no per-series prose.

### 3. Data privacy / no secrets

No config, VLM key, or env data touches the new field; it is pure artifact
reads (same as the CSV path).

## Testing

- `tests/test_webapp.py`:
  - `training_run` returns parsed `before_after` when the file exists and is
    valid (build a fixture run dir in `tmp_path`, point config at it).
  - returns `null` when the file is absent.
  - returns `null` (no raise) when the file is corrupt JSON or contains
    non-finite numbers.
- jsdom harness (repo-external, used during E2E): render
  `renderTrainingDetail("#view-training", "yolov8n-v004")` against real
  fixtures; assert zero render errors, `scales.y.max === 1`, dashed reference
  series present with constant `0.3333…` data, and the run-list view still
  renders 3 runs.
- `ruff` clean on changed files; full pytest suite stays green.

## Out of scope

- Pinning the benchmark scatter (`renderScatter`) y-axis.
- Re-running training for `v001`/`v003` so they gain `before_after.json`.
- Explaining epoch-wise mAP50-95 vs mAP50 in the metric table.
- Any `cli.py`, `training/trainer.py`, or pipeline changes.