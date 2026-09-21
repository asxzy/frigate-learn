# Dataset audit & pseudo-labeling (`frigate-learn audit`)

Curates an existing Frigate detection dataset into YOLO training material
without touching the Frigate vocabulary. For every stored detection crop the
pipeline runs SAM 3.1 (MLX) as an independent segmenter, computes
deterministic geometry checks, renders a reconciliation image, and asks an
oMLX VLM for an *independent* reading of the crop. The VLM is blind: it only
sees the crop with the Frigate box drawn on it, never any class label, and
reports 1) whether the box covers the object and 2) its own label for the
object. Agreement is computed afterwards by the decision engine: SAM's class
vs the VLM's class, plus deterministic SAM↔Frigate-bbox alignment gates.
Labels and classes always come from the Frigate vocabulary; SAM and the VLM
can only confirm or reject them.

## Commands

```bash
.venv/bin/frigate-learn audit run --config config.yaml --limit 200
.venv/bin/frigate-learn audit run --config config.yaml --resume
.venv/bin/frigate-learn audit stats audit/
.venv/bin/frigate-learn audit-dataset run --config config.yaml   # alias for audit run
```

`run` options: `--input`, `--output`, `--training-output`, `--resume`,
`--limit N`, `--class <name>` (repeatable), `--sam-only` (never calls the
VLM), `--stats-json`, `--no-hard-negatives`, `--negative-samples N`.
`stats <audit-dir>` rebuilds the aggregate report from existing
`decision.json` files without re-running any model.

Audit runs register a job in the dashboard's Jobs drawer and stream a live
log tail (start / per-sample outcomes / finish summary) into it as they run;
the same applies to `collect` and `verify` runs.

## Inputs (the adapter auto-detects, in order)

* a SQLite database file: `frigate-learn audit run --input data/frigate_learn.db`
* a dataset root containing `frigate_learn.db` or `frigate.db` (this repo's
  `<data>/` layout); images resolve against `<root>/images`
* a manifest root with `annotations/manifest.json` or `manifest.json` — the
  explicit interchange format (see `adapter.py` for the schema and every
  coordinate convention supported: `xyxy|xywh|cxcywh`, normalized or not,
  image- or frame-relative)

Boxes are normalized to absolute `xyxy` pixels in crop space by the adapter
and clamped to the crop. Everything else in the pipeline speaks crop pixels.

## Config (`audit:` section)

```yaml
audit:
  pipeline_version: 1            # bump to invalidate all caches
  max_consecutive_vlm_errors: 5  # abort when the VLM server stays down
  input: "data"
  output: "audit"                # per-sample caches + decisions + provenance
  training: "training_audit"     # positive/ + hard_negative/ + classes.txt
  hard_negatives_enabled: false  # export VLM-confirmed false positives
  negative_sampling_enabled: false
  samples_per_crop: 0            # optional in-crop background negatives
  models:
    sam:
      checkpoint: ""             # "" -> weights from the HF repo
      load_from_hf: true
      hf_repo: "mlx-community/sam3-image"   # native sam3_mlx format; transformers-format quantized repos (sam3-8bit/4bit) are rejected
      quantize_bits: 0           # 0 = keep precision; 4/8 = in-memory mx.quantize (breaks sam3_mlx decoder heads upstream)
      resolution: 1008
      confidence_threshold: 0.5
      candidate_classes: []      # extra candidates; Frigate class always asked
    vlm:
      base_url: "http://127.0.0.1:8080/v1"   # oMLX OpenAI-compatible endpoint
      model: ""
      temperature: 0.0
      timeout_seconds: 180     # hard wall-clock deadline per attempt (slow/stalled servers cannot hang a run)
      max_retries: 2
      json_mode: true
  geometry:
    min_mask_area: 0             # all gates opt-in (record-only by default)
    min_bbox_iou: 0.0
    mask_bbox_ratio_min: 0.0
    mask_bbox_ratio_max: 9999.0
    min_mask_frigate_containment: 0.0
```

## Pipeline (one object at a time)

1. load the existing crop (never re-crops, never loads full frames)
2. SAM 3.1 (cached): mask, refined bbox, class hypothesis from the Frigate
   vocabulary only
3. deterministic geometry metrics (IoU, edges, mask/bbox ratios) — this is
   where the SAM segment is checked against the Frigate bbox
4. render the full-overlay reconciliation image (cached, for humans) and a
   separate blind VLM input (crop + Frigate box only, no SAM box, no mask,
   no label text)
5. VLM independent verdict: `bbox_covers_object` (bool) and `class_label`
   (its own label for the object, `""` when nothing is present). The model
   is prompted with the label vocabulary as the answer set — never with the
   expected class — and its label is canonicalized (common synonyms
   resolved) at aggregation time. Malformed or transport-failed answers are
   DROP/PENDING, never a guess.
6. the one conservative rule: KEEP only when SAM is valid, the VLM verdict
   passes, the independent SAM and VLM class readings agree
   (`class_mismatch` otherwise), and any configured geometry gates pass
7. write provenance; export positives (images/labels/masks) and, when
   enabled, hard negatives (crops where the VLM found no object = confirmed
   false positives)

Accepted samples land in `<training>/positive/{images,labels,masks}/`,
`provenance.jsonl` and `classes.txt` (class id = line index). Hard negatives
and optional random backgrounds land in `<training>/hard_negative/`.

## Determinism, caches and resume

Every artifact is keyed by a stable sample hash (image content + class +
bbox + SAM key + VLM key + pipeline version). The SAM key is derived from the
model config (weight source, resolution, confidence threshold, candidate
classes) and the VLM key from the endpoint base URL, model, temperature and
json mode — so changing any of those invalidates the matching caches and
`--resume` re-decides affected samples. A rerun reuses SAM, reconciliation and
VLM caches; `--resume` skips samples with a valid final KEEP/DROP decision and
retries PENDING ones. Decisions written by the *old* biased VLM schema (six
self-confirming booleans that were told the expected class) are detected and
refreshed on the next `--resume`: those samples get a fresh blind VLM reading
without re-running SAM. A cached `sam_failure` DROP recorded while `sam3_mlx`
was missing (provenance `vlm.error_kind == "import"`) is refreshed on the next
`--resume` once the backend is importable — install `sam3_mlx` and resume to
re-attempt those samples; model-driven failures (`error_kind == "run"`) stay
final because they are deterministic for the given weights. SAM candidate
selection is deterministic (Frigate-class tie-break, documented in `sam.py`).

## Degradation without extras/server

With `sam3_mlx`/`mlx` absent the `MlxSam3Teacher` raises a clear
`SamTeacherError`, every sample is dropped with reason `sam_failure`, and
the run still writes decisions/provenance and exits 0 — the CLI never
crashes on a bare `pip install -e .[dev]`.

## End-to-end tests (no deployment)

`tests/test_audit_cli_e2e.py` drives the real click group over real on-disk
datasets (manifest and SQLite layouts) with a deterministic SAM stand-in and
a local HTTP fake of the oMLX endpoint: it asserts decisions, caches,
`--resume` (including PENDING retries), class filters, exports, hard
negatives, `audit stats`, and the no-extras degradation path.
