# Multi-Object Per Frame — Implementation Plan

Date: 2026-09-21
Status: implemented
Depends on: current `master` at `8f3efef` plus the in-progress audit/frozen-head working tree.

## Goal

Make the pipeline treat **one frame as one sample** and **one object as one annotation**, so an image containing several objects is collected, verified, built, audited, and evaluated with all boxes — not just the first Frigate event/track.

**Training-domain (this deployment):** the training input is the **cropped region
image** — exactly what Frigate sends to its detector at inference — never the full
camera frame. "One frame" below means one cropped image, and multi-object means
several objects *inside* that cropped image. Multi-object support must not change
the collected/training input to full frames. Crop-mode collection stays one sample
per event (Frigate event boxes are frame-relative, so duplicate crops are skipped,
not geometry-merged); extra objects inside a crop are expressed as additional
annotation rows on the same sample (VLM verifier, crop space).

Current known limitation from `IMPLEMENTATION_REPORT_P0_P1.md`:

> A single image containing several objects is stored as one `Sample` carrying the primary `frigate_label`; the VLM *can* emit multiple objects and the verifier writes each as its own annotation, but the fallbacks used downstream ... assume one box per sample.

## Design principle

```
samples:   one row per physical camera frame/image
annotations: one row per object in that frame
           ─ source = frigate | vlm | human | external
           ─ label, box, confidence, verified
           ─ event_id (new) for Frigate provenance
```

`samples.event_id` remains a **primary/origin event id** for convenience and backwards compatibility, but it must no longer be the identity of a frame. The annotation table is the authoritative object store.

## Phase 1 — Core multi-object support

This phase unblocks the main Frigate → VLM → dataset flow:

1. Persist which Frigate event produced each annotation.
2. When two Frigate events share the same exact frame, merge the second event as an extra Frigate annotation instead of dropping it.
3. Dataset builder emits all relevant boxes for a frame.
4. Golden/eval tests prove multi-box ground truth works end to end.

### Task 1 — Migration: add event provenance to annotations

**Files**
- Create: `src/frigate_learn/migrations/0004_multi_object.sql`
- Modify: `src/frigate_learn/models.py`
- Modify: `tests/test_db.py`

**SQL migration**

```sql
ALTER TABLE annotations ADD COLUMN event_id TEXT;
CREATE INDEX IF NOT EXISTS ix_annotations_event ON annotations(event_id);
CREATE UNIQUE INDEX IF NOT EXISTS ux_annotations_sample_event
    ON annotations(sample_id, event_id)
    WHERE event_id IS NOT NULL;
```

**ORM**

Add to `Annotation`:

```python
event_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
```

**Tests**
- `test_applied_and_pending` expected migrations become `["0001_initial.sql", "0002_multiframe_phash.sql", "0003_review_sync.sql", "0004_multi_object.sql"]` and schema version becomes `4`.
- Add `test_0004_adds_annotation_event_id`: `PRAGMA table_info(annotations)` contains `event_id`.
- Add a test that the partial unique index rejects a second annotation with the same `(sample_id, event_id)` but allows `NULL` event ids.

**Review gate**
- Fresh DB and existing DB both migrate cleanly.
- Existing `annotations.event_id = NULL` rows are unaffected.

### Task 2 — Collector: merge exact same-frame events

**Files**
- Modify: `src/frigate_learn/collection/collector.py`
- Modify: `tests/test_collector.py`
- Possibly modify: `src/frigate_learn/run.py` / `src/frigate_learn/cli.py` summary message

**Changes**

- Add `merged_events: int = 0` to `CollectSummary`.
- Add a method to find an existing exact-frame sample:

```python
def _find_exact_duplicate_sample(self, camera: str, image_hash: str) -> Sample | None
```

- In `_process_event`, when the downloaded image is a duplicate:

  1. If there is an exact-duplicate sample for the same camera + `image_hash`, attempt to merge the current event into that sample.
  2. If an annotation for `(existing_sample.id, event.id)` already exists, skip without duplicating.
  3. Otherwise add an `Annotation(source="frigate", event_id=event.id, label=..., box=..., verified=0)`.
  4. Delete the just-downloaded duplicate image/debug file.
  5. Return `EventOutcome(status="merged")` or a similar distinct status so the summary counts it separately.
  6. Near-phash duplicates (not exact SHA) continue to be skipped as today.

- Update `_prune_existing` to treat an event as existing if it appears either in `samples.event_id` or in `annotations.event_id`. This is necessary for idempotent re-runs after merging.
- Update `_delete_event` for merged events:
  - Delete annotations whose `event_id = :event_id`.
  - If a sample no longer has any annotations and its own `event_id` is the deleted event, delete the sample and image.
  - Otherwise keep the sample but remove only that event's contribution.
- Update `_download_and_store` to count `merged_events`.

**Tests**
- Two events share the same bytes/full-frame image:
  - Result is one `Sample`, two `source='frigate'` annotations, `merged_events == 1`.
  - The second downloaded file is removed; only one image exists.
- Running `collect` again does not add another annotation for the merged event.
- Temporal sampling with the same event and an exact duplicate frame does not add a duplicate annotation for the same event.
- `_delete_event` for a merged event removes only that event's annotation and leaves the other object's annotation/image intact.

**Review gate**
- Full-frame collection remains idempotent.
- Crop/region mode is unchanged: crops are per-event and must not be merged.

### Task 3 — Dataset builder: resolve all boxes per frame

**Files**
- Modify: `src/frigate_learn/dataset/builder.py`
- Modify: `tests/test_dataset_builder.py`

**Changes**

- Replace `_best_annotations` fallback:

```python
if verified_only:
    return verified_anns
return _dedupe_annotations(verified_anns + unverified_anns)
```

- Add `_dedupe_annotations`:
  - Priority: verified annotations first, then by source/created_at.
  - Suppress a lower-priority annotation when it has the same label as an already-selected annotation and IoU > `0.5`.
  - Because boxes are normalized, IoU can be computed directly from `x1/y1/x2/y2` without new dependencies.
  - Do not suppress different-label boxes even if they overlap; a person on a motorcycle is still two objects.

- Change `max_per_class` handling:
  - Count boxes per class, not samples.
  - If a class has reached the cap, filter those boxes out of the emitted label file.
  - If filtering leaves no boxes, skip the image and increment `skipped_cap`.
  - If at least one non-capped box remains, write the image with the remaining boxes.

- Manifest `"labels"` must contain every emitted label, not only the first.

**Tests**
- `verified_only=False` with two different unverified Frigate boxes on one sample produces a label file with two YOLO lines.
- A verified VLM box plus an overlapping unverified Frigate box produces one label line after dedupe.
- Two different-label overlapping boxes are both kept.
- `max_per_class` counts all boxes of that class; a skipped class does not necessarily skip the whole image if another class remains.
- Existing verified-only behavior stays green.

**Review gate**
- Build output remains deterministic.
- YOLO label files can contain multiple lines; `dataset.yaml`/manifest remain consistent.

### Task 4 — Golden/eval multi-box tests

**Files**
- Modify: `scripts/build-golden.py` (if needed)
- Modify/Add: `tests/test_golden.py` or `tests/test_benchmark.py`

**Changes**
- `build-golden.py` already reads all YOLO lines; remove any reliance on `yolo_lines[0]` for class counts/decision.
- Add a test that `GoldenDataset.add_image(..., labels=[two YoloLine objects])` round-trips and `golden_to_examples()` returns two `GroundTruth` objects.
- Add a benchmark-level test (or metrics test) that a model predicting both boxes is scored against both ground truths and a model predicting only one is not 100% recall.

**Review gate**
- Multi-object golden images are validated and benchmarked with all GT boxes.

## Phase 2 — Proper audit/adapter and user-facing filters

After Phase 1, training from VLM-verified annotations is multi-object correct. Phase 2 makes the audit/pseudo-labeling path and label filtering equally correct.

### Task 5 — `FrigateDatabaseAdapter` iterates annotations, not primary sample boxes

**Files**
- Modify: `src/frigate_learn/audit/adapter.py`
- Modify: `tests/test_audit_adapter_db.py`
- Modify: callers if they assume a one-to-one sample-to-object relationship.

**Changes**
- Query from `annotations JOIN samples`, filtering `annotations.source = 'frigate'` by default (or by an explicit source parameter).
- Use the annotation row as the object:
  - `FrigateObject.sample_id` must be unique per object, e.g. `annotation.id` or `f"{sample.id}:{annotation.id}"`.
  - Keep the original sample id in `extra["sample_id"]`.
  - Add `extra["event_id"]`, `extra["review_id"]`, `extra["annotation_id"]`.
- `ontology()` should derive from `annotations.label` rather than only `samples.frigate_label`.
- Duplicate sample ids from multiple annotations are not a manifest error; the adapter must not reject them.

**Tests**
- One sample with two Frigate annotations yields two `FrigateObject`s with distinct `sample_id`, same `image_path`, and the correct per-object boxes.
- Existing one-annotation cases still pass.

### Task 6 — Audit export groups positives by frame

**Files**
- Modify: `src/frigate_learn/audit/pipeline.py`
- Modify: `src/frigate_learn/audit/export.py`
- Modify/Add: `tests/test_audit_cli_e2e.py` or `tests/test_audit_pipeline.py`

**Changes**
- Thread a stable `frame_key` through `FrigateObject.extra` (the original sample id or image path is fine).
- In `write_positive_sample`:
  - Use the frame key as the image filename so one physical frame is written once.
  - Append each accepted object's YOLO line to the shared `.txt` label file instead of overwriting it.
  - Write per-object masks using object-unique filenames if mask support is required.
- Preserve per-object provenance records in `provenance.jsonl`.

**Tests**
- Two accepted objects in the same frame produce one image file, one label file with two lines, and two provenance rows.
- A single-object frame still matches the old output shape.

### Task 7 — Label filters/views use annotations

**Files**
- Modify: `src/frigate_learn/annotation/verifier.py`
- Modify: `src/frigate_learn/dataset/builder.py`
- Modify: `src/frigate_learn/webapp/queries.py`
- Modify: `tests/test_verifier.py`, `tests/test_dataset_builder.py`, `tests/test_webapp.py`

**Changes**
- Where a `labels` filter should mean "frames containing this object label", query through `annotations` instead of `samples.frigate_label`.
- `Verifier.verify(labels=[...])` should select samples that have an annotation with that label (or a Frigate annotation with that label).
- `DatasetBuilder._query_samples(labels=[...])` should similarly include samples whose annotations contain the requested label, not just the primary `frigate_label`.
- Webapp sample list can keep the primary label for display, but any API filter that filters by label should consider annotations.

**Tests**
- A sample whose primary label is `person` but also has a `car` annotation appears when filtering by `car` in verify/build paths.
- Webapp label-based filtering behaves consistently.

## Edge cases and decisions

1. **Exact SHA merge only.** Near-phash duplicates are still skipped, not merged, because boxes from slightly different frames may be stale. This can be revisited with a configurable merge policy later.
2. **`samples.verified` remains “VLM processed this frame”.** Per-box acceptance lives in `annotations.verified`. Do not overload sample-level verified to mean “all boxes are correct”.
3. **Frigate vs VLM duplicate labels.** IoU dedupe in the builder prevents the same physical object from being emitted twice. VLM is authoritative whenever a verified annotation exists.
4. **Refresh semantics.** Phase 1 implements idempotent merge and event-level deletion, but full `--refresh` semantics for merged frames should be covered by focused tests before relying on it in production.
5. **Review sync with multiple reviews per frame is out of scope.** A frame may be part of multiple Frigate review segments after merging. `samples.review_id` remains the primary review; review-sync still operates on the primary sample review. A `sample_reviews` join table can be a follow-up if multi-review confirmation is needed.
6. **Crop/region mode keeps one sample per event.** The training input is the cropped image; Frigate event boxes are frame-relative, so duplicate crops across events are skipped, never geometry-merged. Multi-object inside a crop is expressed by multiple annotation rows on the same sample (VLM verifier writes each object it sees in crop space; the dataset builder emits every box).

## Test suite expectations

- All existing tests must stay green except intentional updates to migration counts, adapter object IDs, and summary fields.
- New coverage should include at least:
  - DB migration `0004`
  - Collector same-frame merge/idempotency/delete
  - Builder multi-annotation resolution/dedupe/cap
  - Golden/eval multi-GT
  - Audit adapter multiple objects per sample
  - Audit export grouped positives
  - Label filtering through annotations

## Out of scope (follow-ups)

- Full normalized `frames`/`sample_events` schema.
- Multi-review human confirmation mapping.
- Per-object quality/triage in the UI.
- Merging near-duplicate frames based on object motion.
- Changing the VLM prompt to consume/confirm existing Frigate boxes in one pass.

## Suggested commit sequence

1. `db: add annotation event_id for multi-object frame provenance`
2. `collect: merge exact same-frame Frigate events into one sample`
3. `dataset: emit all per-frame annotations and cap by box count`
4. `golden: test multi-object ground truth through eval`
5. `audit: iterate annotations as objects`
6. `audit: group positive exports by physical frame`
7. `filters: route label filters through annotations`

