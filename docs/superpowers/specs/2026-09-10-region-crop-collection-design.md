# Region-crop collection — Frigate I/O matching (design)

Date: 2026-09-10
Status: approved (chat), 2026-09-10

## Principle

This repo's trained model is deployed into Frigate and consumes/dispenses
exactly Frigate's detector I/O. Therefore collected training data must match
what Frigate sends the detector and what it asks the detector to produce:

- *Input:* region crops of the camera frame, resized toward the model input
  size (`training.image_size`), i.e. object-zoomed crops, not full frames.
- *Output:* detection boxes normalized to that crop's coordinate space (same
  space as the deployed model's output and Frigate's box normalisation).

The full frame is never stored. The dataset builder performs no pixel
transforms; the crop is already the training image.

## Collection

- `frigate/snapshots.py`: new `region_crop_params(height)` →
  `{crop:1, bbox:0, timestamp:0, height:<H>, download:1}`. Frigate 0.18 honors
  `crop`/`height`/`bbox`/`timestamp` for completed events (0.18 release notes).
- `frigate/client.py`: new `download_region_crop(event_id, path, height,
  timestamp=None)` hitting the same `/api/events/{id}/snapshot.jpg` endpoint.
  `download_clean_snapshot` is retained for the non-crop path.
- `collection/collector.py` `_process_event`: when region-crop is enabled,
  download the cropped JPEG only; the full frame never touches disk.
  - Events with missing or degenerate boxes are skipped and counted (a sane
    box is required for a useful crop).
  - No Frigate annotation row is created in crop mode (the event box describes
    the full frame, not the crop, so it would be a lying label). The event box
    is still kept on `Sample.frigate_*` for traceability.
  - Debug annotated snapshot (if `keep_annotated_snapshots`) becomes an
    annotated *crop* (`crop=1&bbox=1`), never a full-frame image.
  - Dedup hashing runs on the stored crop (unchanged code path).
  - Temporal sampling (`sampling.enabled`) is force-disabled in crop mode:
    the event box is a single-frame artifact and a stale box yields wrong
    crops on resampled frames.

## Labels / ground truth

No changes to the VLM verifier: it reads `sample.image_path` (the crop) and
returns boxes normalized to that image. Verified VLM rows are the ground truth
and the only labels in crop mode.

## Splits

Deterministic sha-256 based splitter (`dataset/splits.py`), unchanged:
train + val + test. `val` is retained for ultralytics checkpoint selection.

## Config / CLI

- `collection.region_crop: bool`, default `true`.
- `collection.region_crop_height: int`, default = `training.image_size` (320,
  pixel density matches the model input); configurable upward if the VLM
  struggles on small crops.
- `frigate-learn collect` gains `--no-region-crop` to force full-frame mode.

## Accepted approximation

`crop=1` returns the tight object box (rectangular). Frigate feeds the model a
square region with context; the box crop is the accepted stand-in per
decision (server-side crop chosen over client-side square-region squaring).
Train-time letterbox handles the remainder. If square-region fidelity is
needed later, revisit with a client-side square.

## Risks

- Frigate `/api/events/{id}` boxes historically showed zero-area rows in the
  DB; if the event boxes are degenerate server-side, crop mode yields little.
  Mitigation: gate on a sane box, count skips, surface in the summary.
- `crop`/`height` params may be ignored by non-0.18 Frigate builds; verified
  against the target 0.18 deployment via a single `collect`.

## Tests (TDD)

- `frigate/snapshots.py` param builder shape.
- `client.download_region_crop` URL/params (respx).
- Collector crop mode: stores crop not full frame, no Frigate annotation row,
  skips degenerate boxes, single-frame on sampling, dedup works.
- Collector non-crop mode regression: unchanged behavior.
- Config parsing for the two new fields.
- CLI `--no-region-crop` override.

## Out of scope

- Golden dataset population should later ingest region crops (same geometry
  as runtime so the gate benchmark stays representative). No production golden
  population path exists yet; documented here as a follow-up.