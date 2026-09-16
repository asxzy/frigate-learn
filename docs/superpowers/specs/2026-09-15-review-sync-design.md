# Frigate Review state sync — human confirmation inside Frigate (design)

Date: 2026-09-15
Status: implemented (single PR)

## Question

The pipeline used to download every image and rely on a separate labeling
surface (VLM on the training host + a local Triage lightbox). Could the
operator instead label/confirm data *inside Frigate* using its built-in
Review feature?

## Feasibility findings (checked against Frigate 0.18 source)

- Frigate's Review UI is a **binary human-confirmation surface**: watch a
  motion segment, then mark it reviewed/unreviewed. The only write API is
  `POST /api/reviews/viewed` (`{ids, reviewed}`), persisted per user in
  `UserReviewStatus.has_been_reviewed` (`frigate/api/review.py`, master).
- There is **no box drawing, no per-detection accept/reject, and no way to
  store custom labels** in Frigate. The closest per-event feedback endpoint,
  `PUT /api/events/{id}/false_positive`, uploads the sample to the Frigate+
  cloud (`PLUS_API_KEY`) for Frigate's own model — unusable as a local
  ground-truth channel.
- Therefore box/label annotation cannot happen inside Frigate. What CAN happen
  inside Frigate is the **human confirmation** step: "this event was real" vs
  "noise". That was previously done in the frigate-learn Triage lightbox.

## Decision

Keep the VLM verifier as the box-labeling authority (labels still produced on
the training machine — a hard constraint: nothing heavy runs on the Frigate
VM). Move the *binary* human signal into Frigate's native Review UI and ingest
it back into the pool:

- New stage `review-sync` (CLI `frigate-learn review-sync`, pipeline step
  `review-sync`, runs right after `collect`): pulls `has_been_reviewed` per
  review segment via the existing `list_reviews` adapter and writes it onto
  every local sample whose `review_id` belongs to that segment.
- New columns on `samples`: `frigate_reviewed` (`NULL` unknown, `0` marked
  unreviewed, `1` marked reviewed) and `reviewed_at` (first observation time;
  cleared on a flip to unreviewed).
- `collect` also stamps the flag at insert time when the review already
  carries it, so freshly collected data is review-aware immediately.
- Policy (configurable): with `collection.review_auto_useful` (default true),
  samples whose segment is reviewed and whose local verdict is unset are
  auto-triaged to `quality=useful`. The remap **only adds** useful and never
  overwrites an explicit verdict; flipping a review back to unreviewed clears
  the flag but does not remove the auto-set quality (a human's explicit
  dashboard verdict still wins).
- Webapp: Quality view shows Frigate-reviewed/unreviewed counts; Triage gains
  a reviewed/unreviewed filter and a "R" badge; the lightbox shows the review
  state pill. `review-sync` appears as a pipeline step in the Overview strip.

## Caveats

- Frigate stores the reviewed flag **per user** (joined on the authenticated
  username). The token user must be the same account that marks reviews in the
  Frigate UI, otherwise the pipeline reads `False` for everything.
- `has_been_reviewed` semantics = "a human watched this segment", not "the
  boxes are correct". It is a confirmation signal, not a box-level label; the
  VLM verifier still owns box correctness. This is why the raw flag and the
  quality mapping are separate columns/settings.
- Design-rule update: the collector still owns its state (dedup, progress);
  `has_been_reviewed` is now an *advisory human-label input*, never used for
  progress tracking.

## Tests

- `tests/test_review_sync.py`: flag storage + auto-useful, idempotency,
  explicit-verdict preservation, `auto_useful=False`, flip semantics, job
  ledger, transport-failure path.
- Collector: `frigate_reviewed`/`reviewed_at` stamped from the review flag.
- `test_db.py`: 0003 migration adds the two columns.
- `test_config.py`: `review_auto_useful` default + parsing.
- `test_run.py`: PIPELINE ordering + `review-sync` step executes.
- `test_cli.py`: `review-sync` output and `--no-auto-useful`.
- `test_webapp.py`: `/api/samples?reviewed=1|0`, sample fields, `/api/quality`
  reviewed counts, overview keys.