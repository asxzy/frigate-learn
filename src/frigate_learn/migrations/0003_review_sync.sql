-- 0003_review_sync.sql
-- Frigate Review state sync: mirrors the per-review has_been_reviewed flag onto
-- the samples that belong to each review segment, so the operator's Frigate
-- Review UI become the binary human-confirmation channel for the pipeline.
--
-- frigate_reviewed: NULL (unknown) | 0 (marked unreviewed) | 1 (marked reviewed)
-- reviewed_at:     first/observing timestamp for the reviewed state (cleared on
--                  an unreviewed flip).
ALTER TABLE samples ADD COLUMN frigate_reviewed INTEGER;
ALTER TABLE samples ADD COLUMN reviewed_at TEXT;

CREATE INDEX IF NOT EXISTS ix_samples_reviewed ON samples (frigate_reviewed);