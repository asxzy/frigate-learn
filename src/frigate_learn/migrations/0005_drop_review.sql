-- 0005_drop_review.sql
-- Review concept removed: Frigate's per-user has_been_reviewed flag
-- (frigate_reviewed / reviewed_at) and the review-segment linkage
-- (review_id on samples and collection_failures) are no longer stored.
-- Indexes referencing the dropped columns must go first (SQLite rule).

DROP INDEX IF EXISTS ix_samples_reviewed;
DROP INDEX IF EXISTS ix_samples_review;

ALTER TABLE samples DROP COLUMN frigate_reviewed;
ALTER TABLE samples DROP COLUMN reviewed_at;
ALTER TABLE samples DROP COLUMN review_id;

ALTER TABLE collection_failures DROP COLUMN review_id;