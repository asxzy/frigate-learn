-- 0004_multi_object.sql
-- Multi-object per frame: the annotations table becomes the authoritative
-- object store. annotations.event_id records which Frigate event produced
-- each annotation, so several events sharing one exact frame can each
-- contribute a box to the same samples row.
--
-- event_id: NULL for VLM/human/external annotations; set for Frigate rows.
-- The partial unique index keeps one Frigate annotation per (sample, event).
ALTER TABLE annotations ADD COLUMN event_id TEXT;

CREATE INDEX IF NOT EXISTS ix_annotations_event ON annotations(event_id);

CREATE UNIQUE INDEX IF NOT EXISTS ux_annotations_sample_event
    ON annotations(sample_id, event_id)
    WHERE event_id IS NOT NULL;