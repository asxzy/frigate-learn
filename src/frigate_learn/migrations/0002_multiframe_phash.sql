-- 0002_multiframe_phash.sql
-- Phase 2/3 schema: multi-frame sampling, perceptual hashing, VLM verified flag,
-- discovery windows, and deployment ledger.

-- Multi-frame sampling within one event: event_id is no longer globally unique;
-- uniqueness moves to (event_id, frame_index).
DROP INDEX IF EXISTS ux_samples_event;
ALTER TABLE samples ADD COLUMN frame_index INTEGER NOT NULL DEFAULT 0;
CREATE UNIQUE INDEX IF NOT EXISTS ux_samples_event_frame ON samples (event_id, frame_index);

-- Perceptual hash (dHash) for near-duplicate detection + VLM verified flag.
ALTER TABLE samples ADD COLUMN perceptual_hash TEXT;
ALTER TABLE samples ADD COLUMN verified INTEGER NOT NULL DEFAULT 0;

CREATE INDEX IF NOT EXISTS ix_samples_phash ON samples (perceptual_hash);
CREATE INDEX IF NOT EXISTS ix_samples_verified ON samples (verified);

-- Motion-only windows found by the P9 discovery phase (potential misses).
CREATE TABLE IF NOT EXISTS discovery_windows (
    id            TEXT PRIMARY KEY,
    camera        TEXT NOT NULL,
    start_time    REAL NOT NULL,
    end_time      REAL NOT NULL,
    motion_score  REAL,
    notes         TEXT,
    created_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_discovery_window_camera ON discovery_windows (camera, start_time);

-- Evaluation / deployment ledger (P11/P12). One row per candidate evaluation and
-- deployment decision, so the acceptance gate is auditable.
CREATE TABLE IF NOT EXISTS deployments (
    id             TEXT PRIMARY KEY,
    model_name     TEXT NOT NULL,
    version        TEXT NOT NULL,               -- dataset/golden version evaluated on
    artifact_path  TEXT,
    artifact_hash  TEXT,
    metrics_json   TEXT,                        -- DetectionMetrics as JSON
    verdict        TEXT,                        -- PASS | FAIL | dry-run
    reasons_json   TEXT,                        -- gate reasons
    deployed       INTEGER NOT NULL DEFAULT 0,
    deployed_at    TEXT,
    created_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_deployments_model ON deployments (model_name);