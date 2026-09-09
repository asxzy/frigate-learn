-- 0001_initial.sql
-- Phase 1 schema: samples, annotations, jobs, collection failures.

CREATE TABLE IF NOT EXISTS samples (
    id            TEXT PRIMARY KEY,             -- stable sample id (uuid4 hex)
    camera        TEXT NOT NULL,
    timestamp     REAL NOT NULL,                -- event start_time (unix)
    event_id      TEXT,                         -- Frigate event/object id
    review_id     TEXT,                         -- Frigate review segment id
    image_path    TEXT,                         -- clean snapshot, training image
    debug_image_path TEXT,                      -- annotated snapshot (debug only)
    image_hash    TEXT,                         -- sha256 of training image
    source        TEXT NOT NULL DEFAULT 'frigate',
    frigate_label TEXT,
    frigate_score REAL,
    frigate_x1    REAL,
    frigate_y1    REAL,
    frigate_x2    REAL,
    frigate_y2    REAL,
    quality       TEXT,                         -- browser verdict: useful|bad|duplicate|ignore
    status        TEXT NOT NULL DEFAULT 'collected',
    created_at    TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_samples_event ON samples (event_id);
CREATE INDEX IF NOT EXISTS ix_samples_camera ON samples (camera);
CREATE INDEX IF NOT EXISTS ix_samples_timestamp ON samples (timestamp);
CREATE INDEX IF NOT EXISTS ix_samples_review ON samples (review_id);
CREATE INDEX IF NOT EXISTS ix_samples_hash ON samples (image_hash);
CREATE INDEX IF NOT EXISTS ix_samples_label ON samples (frigate_label);

CREATE TABLE IF NOT EXISTS annotations (
    id          TEXT PRIMARY KEY,
    sample_id   TEXT NOT NULL REFERENCES samples (id),
    source      TEXT NOT NULL,                  -- frigate|vlm|yolov8l|human|external
    label       TEXT NOT NULL,
    x1          REAL,
    y1          REAL,
    x2          REAL,
    y2          REAL,
    confidence  REAL,
    verified    INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_annotations_sample ON annotations (sample_id);

CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    type          TEXT NOT NULL,               -- collect|annotate|build_dataset|train|evaluate|benchmark
    status        TEXT NOT NULL,               -- running|finished|failed
    started_at    TEXT,
    finished_at   TEXT,
    error         TEXT,
    metadata_json TEXT
);

CREATE INDEX IF NOT EXISTS ix_jobs_status ON jobs (status);
CREATE INDEX IF NOT EXISTS ix_jobs_type ON jobs (type);

-- Per-item failure ledger. A failed event is never silently dropped; it is
-- recorded here and retried on the next collect run.
CREATE TABLE IF NOT EXISTS collection_failures (
    id            TEXT PRIMARY KEY,
    event_id      TEXT NOT NULL,
    review_id     TEXT,
    camera        TEXT,
    error         TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_failures_event ON collection_failures (event_id);