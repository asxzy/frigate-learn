# Region-crop collection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collect training data from Frigate as server-side region crops (`crop=1&bbox=0&height=<H>` snapshots) so stored images match what Frigate feeds the detector; VLM labels become the ground truth; the full frame is never stored.

**Architecture:** New snapshot params + client download methods (Frigate HTTP layer) → new `collection.region_crop`/`region_crop_height` config → collector downloads only crops, skips events with degenerate boxes, creates no Frigate annotation rows in crop mode, and forces single-frame sampling → CLI `--no-region-crop` override. Dataset builder and VLM verifier are intentionally unchanged: they already operate on whatever `sample.image_path` points to.

**Tech Stack:** Python 3.11, click, httpx, pytest (+ respx for client transport tests), Pillow (unused here — crop is server-side).

## Global Constraints

- No comments in production code except `# ---` section banners (repo convention).
- `frigate/` is the only place that knows Frigate URLs/params. Nothing else may build request params.
- Frigate 0.18 snapshot endpoint: `GET /api/events/{id}/snapshot.jpg` honors `crop`, `bbox`, `timestamp`, `height` for in-progress AND completed events (per 0.18 release notes).
- `collection.region_crop` defaults to `True`. Existing collector tests must opt out explicitly (`config.collection.region_crop = False`) so the legacy full-frame path keeps regression coverage.
- Sampling (`sampling.enabled`) is force-disabled in crop mode: the event box is single-frame and stale boxes produce wrong crops.
- Do NOT run `git commit` steps unless the user explicitly asks — the operator has pending untracked files and does not want auto-commits.

---

### Task 1: Frigate snapshot params + client download methods

**Files:**
- Modify: `src/frigate_learn/frigate/snapshots.py`
- Modify: `src/frigate_learn/frigate/client.py:19-26` (imports) and `:380-387` (annotated snapshot section)
- Test: `tests/test_frigate_client.py`

**Interfaces:**
- Produces (consumed by Collector, Task 3):
  - `region_crop_params(height: int) -> dict[str, Any]` from `frigate_learn.frigate.snapshots`
  - `annotated_region_crop_params() -> dict[str, Any]` from `frigate_learn.frigate.snapshots`
  - `FrigateClient.download_region_crop(event_id: str, output_path, height: int, timestamp: float | None = None) -> str`
  - `FrigateClient.download_annotated_crop(event_id: str, output_path) -> str`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_frigate_client.py`)

```python
def test_download_region_crop_params(mock_api, tmp_path):
    payload = b"\xff\xd8\xff\xe0fakecropjpeg"
    mock_api.get("/api/events/e1/snapshot.jpg").mock(
        return_value=Response(200, content=payload)
    )
    client = make_client()
    dst = client.download_region_crop("e1", tmp_path / "crop" / "e1.jpg", height=320)
    assert Path(dst).read_bytes() == payload
    params = mock_api.calls[0].request.url.params
    assert params["crop"] == "1"
    assert params["bbox"] == "0"
    assert params["timestamp"] == "0"
    assert params["height"] == "320"
    client.close()


def test_download_region_crop_timestamp_param(mock_api, tmp_path):
    mock_api.get("/api/events/e1/snapshot.jpg").mock(
        return_value=Response(200, content=b"x")
    )
    client = make_client()
    client.download_region_crop("e1", tmp_path / "e1.jpg", height=640, timestamp=123.0)
    assert mock_api.calls[0].request.url.params["timestamp"] == "123.0"
    client.close()


def test_download_annotated_crop_params(mock_api, tmp_path):
    mock_api.get("/api/events/e1/snapshot.jpg").mock(
        return_value=Response(200, content=b"x")
    )
    client = make_client()
    client.download_annotated_crop("e1", tmp_path / "e1-debug.jpg")
    params = mock_api.calls[0].request.url.params
    assert params["crop"] == "1"
    assert "bbox" not in params  # annotated crop keeps overlays (no bbox=0)
    client.close()
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_frigate_client.py -p no:warnings -k "region_crop or annotated_crop"`
Expected: FAIL with `AttributeError: ... has no attribute 'download_region_crop'`

- [ ] **Step 3: Implement** — `src/frigate_learn/frigate/snapshots.py`, after `annotated_snapshot_params`:

```python
def region_crop_params(height: int) -> dict[str, Any]:
    """Server-side region crop: clean (no overlay) snapshot cropped to the event box."""
    return clean_snapshot_params(crop=1, height=height)


def annotated_region_crop_params() -> dict[str, Any]:
    """Annotated (overlay) region crop used for debug snapshots."""
    return {"download": 1, "crop": 1}
```

Add both names to the module `__all__`.

- `src/frigate_learn/frigate/client.py`, extend the snapshots import block:

```python
from .snapshots import (
    annotated_region_crop_params,
    annotated_snapshot_params,
    annotated_snapshot_url,
    clean_snapshot_params,
    clean_snapshot_url,
    motion_activity_url,
    region_crop_params,
    review_preview_url,
)
```

Add methods after `download_event_snapshot` (line ~387):

```python
def download_region_crop(
    self, event_id: str, output_path: Any, height: int, timestamp: float | None = None
) -> str:
    """Download a server-side region crop (clean, no overlays) for an event.

    ``height`` is the requested output height in pixels; ``crop=1`` crops to
    the event bounding box. See snapshots.py.
    """
    params = region_crop_params(height)
    if timestamp is not None and timestamp > 0:
        params["timestamp"] = timestamp
    return self._download(
        clean_snapshot_url(event_id),
        output_path,
        params=params,
        timeout=self.timeout,
    )

def download_annotated_crop(self, event_id: str, output_path: Any) -> str:
    """Download an annotated region crop (debug only)."""
    return self._download(
        annotated_snapshot_url(event_id),
        output_path,
        params=annotated_region_crop_params(),
        timeout=self.timeout,
    )
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_frigate_client.py -p no:warnings`
Expected: PASS (all client tests)

- [ ] **Step 5 (optional, ask user):** commit

---

### Task 2: Config fields + example config

**Files:**
- Modify: `src/frigate_learn/config.py:38-54` (`CollectionSettings`), `:246-266` (collection parsing)
- Modify: `config.example.yaml:21-44` (collection section)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces (consumed by Collector Task 3 and CLI Task 4):
  - `config.collection.region_crop: bool` (default `True`)
  - `config.collection.region_crop_height: int | None` (default `None` → falls back to `config.training.image_size` at collect time)

- [ ] **Step 1: Write the failing tests** — extend `tests/test_config.py`

In `test_defaults` add:

```python
    assert cfg.collection.region_crop is True
    assert cfg.collection.region_crop_height is None
```

Add a new test:

```python
def test_collection_region_crop_settings():
    raw = {
        "classes": ["person"],
        "collection": {"region_crop": False, "region_crop_height": 512},
        "data": {"root": "var"},
    }
    cfg = build_config(raw, base_dir=Path("/tmp/x"))
    assert cfg.collection.region_crop is False
    assert cfg.collection.region_crop_height == 512
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_config.py -p no:warnings`
Expected: FAIL with `AttributeError: 'CollectionSettings' object has no attribute 'region_crop'`

- [ ] **Step 3: Implement** — `src/frigate_learn/config.py`, `CollectionSettings` add after `phash_threshold`:

```python
    # Collect server-side region crops (crop=1&height=<H> snapshots) instead of
    # full frames, matching what Frigate feeds the detector. None -> use
    # training.image_size as the crop height.
    region_crop: bool = True
    region_crop_height: int | None = None
```

In module-level `build_config` after the `phash_threshold` parse (line ~264), add:

```python
    cfg.collection.region_crop = bool(_pop(col, "region_crop", cfg.collection.region_crop))
    cfg.collection.region_crop_height = _as_optional_int(
        _pop(col, "region_crop_height", None)
    )
```

- `config.example.yaml`, inside the `collection:` section after `phash_threshold`:

```yaml
  # region-crop collection (default on): download server-side crops
  # (snapshot.jpg?crop=1&bbox=0&height=<H>) matching the detector input instead
  # of full frames. height=None falls back to training.image_size.
  region_crop: true
  region_crop_height: null
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_config.py -p no:warnings`
Expected: PASS

- [ ] **Step 5 (optional, ask user):** commit

---

### Task 3: Collector crop-mode behavior

**Files:**
- Modify: `src/frigate_learn/collection/collector.py`
- Modify: `src/frigate_learn/cli.py:143-165` (`_summary_lines`)
- Test: `tests/test_collector.py`

**Interfaces:**
- Consumes: `download_region_crop`, `download_annotated_crop`, `region_crop`/`region_crop_height` config.
- Produces: `CollectSummary.skipped_no_box: int` (events skipped for degenerate boxes).

- [ ] **Step 1: Write the failing tests**

First, update the `FakeFrigate` helper in `tests/test_collector.py` to add the two crop methods after `download_event_snapshot` (line ~57):

```python
    def download_region_crop(self, event_id, output_path, height, timestamp=None) -> str:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\xff\xd8\xff\xe0fakecrop-" + event_id.encode())
        self.downloads.append(("crop", event_id, dest, {"height": height, "timestamp": timestamp}))
        return str(dest)

    def download_annotated_crop(self, event_id, output_path) -> str:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"annotated-crop-" + event_id.encode())
        self.downloads.append(("debug-crop", event_id, dest))
        return str(dest)
```

Add `from frigate_learn.models import Annotation, Sample` to the existing import (currently `Sample` only).

Now mark every existing test as legacy-full-frame mode by adding this line as the first statement of each test body (all 7 test functions: `test_happy_path`, `test_idempotent_second_run`, `test_shared_event_deduped_across_reviews`, `test_failures_recorded`, `test_incomplete_event_skipped_when_completed_only`, `test_keep_annotated_snapshots`, `test_labels_cameras_filters_passthrough`; note `RecordingFake` usage is inside the last one):

```python
    config.collection.region_crop = False
```

Append the new crop-mode tests:

```python
def test_region_crop_default_collects_crops_not_full_frames(config, db):
    fake = FakeFrigate(
        reviews_raw=[_review("r1", "front", 200, ["e1"])],
        events_raw={"e1": _event("e1", "front", "person", 200)},
    )
    summary = Collector(config, db, client=fake).collect(from_ts=0)
    assert summary.new_samples == 1
    assert summary.new_annotations == 0
    assert [d[0] for d in fake.downloads] == ["crop"]
    with db.session() as s:
        row = s.query(Sample).one()
        attrs = s.query(Annotation).all()
    assert attrs == []
    assert (row.frigate_x1, row.frigate_y1, row.frigate_x2, row.frigate_y2) == (0.1, 0.2, 0.4, 0.8)
    image = Path(row.image_path)
    assert image.is_file()
    assert image.read_bytes().startswith(b"\xff\xd8\xff\xe0fakecrop-")


def test_region_crop_default_height_from_training(config, db):
    fake = FakeFrigate(
        reviews_raw=[_review("r1", "front", 200, ["e1"])],
        events_raw={"e1": _event("e1", "front", "person", 200)},
    )
    Collector(config, db, client=fake).collect(from_ts=0)
    calls = [d for d in fake.downloads if d[0] == "crop"]
    assert calls[0][3]["height"] == config.training.image_size


def test_region_crop_configured_height(config, db):
    config.collection.region_crop_height = 512
    fake = FakeFrigate(
        reviews_raw=[_review("r1", "front", 200, ["e1"])],
        events_raw={"e1": _event("e1", "front", "person", 200)},
    )
    Collector(config, db, client=fake).collect(from_ts=0)
    calls = [d for d in fake.downloads if d[0] == "crop"]
    assert calls[0][3]["height"] == 512


def test_region_crop_skips_degenerate_box(config, db):
    fake = FakeFrigate(
        reviews_raw=[_review("r1", "front", 200, ["e_bad"])],
        events_raw={
            "e_bad": _event("e_bad", "front", "person", 200, box=[0.0, 0.0, 0.0, 0.0]),
        },
    )
    summary = Collector(config, db, client=fake).collect(from_ts=0)
    assert summary.new_samples == 0
    assert summary.skipped_no_box == 1
    assert fake.downloads == []
    with db.session() as s:
        assert s.query(Sample).count() == 0


def test_region_crop_forces_single_frame(config, db):
    config.sampling.enabled = True
    config.sampling.max_samples_per_event = 3
    fake = FakeFrigate(
        reviews_raw=[_review("r1", "front", 200, ["e1"])],
        events_raw={"e1": _event("e1", "front", "person", 200)},
    )
    summary = Collector(config, db, client=fake).collect(from_ts=0)
    assert summary.new_samples == 1
    crops = [d for d in fake.downloads if d[0] == "crop"]
    assert len(crops) == 1
    assert crops[0][3]["timestamp"] is None


def test_region_crop_debug_uses_annotated_crop(config, db):
    config.collection.keep_annotated_snapshots = True
    fake = FakeFrigate(
        reviews_raw=[_review("r1", "front", 200, ["e1"])],
        events_raw={"e1": _event("e1", "front", "person", 200)},
    )
    summary = Collector(config, db, client=fake).collect(from_ts=0)
    assert summary.new_samples == 1
    kinds = [d[0] for d in fake.downloads]
    assert "debug-crop" in kinds
    assert "clean" not in kinds
    assert "debug" not in kinds
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_collector.py -p no:warnings`
Expected: the 6 new crop-mode tests FAIL (collector has no crop path, `skipped_no_box` missing); the opted-out legacy tests still PASS at this step. (If a legacy test is left without `region_crop = False`, it fails on `new_annotations` — that is the guard for forgetting the opt-out.)

- [ ] **Step 3: Implement** — `src/frigate_learn/collection/collector.py`:

`EventOutcome` (lines 63-70, already has `error`) add one field:

```python
    annotations: int = 0      # annotation rows inserted for this event
```

`CollectSummary` add a counter after `failures` (line ~51):

```python
    skipped_no_box: int = 0   # events skipped in crop mode for a degenerate box
```

In `_download_and_store`, replace the status accounting block (lines ~222-229):

```python
                summary.duplicate_samples += outcome.skipped
                if outcome.status == "ok":
                    summary.new_samples += outcome.stored
                    summary.new_annotations += outcome.annotations
                elif outcome.status == "skip":
                    summary.skipped_no_box += 1
                elif outcome.status == "fail":
                    summary.failures += 1
                else:
                    summary.duplicate_samples += 1
```

Rewrite `_frame_times` (lines ~237-250) to force single-frame in crop mode:

```python
    def _frame_times(self, event) -> list[float | None]:
        """Decide which frame times to fetch for one event (P3 temporal sampling).

        Region crops are single-frame only: the event box is a single-frame
        artifact, so a stale box on a resampled frame would crop the wrong area.

        Returns ``[None]`` (the Frigate default frame) unless sampling is
        enabled *and* crop collection is off and the event spans enough time.
        """
        if self.config.collection.region_crop:
            return [None]
        samp = self.config.sampling
        if not samp.enabled or samp.max_samples_per_event <= 1:
            return [None]
        times = sample_timestamps(
            event.start_time, event.end_time, samp.max_samples_per_event,
            min_gap=samp.min_seconds_between_samples,
        )
        return [t if t != event.start_time else None for t in times]
```

Add a sanity helper as a private staticmethod near `_frame_times`:

```python
    @staticmethod
    def _has_sane_box(box) -> bool:
        """A usable Frigate event box: 4 normalized coords with positive area."""
        if not box or len(box) != 4:
            return False
        x1, y1, x2, y2 = box
        return 0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0
```

Rewrite the top of `_process_event` (lines 286-292) and the frame loop download/storage (lines 293-369) to branch on `region_crop`. Note the original ok/dup returns are lines 367-369 — the whole `try` body (with its original returns) must be replaced, i.e. lines **286-369**:

Replace lines 286-369 with:

```python
        try:
            event = self.client.get_event(event_id)
            if self.config.frigate.completed_events_only and not event.completed:
                raise ValueError(
                    f"event not completed (end_time is None); refusing in-progress snapshot"
                )

            crop = self.config.collection.region_crop
            if crop and not self._has_sane_box(event.box):
                return EventOutcome(status="skip")
            crop_height = self.config.collection.region_crop_height or self.config.training.image_size

            stored = 0
            skipped = 0
            annotations = 0
            for frame_index, frame_ts in enumerate(self._frame_times(event)):
                sample_id = str(uuid.uuid4())
                image_path = self._image_path(event.camera, event.start_time, sample_id)
                if crop:
                    self.client.download_region_crop(
                        event_id, image_path, height=crop_height, timestamp=frame_ts
                    )
                else:
                    self.client.download_clean_snapshot(event_id, image_path, timestamp=frame_ts)
                image_hash = self._hash_file(image_path)

                if self._is_duplicate_frame(event.camera, image_hash, image_path):
                    image_path.unlink(missing_ok=True)
                    skipped += 1
                    debug("skipped duplicate frame", event_id=event.id, camera=event.camera)
                    continue

                debug_path = None
                if self.config.collection.keep_annotated_snapshots:
                    debug_path = image_path.with_name(f"{sample_id}-debug.jpg")
                    if crop:
                        self.client.download_annotated_crop(event_id, debug_path)
                    else:
                        self.client.download_event_snapshot(event_id, debug_path)

                phash = None
                if self.config.collection.dedup_enabled:
                    try:
                        phash = dhash_file(image_path)
                    except Exception as exc:
                        debug("phash failed", event_id=event.id, error=str(exc))

                box = event.box
                sample = Sample(
                    id=sample_id,
                    camera=event.camera,
                    timestamp=frame_ts if frame_ts is not None else event.start_time,
                    event_id=event.id,
                    frame_index=frame_index,
                    review_id=review_id,
                    image_path=str(image_path),
                    debug_image_path=str(debug_path) if debug_path else None,
                    image_hash=image_hash,
                    perceptual_hash=phash,
                    source="frigate",
                    frigate_label=event.label,
                    frigate_score=event.score if event.score is not None else event.top_score,
                    frigate_x1=box[0] if box else None,
                    frigate_y1=box[1] if box else None,
                    frigate_x2=box[2] if box else None,
                    frigate_y2=box[3] if box else None,
                    status="collected",
                )
                with self.db.session() as session:
                    session.add(sample)
                    session.flush()  # persist the sample before its FK-dependent annotation
                    if not crop:
                        session.add(
                            Annotation(
                                id=str(uuid.uuid4()),
                                sample_id=sample_id,
                                source="frigate",
                                label=event.label,
                                x1=sample.frigate_x1,
                                y1=sample.frigate_y1,
                                x2=sample.frigate_x2,
                                y2=sample.frigate_y2,
                                confidence=sample.frigate_score,
                                verified=0,
                            )
                        )
                        annotations += 1
                    session.commit()
                info(
                    "collected sample",
                    camera=event.camera,
                    event_id=event.id,
                    frame_index=frame_index,
                    sample_id=sample_id,
                    label=event.label,
                    crop=crop,
                )
                stored += 1

            if stored:
                return EventOutcome(status="ok", stored=stored, skipped=skipped,
                                    annotations=annotations)
            return EventOutcome(status="dup", stored=0, skipped=skipped)
```

In `_finish_job` metadata dict (line ~429) add:

```python
                "skipped_no_box": summary.skipped_no_box,
```

In `src/frigate_learn/cli.py` `_summary_lines`, after the `failures` line add:

```python
    if summary.skipped_no_box:
        lines.append(f"No-box skipped: {summary.skipped_no_box:,}")
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_collector.py tests/test_cli.py -p no:warnings`
Expected: PASS

- [ ] **Step 5 (optional, ask user):** commit

---

### Task 4: CLI `--no-region-crop` override

**Files:**
- Modify: `src/frigate_learn/cli.py:167-228` (collect command)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `config.collection.region_crop` (Task 2).

- [ ] **Step 1: Write the failing test** — append to `tests/test_cli.py`

```python
def test_collect_no_region_crop_flag(dbenv, monkeypatch):
    captured = {}

    class RecordingCollector:
        def __init__(self, config, database):
            captured["region_crop"] = config.collection.region_crop
            self.client = type("C", (), {"close": lambda self: None})()

        def collect(self, **kw):
            return CollectSummary(new_samples=0)

    monkeypatch.setattr("frigate_learn.cli.Collector", RecordingCollector)
    result = CliRunner().invoke(cli, ["collect", "--no-region-crop"], env=dbenv)
    assert result.exit_code == 0
    assert captured["region_crop"] is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cli.py::test_collect_no_region_crop_flag -p no:warnings`
Expected: FAIL (exit code 2 — no `--no-region-crop` option)

- [ ] **Step 3: Implement** — `src/frigate_learn/cli.py` collect command.

Add the option after `--concurrency` (line ~176):

```python
@click.option("--no-region-crop", "no_region_crop", is_flag=True, default=False,
              help="Collect full-frame clean snapshots instead of region crops.")
```

Add the parameter `no_region_crop: bool` to the `collect` signature (after `concurrency`), and after `config = _load_config(ctx)` (line ~190):

```python
    if no_region_crop:
        config.collection.region_crop = False
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_cli.py -p no:warnings`
Expected: PASS

- [ ] **Step 5 (optional, ask user):** commit

---

## Self-review

**Spec coverage:** params builder (Task 1) ✓, client methods (Task 1) ✓, config fields + example yaml (Task 2) ✓, collector crop storage + no-frigate-annotation + degenerate-box skip + forced single-frame (Task 3) ✓, CLI override (Task 4) ✓, dedup on crop (unchanged path, covered by Task 3 using the same `_is_duplicate_frame` call) ✓, VLM/builder untouched ✓, train/val/test split untouched (spec: "no pixel transforms at build") ✓, golden-follow-up documented, out of scope ✓.

**Placeholder scan:** no TBD/TODO; every step has real code/commands.

**Type consistency:** `region_crop_params(height: int)` / `annotated_region_crop_params()` match the imports in client.py; `download_region_crop(event_id, output_path, height, timestamp=None)` and `download_annotated_crop(event_id, output_path)` match FakeFrigate and collector calls; `CollectSummary.skipped_no_box` and `EventOutcome.annotations` used consistently in `_download_and_store` and `cli._summary_lines`.