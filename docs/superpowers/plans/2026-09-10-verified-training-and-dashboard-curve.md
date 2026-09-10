# Verified-only Training, Honest Splits, Before/After Metrics, Dashboard Curve Fix

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make training consume only VLM-verified images with honest train/val/test splits, report pretrained→fine-tuned (before/after) performance on the golden set, and stop the dashboard epoch curve from breaking on non-finite CSV cells.

**Architecture:** The dataset builder flips to verified-only by default and writes images/labels into `images/{train,val,test}/` + `labels/{train,val,test}/` subdirectories that ultralytics resolves correctly (no split-list indirection). The trainer evaluates before (pretrained weights) and after (`weights/best.pt`) backends on the immutable golden dataset and writes `before_after.json`. The benchmark step/CLI auto-adds the latest trained run as a candidate so the gate reports fine-tuned vs baseline. The webapp serializes non-finite CSV cells as JSON `null` and the JS chart treats `null` as a missing point (with `spanGaps`).

**Tech Stack:** Python 3.11+, SQLAlchemy/SQLite, ultralytics 8.4 (train/benchmark), FastAPI+Starlette (dashboard), vanilla JS + vendored uPlot (chart), click (CLI), pytest.

**Spec:** `docs/superpowers/specs/2026-09-10-verified-training-and-dashboard-curve-design.md`

## Global Constraints

- **No comments in code** (webapp directive is strict; apply repo-wide edit discipline: do not add comments).
- `dataset/`/`training/`/`evaluation/` modules stay lazy-import-gated; tests must pass without the `ml` extra (importorskip pattern; trainer/evaluate tests must not import ultralytics).
- Full suite must stay green: `.venv/bin/python -m pytest -p no:warnings 2>&1 | tail -1` → `290 passed`.
- Ruff: introduce **no new** errors (`ruff check` currently reports 34 pre-existing).
- Commit after each task (`git add <files> && git commit -m "<message>"`), matching repo style (lowercase `area: subject`).
- Never stage `data/`, `*.pt`, `.venv`, `node_modules`, `.DS_Store` (gitignored anyway); `config.yaml`/`.env` are untracked.
- Keep `Sample`/`Annotation` schema untouched; do not change golden dataset layout (`data/golden/golden-v001` stays flat `images/` + `labels/`).

---

### Task 1: Builder — verified-only default and split-directory layout

**Files:**
- Modify: `src/frigate_learn/dataset/builder.py` (docstring tree lines 7-10, `build()` lines 98-99, 125-157, `_write_split_files` lines 222-231)
- Modify: `src/frigate_learn/evaluation/golden.py:177-185` (`write_dataset_yaml` gains optional split-dir params)
- Modify: `src/frigate_learn/cli.py:458` (`dataset build` flag swap)
- Modify: `tests/test_run.py:36-54` (`_seed_sample` switches its frigate annotation to `verified=1`)
- Create: `tests/test_dataset_builder.py`
- Test: `tests/test_run.py` (existing `test_run_collect_build_with_stubbed_collector` must still pass)

**Interfaces:**
- Consumes: `DatasetBuilder(config, db).build(version, *, cameras, labels, quality, verified_only, max_per_class, seed, overwrite)` (existing signature; only the `verified_only` default changes), `iter_manifest` from `dataset/manifest.py`, `VALID_SPLITS`/`assign_split` from `dataset/splits.py`, `write_dataset_yaml` from `evaluation/golden.py`.
- Produces: `write_dataset_yaml(path, class_names, *, train="images", val="images", test=None)` — builder calls with `train="images/train"`, `val="images/val"`, `test="images/test"`. Golden callers keep the unchanged defaults.

- [ ] **Step 1: Write the failing builder tests**

`tests/test_dataset_builder.py`:

```python
"""Dataset builder: verified-only default and split-directory layout."""

from __future__ import annotations

from PIL import Image

from frigate_learn.dataset.builder import DatasetBuilder
from frigate_learn.dataset.manifest import iter_manifest
from frigate_learn.models import Annotation, Sample, utcnow


def _seed_sample(config, db, sample_id="s1", verified=True):
    img_dir = config.images_dir() / "20260901" / "front"
    img_dir.mkdir(parents=True, exist_ok=True)
    img = img_dir / f"{sample_id}.jpg"
    Image.new("RGB", (320, 320), (30, 30, 30)).save(img)
    with db.session() as s:
        s.add(
            Sample(
                id=sample_id, camera="front", timestamp=100.0, event_id="e1",
                frame_index=0, image_path=str(img), status="collected",
                frigate_label="person", created_at=utcnow(),
            )
        )
        s.flush()
        s.add(
            Annotation(
                id=f"a{sample_id}", sample_id=sample_id,
                source="vlm" if verified else "frigate",
                label="person", x1=0.1, y1=0.1, x2=0.6, y2=0.8,
                confidence=0.9, verified=int(verified), created_at=utcnow(),
            )
        )
        s.commit()


def test_builder_default_is_verified_only(config, db, tmp_path):
    _seed_sample(config, db, verified=False)
    summary = DatasetBuilder(config, db).build("v001")
    assert summary.images_written == 0
    assert summary.skipped_no_box == 1


def test_builder_include_unverified_opt_in(config, db, tmp_path):
    _seed_sample(config, db, verified=False)
    summary = DatasetBuilder(config, db).build("v001", verified_only=False)
    assert summary.images_written == 1


def test_builder_writes_split_directories(config, db, tmp_path):
    _seed_sample(config, db)
    summary = DatasetBuilder(config, db).build("v001")
    assert summary.images_written == 1
    target = config.datasets_dir() / "v001"
    records = list(iter_manifest(target / "manifest.jsonl"))
    assert len(records) == 1
    split = records[0]["split"]
    assert (target / "images" / split / "s1.jpg").is_file()
    assert (target / "labels" / split / "s1.txt").is_file()
    assert not (target / "images" / "s1.jpg").exists()
    yaml_text = (target / "dataset.yaml").read_text(encoding="utf-8")
    assert "train: images/train" in yaml_text
    assert "val: images/val" in yaml_text
    assert "test: images/test" in yaml_text
    assert f"images/{split}/s1.jpg" in (target / f"{split}.txt").read_text(encoding="utf-8")
```

Update `tests/test_run.py` `_seed_sample` (lines 36-54) so its frigate annotation is verified (the dry-run build must still write 1 image under the new default):

```python
        s.add(
            Annotation(
                id=f"a{sample_id}", sample_id=sample_id, source="frigate",
                label="person", x1=0.1, y1=0.1, x2=0.6, y2=0.8,
                confidence=0.9, verified=1, created_at=utcnow(),
            )
        )
```

- [ ] **Step 2: Run the failing tests**

```bash
.venv/bin/python -m pytest tests/test_dataset_builder.py -v
.venv/bin/python -m pytest tests/test_run.py::test_run_collect_build_with_stubbed_collector -v
```

Expected: the three new builder tests FAIL against the current builder (still `verified_only=False` default, flat files, `train: images`). The run test PASSES with the edited `_seed_sample` — that edit is included in Step 1 and represents collect+verify having produced a verified annotation before the (dry-run) build.

- [ ] **Step 3: Update `write_dataset_yaml`**

`src/frigate_learn/evaluation/golden.py:177`:

```python
def write_dataset_yaml(
    path: Path,
    class_names: list[str],
    *,
    train: str = "images",
    val: str = "images",
    test: str | None = None,
) -> None:
    payload = {
        "path": str(path.parent),
        "train": train,
        "val": val,
        "names": {i: name for i, name in enumerate(class_names)},
    }
    if test is not None:
        payload["test"] = test
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
```

- [ ] **Step 4: Update the builder**

`src/frigate_learn/dataset/builder.py`:

1. Module docstring tree (lines 8-10) becomes:

```
├── images/            <split>/<sample_id>.<ext>   (split in train|val|test)
├── labels/            <split>/<sample_id>.txt   (YOLO class cx cy w h)
```

2. `build()` default flip (line 99): `verified_only: bool = False` → `verified_only: bool = True`.

3. Split-dir creation (lines 100-103):

```python
        images_dir = target / "images"
        labels_dir = target / "labels"
        for split in VALID_SPLITS:
            (images_dir / split).mkdir(parents=True, exist_ok=True)
            (labels_dir / split).mkdir(parents=True, exist_ok=True)
```

4. Move split assignment before the copy and write into split subdirs (lines 125-157):

```python
            split = assign_split(sample.id, seed=split_seed)
            image_dst = images_dir / split / f"{sample.id}{ext}"
            try:
                shutil.copy2(source_path, image_dst)
            except OSError:
                summary.skipped_no_box += 1
                continue

            lines = [_to_yolo_line(a, class_id) for a in usable]
            write_yolo_label(labels_dir / split / f"{sample.id}.txt", lines)

            summary.split_counts[split] = summary.split_counts.get(split, 0) + 1
```

(delete the later `split = assign_split(sample.id, seed=split_seed)` line that currently follows `write_yolo_label(...)`)

5. dataset.yaml call (line 156-158):

```python
        _write_dataset_yaml(
            target / "dataset.yaml",
            classes,
            train="images/train",
            val="images/val",
            test="images/test",
        )
```

6. `_write_split_files` (lines 222-231):

```python
    @staticmethod
    def _write_split_files(target: Path) -> None:
        splits: dict[str, list[str]] = {s: [] for s in VALID_SPLITS}
        for rec in iter_manifest(target / "manifest.jsonl"):
            split = rec.get("split", "train")
            splits.setdefault(split, []).append(f"images/{split}/{rec['image']}")
        for split, lines in splits.items():
            if not lines:
                continue
            (target / f"{split}.txt").write_text(
                "\n".join(sorted(lines)) + "\n", encoding="utf-8"
            )
```

- [ ] **Step 5: Swap the CLI flag**

`src/frigate_learn/cli.py:458`:

Remove:
```python
@click.option("--verified-only", is_flag=True, help="Only samples with a verified annotation.")
```

Add:
```python
@click.option("--include-unverified", is_flag=True, help="Also write samples labeled only by Frigate (default: verified-only).")
```

Update the command callback signature: replace `verified_only: bool,` with `include_unverified: bool,` and the build call `verified_only=verified_only,` → `verified_only=not include_unverified,`.

- [ ] **Step 6: Run the full task test set**

```bash
.venv/bin/python -m pytest tests/test_dataset_builder.py tests/test_run.py -q
```

Expected: PASS (3 new + all run tests).

- [ ] **Step 7: Commit**

```bash
git add src/frigate_learn/dataset/builder.py src/frigate_learn/evaluation/golden.py src/frigate_learn/cli.py tests/test_dataset_builder.py tests/test_run.py
git commit -m "dataset: verified-only default with split directories"
```

---

### Task 2: Trainer — before/after metrics on the golden dataset

**Files:**
- Modify: `src/frigate_learn/training/trainer.py` (add `evaluate_on_golden`, `_write_before_after`; wire into `train()`)
- Create: `tests/test_trainer.py`
- Test: `tests/test_trainer.py`

**Interfaces:**
- Consumes: `AppConfig.golden_dir()`, `config.evaluation.golden_dataset`, `config.training.image_size`, `config.classes`; `GoldenDataset.load` / `golden_to_examples` / `benchmark_candidate` / `BenchmarkConfig` from `evaluation/`; `CandidateResult` fields `map50`, `metrics.recall`, `latency_ms`.
- Produces: `evaluate_on_golden(config, weights: str, *, name: str) -> dict | None` returns `{"name", "map50", "recall", "latency_ms"}` or `None` when the golden set is absent/empty. `_write_before_after(config, before_weights, after_weights, run_dir, *, name) -> bool` writes `run_dir/before_after.json` with keys `before`, `after`, `delta` (`delta` has `map50`, `recall`).
- Does NOT import ultralytics modulo the lazy import inside `evaluate_on_golden` (guarded by `try/except ImportError`), so tests run on a collection-only install.

- [ ] **Step 1: Write the failing tests**

`tests/test_trainer.py`:

```python
"""Trainer before/after golden evaluation tests."""

from __future__ import annotations

import json

from frigate_learn.evaluation.benchmark import CandidateResult
from frigate_learn.evaluation.metrics import DetectionMetrics
from frigate_learn.training.trainer import _write_before_after, evaluate_on_golden


def test_evaluate_on_golden_missing_golden_dir_returns_none(config):
    assert evaluate_on_golden(config, "/nonexistent/weights.pt", name="yolov8n-v001") is None


def test_evaluate_on_golden_returns_metrics(config, monkeypatch):
    golden = config.golden_dir() / config.evaluation.golden_dataset
    golden.mkdir(parents=True)

    from frigate_learn.evaluation import backends, golden as golden_mod

    class DummyBackend:
        def __init__(self, weights, *, imgsz, device=None, name=None):
            self._name = name or weights

        @property
        def name(self):
            return self._name

        def predict(self, image_path, confidence=0.25):
            return []

    class FakeGolden:
        def samples(self):
            return []

    monkeypatch.setattr(golden_mod.GoldenDataset, "load", lambda path: FakeGolden())
    monkeypatch.setattr(backends, "UltralyticsBackend", DummyBackend)
    monkeypatch.setattr(
        "frigate_learn.evaluation.benchmark.golden_to_examples",
        lambda golden, classes, limit=None: [object()],
    )
    monkeypatch.setattr(
        "frigate_learn.evaluation.benchmark.benchmark_candidate",
        lambda backend, examples, *, version, kind="golden", config=None, evaluator=None: (
            CandidateResult(
                name=backend.name,
                metrics=DetectionMetrics(
                    precision=0.9, recall=0.85, f1=0.87, map50=0.91, map50_95=0.5,
                    small_object_recall=0.8, false_positive_rate=0.1,
                ),
                version=version,
                kind=kind,
            )
        ),
    )
    result = evaluate_on_golden(config, "/w.pt", name="yolov8n-v001")
    assert result == {"name": "yolov8n-v001", "map50": 0.91, "recall": 0.85, "latency_ms": None}


def test_write_before_after_writes_json(config, tmp_path, monkeypatch):
    run_dir = tmp_path / "runs" / "x"
    run_dir.mkdir(parents=True)
    monkeypatch.setattr(
        "frigate_learn.training.trainer.evaluate_on_golden",
        lambda config, weights, *, name: (
            {"name": name, "map50": 0.5, "recall": 0.4, "latency_ms": 12.0}
            if name.endswith("-before")
            else {"name": name, "map50": 0.8, "recall": 0.7, "latency_ms": 10.0}
        ),
    )
    ok = _write_before_after(config, "b.pt", "a.pt", run_dir, name="yolov8n-v003")
    assert ok is True
    payload = json.loads((run_dir / "before_after.json").read_text(encoding="utf-8"))
    assert payload["delta"] == {"map50": 0.3, "recall": 0.3}
    assert payload["before"]["name"] == "yolov8n-v003-before"
    assert payload["after"]["name"] == "yolov8n-v003"


def test_write_before_after_missing_golden_returns_false(config, tmp_path, monkeypatch):
    run_dir = tmp_path / "runs" / "x"
    run_dir.mkdir(parents=True)
    monkeypatch.setattr(
        "frigate_learn.training.trainer.evaluate_on_golden",
        lambda config, weights, *, name: None,
    )
    assert _write_before_after(config, "b.pt", "a.pt", run_dir, name="y") is False
    assert not (run_dir / "before_after.json").exists()
```

- [ ] **Step 2: Run the failing tests**

```bash
.venv/bin/python -m pytest tests/test_trainer.py -v
```

Expected: FAIL — `ImportError: cannot import name 'evaluate_on_golden'`.

- [ ] **Step 3: Implement `evaluate_on_golden` and `_write_before_after`**

`src/frigate_learn/training/trainer.py` — after `TrainResult`/before `class Trainer` (or at module bottom after `_to_yolo_line`-style helpers; keep the lazy imports inside the function):

```python
def evaluate_on_golden(config: AppConfig, weights: str, *, name: str) -> dict | None:
    golden_dir = config.golden_dir() / config.evaluation.golden_dataset
    if not golden_dir.is_dir():
        return None
    try:
        from ..evaluation.backends import UltralyticsBackend
        from ..evaluation.benchmark import (
            BenchmarkConfig,
            benchmark_candidate,
            golden_to_examples,
        )
        from ..evaluation.golden import GoldenDataset
    except ImportError:
        return None

    golden = GoldenDataset.load(golden_dir)
    examples = golden_to_examples(golden, config.classes)
    if not examples:
        return None
    backend = UltralyticsBackend(
        weights, imgsz=config.training.image_size, name=name
    )
    result = benchmark_candidate(
        backend,
        examples,
        version=config.evaluation.golden_dataset,
        kind="golden",
        config=BenchmarkConfig(classes=list(config.classes)),
    )
    return {
        "name": result.name,
        "map50": result.map50,
        "recall": result.metrics.recall,
        "latency_ms": result.latency_ms,
    }


def _write_before_after(
    config: AppConfig,
    before_weights: str,
    after_weights: str,
    run_dir: Path,
    *,
    name: str,
) -> bool:
    before = evaluate_on_golden(config, before_weights, name=f"{name}-before")
    after = evaluate_on_golden(config, after_weights, name=name)
    if before is None or after is None:
        return False
    payload = {
        "before": before,
        "after": after,
        "delta": {
            "map50": round(after["map50"] - before["map50"], 4),
            "recall": round(after["recall"] - before["recall"], 4),
        },
    }
    (run_dir / "before_after.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    info(
        "before/after on golden",
        before=f"{before['map50']:.3f}",
        after=f"{after['map50']:.3f}",
        delta=f"{payload['delta']['map50']:+.3f}",
    )
    return True
```

- [ ] **Step 4: Wire into `Trainer.train`**

Inside `train()`, after real training (right after the `metrics = dict(...)` line, before the `TrainResult(...)` return):

```python
            if best.exists():
                _write_before_after(
                    self.config, str(candidate.weights), str(best), run_dir, name=name
                )
```

- [ ] **Step 5: Verify tests pass**

```bash
.venv/bin/python -m pytest tests/test_trainer.py -v
```

Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add src/frigate_learn/training/trainer.py tests/test_trainer.py
git commit -m "training: before/after golden metrics on real runs"
```

---

### Task 3: Benchmark auto-includes the latest trained run

**Files:**
- Modify: `src/frigate_learn/evaluation/benchmark.py` (add `latest_trained_run`)
- Modify: `src/frigate_learn/run.py` (`_step_benchmark`, `import` line ~181)
- Modify: `src/frigate_learn/cli.py` (`benchmark` command, ~634-699)
- Test: `tests/test_run.py`

**Interfaces:**
- Consumes: `AppConfig.resolve`, `config.data.root`, existing `UltralyticsBackend(weights, imgsz, device, name)` and `benchmark_candidate` in `_step_benchmark`/CLI.
- Produces: `latest_trained_run(config) -> tuple[str, Path] | None` — `(run_name, weights_best_pt)` for the newest training run under `config.resolve(config.data.root, "training")` that has `weights/best.pt` (maturity by `results.csv` mtime, falling back to `best.pt` mtime). `None` when the training dir is missing or no run has weights. Callers skip adding when a result with that name already exists.

- [ ] **Step 1: Write the failing tests**

`tests/test_run.py` (append; `import os` at top of the test file):

```python
def test_latest_trained_run_none_when_training_dir_missing(config, tmp_path):
    from frigate_learn.evaluation.benchmark import latest_trained_run
    assert latest_trained_run(config) is None


def test_latest_trained_run_picks_newest_with_weights(config, tmp_path):
    from frigate_learn.evaluation.benchmark import latest_trained_run
    root = config.resolve(config.data.root, "training")
    old = root / "yolov8s-v001"
    new = root / "yolov8n-v003"
    for d, mtime in ((old, 10), (new, 20)):
        (d / "weights").mkdir(parents=True)
        (d / "weights" / "best.pt").write_bytes(b"x")
        (d / "results.csv").write_text("epoch\n", encoding="utf-8")
        os.utime(d / "results.csv", (mtime, mtime))
    name, weights = latest_trained_run(config)
    assert name == "yolov8n-v003"
    assert weights == new / "weights" / "best.pt"


def test_latest_trained_run_skips_dir_without_weights(config, tmp_path):
    from frigate_learn.evaluation.benchmark import latest_trained_run
    root = config.resolve(config.data.root, "training")
    (root / "yolov8n-v001" / "weights").mkdir(parents=True)
    (root / "yolov8n-v001" / "results.csv").write_text("epoch\n", encoding="utf-8")
    assert latest_trained_run(config) is None
```

- [ ] **Step 2: Run the failing tests**

```bash
.venv/bin/python -m pytest tests/test_run.py::test_latest_trained_run_none_when_training_dir_missing tests/test_run.py::test_latest_trained_run_picks_newest_with_weights tests/test_run.py::test_latest_trained_run_skips_dir_without_weights -v
```

Expected: FAIL — `ImportError: cannot import name 'latest_trained_run'`.

- [ ] **Step 3: Implement `latest_trained_run`**

`src/frigate_learn/evaluation/benchmark.py` (after `benchmark_candidate`, before the `restore_*`/json helpers):

```python
def latest_trained_run(config) -> tuple[str, Path] | None:
    root = config.resolve(config.data.root, "training")
    if not root.is_dir():
        return None
    best_entry: tuple[str, Path, float] | None = None
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        weights = entry / "weights" / "best.pt"
        if not weights.is_file():
            continue
        csv = entry / "results.csv"
        mtime = csv.stat().st_mtime if csv.is_file() else weights.stat().st_mtime
        if best_entry is None or mtime > best_entry[2]:
            best_entry = (entry.name, weights, mtime)
    return None if best_entry is None else (best_entry[0], best_entry[1])
```

- [ ] **Step 4: Wire into `run.py::_step_benchmark`**

Add `latest_trained_run` to the benchmark import block (line ~181). After the `for cand_name in ...` loop (right before `out_path = config.resolve(...)`), add:

```python
    trained = latest_trained_run(config)
    if trained is not None and not any(r.name == trained[0] for r in results):
        run_name, weights = trained
        backend = UltralyticsBackend(
            str(weights),
            imgsz=config.training.image_size,
            device=ctx.get("device"),
            name=run_name,
        )
        results.append(
            benchmark_candidate(
                backend,
                examples,
                version=config.evaluation.golden_dataset,
                kind="golden",
                config=BenchmarkConfig(classes=list(config.classes)),
            )
        )
```

- [ ] **Step 5: Wire into the CLI `benchmark` command**

`src/frigate_learn/cli.py` — add `latest_trained_run` to the benchmark function's import block (line ~639). After the `for name in names:` loop, before `out_path = config.resolve(...)`:

```python
    trained = latest_trained_run(config)
    if trained is not None and not any(r.name == trained[0] for r in results):
        run_name, weights = trained
        if dry_run:
            result = benchmark_candidate(
                _DryBackend(run_name),
                examples,
                version=golden.root.name,
                kind="golden",
                config=BenchmarkConfig(classes=list(config.classes)),
            )
            result.metrics.latency_ms = 0.0
        else:
            _require_ml()
            backend = UltralyticsBackend(
                str(weights),
                imgsz=config.training.image_size,
                device=config.training.device,
                name=run_name,
            )
            result = benchmark_candidate(
                backend,
                examples,
                version=golden.root.name,
                kind="golden",
                config=BenchmarkConfig(classes=list(config.classes)),
            )
        results.append(result)
```

- [ ] **Step 6: Run tests**

```bash
.venv/bin/python -m pytest tests/test_run.py -q
```

Expected: PASS (3 new + existing).

- [ ] **Step 7: Commit**

```bash
git add src/frigate_learn/evaluation/benchmark.py src/frigate_learn/run.py src/frigate_learn/cli.py tests/test_run.py
git commit -m "evaluation: benchmark latest trained run alongside baseline candidates"
```

---

### Task 4: Webapp — non-finite CSV cells and curve rendering

**Files:**
- Modify: `src/frigate_learn/webapp/queries.py` (`import math`, `_to_float_or_str` ~374-381)
- Modify: `src/frigate_learn/webapp/static/app.js` (epoch mapping ~576-584, series ~150-152)
- Test: `tests/test_webapp.py`

**Interfaces:**
- Consumes: `_read_csv`/`_to_float_or_str`, `training_run`, `training_index` in `webapp/queries.py`; `renderTrainingDetail`/`renderLineChart` in `static/app.js`.
- Produces: `_to_float_or_str` returns `None` for `"nan"`/`"inf"`/`"-inf"`/`""` and non-finite parsed floats; JS pushes `null` (missing) for `null`/`""`/non-finite values and `spanGaps: true` on both line series.

- [ ] **Step 1: Write the failing webapp tests**

`tests/test_webapp.py` (append near the other `training_run` query tests):

```python
def test_training_run_nan_empty_inf_become_none(tmp_path):
    cfg = build_config({"data": {"root": "data"}}, tmp_path)
    run_dir = tmp_path / "data" / "training" / "yolov8n-vnan"
    run_dir.mkdir(parents=True)
    (run_dir / "results.csv").write_text(
        "epoch,metrics/mAP50(B),metrics/recall(B)\n"
        "1,0.5,nan\n"
        "2,,\n"
        "3,inf,0.7\n",
        encoding="utf-8",
    )
    result = queries.training_run(cfg, "yolov8n-vnan")
    assert result["epochs"] == 3
    assert result["rows"][0]["metrics/recall(B)"] is None
    assert result["rows"][1]["metrics/mAP50(B)"] is None
    assert result["rows"][2]["metrics/mAP50(B)"] is None
    assert result["best_map50"] is None
    assert result["latest_map50"] is None
```

API-level regression (the 500): right after `test_api_training_run`:

```python
def test_api_training_run_with_nan_returns_200(tmp_path):
    c, _ = _api_client(tmp_path)
    run_dir = tmp_path / "data" / "training" / "yolov8n-vnan"
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "results.csv").write_text(
        "epoch,metrics/mAP50(B),metrics/recall(B)\n1,0.5,nan\n2,0.6,\n",
        encoding="utf-8",
    )
    r = c.get("/api/training/yolov8n-vnan")
    assert r.status_code == 200
    body = r.json()
    assert body["rows"][0]["metrics/recall(B)"] is None
    assert body["rows"][1]["metrics/mAP50(B)"] is None
```

- [ ] **Step 2: Run the failing tests**

```bash
.venv/bin/python -m pytest tests/test_webapp.py -k "nan" -v
```

Expected: FAIL — `_to_float_or_str` returns `float("nan")`/`float("inf")`; the API test 500s (`ValueError: Out of range float values are not JSON compliant: nan`), the query test returns non-`None` NaN objects.

- [ ] **Step 3: Fix `_to_float_or_str`**

`src/frigate_learn/webapp/queries.py` — add `import math` to the module imports; replace the function:

```python
def _to_float_or_str(value: str | None) -> float | str | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return value
    return parsed if math.isfinite(parsed) else None
```

- [ ] **Step 4: Fix the JS epoch mapping**

`src/frigate_learn/webapp/static/app.js` — replace the loop body (~576-584):

```js
  for (const row of data.rows || []) {
    const e = Number(row.epoch);
    if (!Number.isFinite(e)) continue;
    xs.push(e);
    const m = row[MAP50_KEY];
    const r = row[RECALL_KEY];
    map.push((m === null || m === undefined || m === "" || !Number.isFinite(Number(m))) ? null : Number(m));
    rec.push((r === null || r === undefined || r === "" || !Number.isFinite(Number(r))) ? null : Number(r));
  }
```

(No comments — webapp directive.)

- [ ] **Step 5: Enable `spanGaps`**

`src/frigate_learn/webapp/static/app.js` series block (~150-152):

```js
    series: [
      {},
      { label: series[0].label, stroke: series[0].color, width: 2, spanGaps: true, points: { show: false } },
      { label: series[1].label, stroke: series[1].color, width: 2, spanGaps: true, points: { show: false } },
    ],
```

- [ ] **Step 6: Run the task test set**

```bash
.venv/bin/python -m pytest tests/test_webapp.py -k "training_run or api_training" -q
```

Expected: PASS (new + existing training/webapp tests).

- [ ] **Step 7: Commit**

```bash
git add src/frigate_learn/webapp/queries.py src/frigate_learn/webapp/static/app.js tests/test_webapp.py
git commit -m "webapp: serialize non-finite training cells as null and bridge epoch curve gaps"
```

---

### Task 5: End-to-end verification on the live config

**Files:** (no product changes)
- Runtime: `config.yaml` (untracked), `data/datasets/v004`, `data/training/yolov8n-v004`
- Reuse: `/var/folders/wy/06f64nbx6kz8z88km48c3b3h0000gn/T/opencode/chart-repro/` jsdom harness (uses `node_modules` now relocated beside it)

**Interfaces:**
- Consumes: everything produced by Tasks 1-4 plus the existing `data/golden/golden-v001`.
- Produces: evidence (checked) for the four objectives. **Do not run VLM verification** (user: enough annotated data already; no `verify` step).

- [ ] **Step 1: Rebuild the dataset with the new default layout**

```bash
.venv/bin/python -m pytest -p no:warnings 2>&1 | tail -1
.venv/bin/frigate-learn --config config.yaml dataset build v004
```

Expected: full suite still `N passed` (no regressions); build output `verified=<42>`-ish, `skipped(no_box=... )`, and `Splits: train=.. val=.. test=..`. Confirm the new layout:

```bash
ls data/datasets/v004/images data/datasets/v004/labels
cat data/datasets/v004/dataset.yaml
```

Expected: `images/{train,val,test}` and `labels/{train,val,test}`; yaml `train: images/train`, `val: images/val`, `test: images/test`; **no** flat files under `data/datasets/v004/images/`.

- [ ] **Step 2: Retrain real (short) and check before/after**

```bash
.venv/bin/frigate-learn --config config.yaml train yolov8n v004 --real --epochs 50
cat data/training/yolov8n-v004/before_after.json
```

Expected: `before_after.json` with `before` (pretrained on golden) and `after` (fine-tuned) map50/recall + `delta`; the CLI log line `before/after on golden before=... after=... delta=...`. Val metrics now come from `images/val` (honest holdout), printed as part of the training summary.

- [ ] **Step 3: Benchmark with the trained run auto-discovered**

```bash
.venv/bin/frigate-learn --config config.yaml benchmark
cat data/benchmark-results.json | .venv/bin/python -c "import json,sys; d=json.load(sys.stdin); print([c['name'] for c in d['results']])"
```

Expected: result names include the configured baseline candidates **and** `yolov8n-v004` (the trained run). Then run the gate:

```bash
.venv/bin/frigate-learn --config config.yaml gate
```

Expected: gate verdicts per candidate including `yolov8n-v004` PASS/FAIL vs baseline, recorded in `deployments` ledger.

- [ ] **Step 4: Verify the dashboard curve**

```bash
cd /var/folders/wy/06f64nbx6kz8z88km48c3b3h0000gn/T/opencode/chart-repro && NODE_PATH=$PWD/node_modules node harness.cjs
```

Expected: harness renders the Training epoch curve with **no zero-spike dips** at missing epochs and **no error box**; lines bridge gaps. Spot-check the live API:

```bash
.venv/bin/frigate-learn web --host 127.0.0.1 --port 8080
# open http://127.0.0.1:8080, Training view: mAP50/recall curves render; inject a NaN via
#  echo 'nan' >> data/training/yolov8n-v004/results.csv  (remove after check) and confirm no 500
```

- [ ] **Step 5: Confirm no stray artifacts and final state**

```bash
git status --short
```

Expected: only the four task commits plus the pre-existing untracked leftovers (`AGENTS.md`, `README.md`, `src/frigate_learn/deploy/hailo.py`, `src/frigate_learn/evaluation/backends.py`, `tests/test_deploy.py`, `tests/test_backends.py`, `yolov8*.pt`, `.DS_Store`). **No** `node_modules/`, `package.json`, `package-lock.json` in the repo root (relocated to the temp harness dir). `data/` contents remain gitignored.