# Frozen-Head Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune a pretrained COCO-80 YOLO on the 8 camera-relevant configured classes (person, car, bicycle, motorcycle, bus, truck, dog, cat) while the other 72 COCO-80 classification head rows stay mathematically frozen — zero gradient, zero feedback into the shared backbone — keeping an 80-class drop-in Frigate labelmap.

**Architecture:** A `training.label_space` setting switches the dataset builder, trainer, and deploy snippet between two modes. Label space `"coco80"` (default): built datasets keep the COCO-80 class order (model output index `i` is `COCO_80[i]`), `dataset.yaml` declares all 80 names so ultralytics builds a 80-class head, and a per-class **loss mask** (subclassed `v8DetectionLoss` / `DetectionTrainer`) zeroes the cls-loss contribution of the 72 frozen columns before `sum()`/`target_scores_sum`, so those rows receive zero gradient. Label space `"subset"`: current behavior (compact head over `config.classes`, plain `YOLO().train()`). Golden set and benchmark are **unchanged** — matching stays name-based.

**Tech Stack:** Python 3.13, ultralytics 8.4.146 (installed in `.venv`, `ml` extra), torch, SQLAlchemy, pytest, ruff.

## Global Constraints

- **No comments in code.** Module docstrings + `# ---` section banners only. No `#` inline comments added anywhere (repo-wide; strict in `webapp/`, but keep all files comment-free).
- Full pytest suite must stay green: `.venv/bin/python -m pytest -p no:warnings 2>&1 | tail -1` → `N passed` (currently 273; baseline at HEAD `cf55b4c`). Do not disturb the still-dirty working tree files (`webapp/static/{app.js,index.html,styles.css}`, `tests/test_backends.py`, `NOTES.md`, `AGENTS.md`, `*.pt`, `.DS_Store`) — stage only files this plan touches.
- `ruff` clean on changed files. Binary at `/opt/homebrew/bin/ruff`; run with `--select E,F` over changed files only (pre-existing violations exist; do not widen scope).
- torch/ultralytics must be imported **lazily** (inside functions) everywhere except the new `training/masked.py`, mirroring `training/trainer.py`. Tests that import masked.py must `pytest.importorskip("torch")`/guarded — torch IS installed in this venv, but the package must not hard-fail on a collection-only install.
- Default shell is zsh; venv binaries at `.venv/bin/...`. Ultralytics caches/weights never committed (`yolov8*.pt` are presently untracked; do not `git add` them). Tests must not download weights — use `YOLO("yolov8n.yaml")` (build-from-yaml, no download, verified working) or monkeypatched stubs.
- Installed ultralytics facts (verified against `.venv/lib/python3.13/site-packages/ultralytics` 8.4.146 on this machine):
  - `v8DetectionLoss.__init__` sets `self.class_weights = getattr(model, "class_weights", None)` reshaped `(1,1,-1)`.
  - `__call__(preds, batch)` → `loss(parse_output(preds), batch)` → `get_assigned_targets_and_loss(preds, batch)`; cls block is `target_scores_sum = max(target_scores.sum(), 1)`, `bce_loss = self.bce(pred_scores, target_scores.to(dtype))` (shape `(bs, num_anchors, nc)`), then `loss[1] = bce_loss.sum() / target_scores_sum`. The `loss` inside `get_assigned_targets_and_loss` and `bce` are local variables (`bce = nn.BCEWithLogitsLoss(reduction="none")`), **not** class attrs — so the mask must be applied inside the overridden method body.
  - `BaseModel.loss(batch, preds=None)` lazily sets `self.criterion = self.init_criterion()` (a `v8DetectionLoss`) then returns `self.criterion(preds, batch)`; `DetectionModel.forward` on a dict delegates to `self.loss(batch)`. Training calls `self.model(batch)` (uncompiled) or `unwrap_model(self.model).loss(batch, preds)` (compiled) — both end at `self.criterion`. So attaching `model.criterion = MaskedDetectionLoss(model, ...)` in `get_model` is the single correct seam.
  - `DetectionTrainer.get_model(cfg, weights=None, verbose=True)` builds `DetectionModel(cfg, nc=self.data["nc"], verbose=verbose)`, sets names, and calls `model.load(weights)`. `DetectionTrainer.__init__(cfg=DEFAULT_CFG, overrides=None, _callbacks=None)` — custom kwargs must go through the subclass's own `__init__`.
  - `trainer.best`/`trainer.last` are `Path` (`wdir/best.pt`, `wdir/last.pt`); `trainer.metrics` is a validator-result dict (`dict(trainer.metrics or {})`).
  - `BaseTrainer.set_class_weights()` is a no-op for detection (no class-weight conflict). EMA `update_attr` copies `class_weights` from the model, safely.
- Command reference: `data` dir for tests is stubbed via `config` fixture (see `tests/conftest.py` `make_config_file` / `build_config({"data": {"root": "data"}}, tmp_path)`); `_seed_sample` helper in `tests/test_dataset_builder.py` seeds one person sample. `write_dataset_yaml(path, class_names, *, train="images", val="images", test=None)` in `src/frigate_learn/evaluation/golden.py` writes `names: {i: name}`.
- Specs: `docs/superpowers/specs/2026-09-11-frozen-head-finetune-design.md`. Out of scope (do NOT touch): `webapp/`, `cli.py`, golden-set writer, benchmark/gate logic, `evaluation/backends.py`, `data/` contents, COCO replay for non-trainable classes (deferred, documented in spec).

---

### Task 1: COCO-80 catalog + mask helpers in `src/frigate_learn/classes.py`

**Files:**
- Create: `src/frigate_learn/classes.py`
- Test: create `tests/test_classes.py`

**Interfaces:**
- `COCO_80: tuple[str, ...]` — the canonical 80 COCO-2017 class names in index order (list below; must match ultralytics `coco.yaml` exactly: person=0 … toothbrush=79).
- `coco_index(name: str) -> int` — returns `COCO_80.index(name)`; raises `ValueError` for non-COCO names (e.g. `"deer"`).
- `collect_indices(names: Iterable[str]) -> list[int]` — `[coco_index(n) for n in names]`.
- `trainable_mask(nc: int, trainable: Iterable[str]) -> list[bool]` — `len == nc`, `True` exactly at `coco_index` positions of `trainable` names.
- Consumes: nothing (pure std-lib module; no torch import — the mask is booleans, Task 4 turns it into a tensor).
- Produces: the helpers above; `__all__` lists them.

**COCO-80 order (index = position):**
`("person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush")`

- [ ] **Step 1: Write the failing tests** — `tests/test_classes.py`
  - `test_coco80_has_80_entries`: `len(COCO_80) == 80` and no duplicates (`len(set(COCO_80)) == 80`).
  - `test_coco80_spot_indices`: `coco_index("person") == 0`, `coco_index("car") == 2`, `coco_index("dog") == 16`, `coco_index("cat") == 15`, `coco_index("bicycle") == 1`, `coco_index("motorcycle") == 3`, `coco_index("bus") == 5`, `coco_index("truck") == 7`, `COCO_80.index("toothbrush") == 79`.
  - `test_coco_index_unknown_name_raises`: `pytest.raises(ValueError, coco_index, "deer")`.
  - `test_collect_indices_default_classes`: `collect_indices(["person","car","bicycle","motorcycle","bus","truck","dog","cat"]) == [0,2,1,3,5,7,16,15]` (order follows input, not COCO order).
  - `test_trainable_mask`: `trainable_mask(80, ["person","cat"])` → `len 80`, `[i==0 or i==15]`; `trainable_mask(80, ["deer"])` raises `ValueError`; `trainable_mask(4, ["person","car"])` → `[True,False,True,False]`.

- [ ] **Step 2: Implement `classes.py`** exposing the four names above. Module docstring notes the catalog is the COCO-2017 train/val 80-class order used by the Frigate labelmap.

- [ ] **Step 3: Verify** — `.venv/bin/python -m pytest tests/test_classes.py -v` green; `.venv/bin/python -m pytest -p no:warnings 2>&1 | tail -1` still ≥ 273 passed; `/opt/homebrew/bin/ruff --select E,F src/frigate_learn/classes.py tests/test_classes.py` clean.

- **Review gate (spec + quality):** pure module, no deps; exact order, spot checks at 0/1/2/3/5/7/15/16/79; ValueError path; mask length = nc; no comments.

- **Commit message:** `classes: frozen-head fine-tuning COCO-80 catalog and mask helpers`

---

### Task 2: `training.label_space` / `lr0` / `lrf` config + `AppConfig` class-space helpers

**Files:**
- Modify: `src/frigate_learn/config.py` (TrainingSettings, build_config, AppConfig methods)
- Modify: `config.example.yaml` (training section; document the new keys)
- Test: `tests/test_config.py`

**Interfaces:**
- `TrainingSettings.label_space: str = "coco80"` — accepts `"coco80"` or `"subset"`; anything else raises `ValueError` at build time.
- `TrainingSettings.lr0: float | None = None`, `TrainingSettings.lrf: float | None = None` — optional initial/final LR overrides passed through to the trainer only when not None.
- `AppConfig.model_class_names() -> list[str]` — `list(COCO_80)` when `label_space == "coco80"`, else `list(self.classes)`.
- `AppConfig.trainable_class_mask() -> list[bool] | None` — `None` when `label_space != "coco80"`; else `trainable_mask(len(COCO_80), self.classes)`.
- Validation in `build_config`: when `label_space == "coco80"`, every entry of `config.classes` must be a COCO-80 name; non-COCO entries raise `ValueError` listing the offenders (so `"deer"` fails fast at load, not at train). Consumes `coco_index`/`COCO_80` from `classes.py`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_config.py`; use `build_config(raw, base_dir=Path("/tmp/x"))` style from existing tests)
  - `test_training_label_space_default_coco80`: `AppConfig().training.label_space == "coco80"`; `AppConfig().training.lr0 is None`; `AppConfig().training.lrf is None`.
  - `test_build_config_label_space_subset`: `build_config({"training": {"label_space": "subset"}}, ...).training.label_space == "subset"`.
  - `test_build_config_invalid_label_space_raises`: `pytest.raises(ValueError)` for `{"training": {"label_space": "bbox"}}`.
  - `test_build_config_lr_overrides`: `build_config({"training": {"lr0": 0.001, "lrf": 0.01}}, ...)` → floats set; empty string → `None`.
  - `test_model_class_names_coco80`: default config → `len(...) == 80`, first is `"person"`, index 0→`"person"`, 2→`"car"`, 15→`"cat"`, 16→`"dog"`.
  - `test_model_class_names_subset_matches_classes`: `build_config({"classes": ["person","car"], "training": {"label_space": "subset"}}, ...)` → `["person","car"]`.
  - `test_trainable_class_mask_coco80`: default → `len 80`, `sum == 8`, `[0,1,2,3,5,7,15,16]` all `True`, e.g. `[4]` (`airplane`) and `[79]` (`toothbrush`) `False`.
  - `test_trainable_class_mask_subset_is_none`: subset config → `None`.
  - `test_coco80_rejects_non_coco_class`: `build_config({"classes": ["person","deer"]}, ...)` → `pytest.raises(ValueError, match="deer")`.
  - Also extend `test_defaults` (optional) to assert `label_space == "coco80"`.

- [ ] **Step 2: Implement** — add fields, coercion (reuse `_as_optional_float`), validation block after `cfg.classes`/training section mapping, and the two `AppConfig` methods. Imports: `from .classes import COCO_80, trainable_mask, coco_index` (no cycle — classes.py imports nothing).

- [ ] **Step 3: Document in `config.example.yaml`** — inside the `training:` block add (with the existing style, no `#` comments in code — YAML comments are fine in example config):
```yaml
  # coco80: keep the full COCO-80 label head, freeze the 72 non-configured classes.
  #         subset: compact head over the configured classes (legacy).
  label_space: coco80
  lr0: null
  lrf: null
```

- [ ] **Step 4: Verify** — `tests/test_config.py` green; full suite still ≥ 273 passed; ruff clean on both files.

- **Review gate:** defaults correct; `"deer"` fails fast under coco80 and passes under subset; mask = 8 True at the exact indices; lr0/lrf None-tolerant; YAML docs added; no comments in code.

- **Commit message:** `config: label_space and lr0/lrf training options with COCO-80 class helpers`

---

### Task 3: Dataset builder emits COCO-80 datasets under `label_space=coco80`

**Files:**
- Modify: `src/frigate_learn/dataset/builder.py`
- Test: `tests/test_dataset_builder.py`

**Interfaces:**
- Consumes: `config.model_class_names()`, `config.trainable_class_mask()`.
- Produces (coco80 mode):
  - `class_id = {name: coco_index(name) for name in model_class_names-that-are-usable}` — i.e. YOLO label class ids become the **COCO-80 index** of the annotation label, so a `person` row is `0`, `cat` is `15`, `dog` is `16`. Filtering (`usable`) still keeps only labels in `config.classes`.
  - `dataset.yaml` written with `model_class_names()` (80 names, indices 0..79) instead of `config.classes`.
  - `build.json` gains `"label_space": <value>` and `"trainable": list(config.trainable_class_mask())` (the bool list; `None`-safe: omit or store when not None).
- Subset mode: byte-for-byte existing behavior (enumerate over `config.classes`, compact yaml).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_dataset_builder.py`; reuse `_seed_sample`, but it only seeds `person` — extend by writing two extra annotations directly via `db.session()`, or add a small `_seed_annotations(config, db, labels)` helper in the test file; fixtures `config`, `db` come from `tests/conftest.py`)
  - `test_builder_coco80_label_ids_and_yaml`: seed `person` + `cat` annotations on one sample; build default (coco80); assert label file `labels/<split>/s1.txt` rows start with `0 ` and `15 ` (split comes from `records[0]["split"]` as in the existing test); `dataset.yaml` `names` has 80 entries, `names["15"] == "cat"`, `names["16"] == "dog"`; `build.json["label_space"] == "coco80"`; `build.json["trainable"]` is a 80-element list with `True` at 0 and 15.
  - `test_builder_subset_still_compact`: build with `config.training.label_space = "subset"` (mutate the fixture config); label rows use sequential 0-based ids; `dataset.yaml` has `len(class_id)==len(config.classes)` names; `build.json` has no `label_space`/`trainable` keys (or stores label_space=subset).
  - `test_builder_filters_non_coco_and_non_configured_labels`: seed `person` + `bird` (COCO but NOT configured) + a `bike` Frigate label — bird/bike dropped, `skipped_class` increments, label rows only person (id 0); under coco80.
  - Existing tests must stay green unchanged (they seed only `person`, whose coco id 0 == sequential id 0, and never assert yaml names count).

- [ ] **Step 2: Implement** — replace the two-line `classes = list(self.config.classes); class_id = {name: i for i, name in enumerate(classes)}` at the top of `build()` with a coco80/subset branch:
```python
model_classes = config.model_class_names()
if label_space == "coco80":
    class_id = {name: coco_index(name) for name in model_classes if name in config.classes}
else:
    class_id = {name: i for i, name in enumerate(model_classes)}
```
(`model_classes` == `config.classes` in subset, so this collapses to the current behavior.) Thread `model_classes` into the `_write_dataset_yaml` call, and `label_space`/`trainable` into `_write_build_json` payload (add `"trainable"` key only when in coco80 mode). Everything else (filtering by `config.classes`, caps, manifest) unchanged.

- [ ] **Step 3: Verify** — `tests/test_dataset_builder.py` green; full suite still ≥ 273 passed; ruff clean.

- **Review gate:** label ids are real COCO indices (spot-check 0/15/16 in the yaml line file); non-configured COCO labels (bird) and non-COCO labels dropped with `skipped_class`; subset mode identical to before; build.json records `label_space` + `trainable`; no comments.

- **Commit message:** `dataset: emit COCO-80 label ids and 80-name dataset.yaml under label_space=coco80`

---

### Task 4: masked cls loss + masked trainer in `src/frigate_learn/training/masked.py`

**Files:**
- Create: `src/frigate_learn/training/masked.py`
- Test: create `tests/test_masked.py` (guard `pytest.importorskip("ultralytics")`/`torch` at top — install present, but keep optionality)

**Interfaces:**
- `mask_cls_loss(bce_loss, target_scores, mask) -> Tensor` — pure torch, no ultralytics dependency. `mask` is a 1-D bool/fp per-class vector. Returns `(bce_loss * mask.view(1,1,-1)).sum() / max((target_scores * mask.view(1,1,-1)).sum(), 1)`. This re-expresses the installed cls term (verified above) so frozen columns contribute exactly 0 before `sum()` **and** the denominator sums only trainable columns.
- `MaskedDetectionLoss(v8DetectionLoss)` — `__init__(self, model, *, trainable: Iterable[int])` builds `self.cls_mask` as a fp32 tensor `(1,1,nc)` with `1.0` on trainable columns, `0.0` elsewhere, then **overrides `get_assigned_targets_and_loss(preds, batch)`** by copying the installed 8.4.146 body with the cls term replaced by `loss[1] = mask_cls_loss(bce_loss, target_scores, self.cls_mask)` (bce computed exactly as the parent does: `self.bce(pred_scores, target_scores.to(dtype))`). Do NOT mutate `self.bce` or reuse `self.class_weights` (both are semantically leased to other purposes); the re-expression is a body copy, and the module docstring notes it mirrors the installed ultralytics 8.4.146 source (kept version-verified, comment-free).
- `MaskedDetectionTrainer(DetectionTrainer)` — `__init__(self, trainable=None, **kwargs)` stores `self._trainable = list(trainable)` then `super().__init__(**kwargs)`; overrides `get_model(cfg, weights=None, verbose=True)` to call `super().get_model(cfg, weights, verbose)` and attach `model.criterion = MaskedDetectionLoss(model, trainable=self._trainable)` before returning. **This is the whole override** — `DetectionModel.nc` (80) comes from `self.data["nc"]` (dataset.yaml), no other plumbing.
- Guard: `MaskedDetectionLoss.__init__` must raise `ValueError` if `trainable` is empty or has indices outside `[0, nc)`.

- [ ] **Step 1: Read the installed source and write the failing tests** — first, dump the local 8.4.146 `get_assigned_targets_and_loss` body (`python -c "import inspect, ultralytics..."`) to transcribe the copy accurately; then write `tests/test_masked.py`:
  - `test_mask_cls_loss_all_trainable_equals_plain`: random `bce_loss`, `target_scores`; `mask=ones` → equals `bce_loss.sum()/max(target_scores.sum(),1)` (torch.allclose).
  - `test_mask_cls_loss_zeroes_frozen_columns`: with one frozen column holding nonzero `target_scores` and losses, `mask_cls_loss(...)` equals the recompute with that column zeroed, and `== 0` when all positives are frozen (denom = 1 from a trainable col).
  - `test_mask_loss_zero_gradient_on_frozen`: `bce_loss = pred_scores` (requires_grad), frozen column target=1; after `mask_cls_loss(...).backward()`, `pred_scores.grad` on the frozen column is **0**, and on a trainable column is nonzero (needs at least one trainable positive).
  - `test_masked_detection_loss_zero_cls_when_only_frozen`: build `model = YOLO("yolov8n.yaml").model`; set `model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)`; attach `model.criterion = MaskedDetectionLoss(model, trainable=(0,))`; synthetic batch (like the verified harness: `img=torch.rand(1,3,64,64)`, `batch_idx=zeros(2)`, `cls=[3,3]` motorcycle-only, `bboxes=[...]`); `loss, items = model.loss(batch)` via `BaseModel.loss`; assert `items["cls_loss"] == 0.0` (frozen), and with `trainable=range(80)` the same batch yields `cls_loss > 0`.
  - `test_masked_trainer_get_model_attaches_criterion`: `MaskedDetectionTrainer(data={...nc:80...}, trainable=(0,15,16))`; stub `get_model`'s super/build so no forward runs (monkeypatch `DetectionTrainer.get_model` to return a bare `DetectionModel`? simplest: monkeypatch the module-level call — see step below); assert returned model's `criterion` is a `MaskedDetectionLoss` with `cls_mask` matching trainable. To keep this light and CPU-safe, construct `MaskedDetectionTrainer(trainable=(0,), overrides={"data": {"nc": 80}})`, monkeypatch `"frigate_learn.training.masked.DetectionTrainer.get_model"` to a lambda returning a minimal fake model exposing `.criterion`/`.nc`, call the override, assert delegation + attach happened.
  - `test_masked_trainer_empty_trainable_raises`: `pytest.raises(ValueError)`.
  - `test_masked_loss_rejects_bad_index`: `trainable=(0, 999)` → `pytest.raises(ValueError)`. If index validation needs `model.nc`, compute `nc = model.model[-1].nc` or `len(model.names)` in the loss init.

- [ ] **Step 2: Implement `masked.py`** per the interfaces. `masked.py` is only imported from `trainer.py`'s coco80 branch (itself inside the ml-guarded path) and from tests that `importorskip`, so it may import `torch` and ultralytics at module top — but wrap those top-level imports in `try/except ImportError` and have both `MaskedDetectionLoss` and `MaskedDetectionTrainer` raise the repo-standard clear `RuntimeError` hint ("ultralytics/torch not installed; install the 'ml' extra ...") from their `__init__` when the base classes are unavailable. This keeps `import frigate_learn.training.masked` side-effect free on a collection-only install without contorting the class definitions. `mask_cls_loss` stays pure torch (import `torch` at module top alongside the guarded imports; it is only reachable through this module).

- [ ] **Step 3: Verify** — `tests/test_masked.py` green; full suite ≥ 273 passed; ruff clean.

- **Review gate (spec + quality):** body copy matches installed 8.4.146 source on this machine (diff against a fresh dump); the ONLY change to the copied body is the cls term via `mask_cls_loss`; denominator sums trainable only; frozen-column gradient provably 0 (test); no comments.

- **Commit message:** `training: masked per-class cls loss and trainer for frozen-head fine-tuning`

---

### Task 5: trainer routes coco80 through `MaskedDetectionTrainer`

**Files:**
- Modify: `src/frigate_learn/training/trainer.py`
- Test: `tests/test_trainer.py`

**Interfaces:**
- Consumes: `config.training.label_space`, `config.trainable_class_mask()`, `config.training.{model,image_size,epochs,batch,device,freeze,project,seed,lr0,lrf}`.
- Produces: in coco80 mode, `Trainer.train` builds `MaskedDetectionTrainer(overrides={...}, trainable=mask)` and calls `.train()` directly (no `YOLO()` object); `best = Path(trainer.best)` (a `Path` from `wdir/best.pt`); `metrics = dict(trainer.metrics or {})`. Subset mode keeps the existing `YOLO(candidate.weights).train(...)` path untouched.
- Overrides dict (only when non-None): `model=str(candidate.weights)`, `data=str(dataset_yaml)`, `epochs`, `imgsz`, `batch`, `project=str(project)`, `name=name`, `exist_ok=True`, `seed`, `device?`, `freeze?`, `lr0?`, `lrf?`. (`DetectionTrainer` accepts `lr0`/`lrf` overrides natively.)
- Same ImportError→RuntimeError guard and `_safe_before_after` call as today. Imports stay lazy inside the coco80 branch: `from .masked import MaskedDetectionTrainer` (guarded).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_trainer.py`)
  - `test_train_coco80_uses_masked_trainer`: monkeypatch `"frigate_learn.training.trainer.MaskedDetectionTrainer"` with a recording fake: `.train()` sets `.best = Path(run_dir)/"weights"/"best.pt"` (create it) and `.metrics = {"mAP50(B)": 0.9}`; then call `Trainer(config, db).train("yolov8n", "v001")` on a config whose `datasets_dir()/v001/dataset.yaml` exists (create stub yaml + a stub `dataset.yaml` under `config.datasets_dir()/v001`); assert the fake was constructed with `overrides["data"]` == that yaml path, `overrides["name"] == "yolov8n-v001"`, `overrides["imgsz"] == config.training.image_size`, `trainable == config.trainable_class_mask()`; assert `TrainResult.weights_path == best path` and `metrics["mAP50(B)"] == 0.9`.
  - `test_train_coco80_forwards_lr_and_freeze`: same fake; config with `training.lr0=0.001`, `training.lrf=0.05`, `training.freeze=10`, `device="cpu"`; assert overrides contain those (and `device == "cpu"`).
  - `test_train_coco80_omits_none_overrides`: fake raising `AssertionError` if `"lr0" in overrides`; default config → overrides have no `lr0`/`lrf`/`device`/`freeze` keys.
  - `test_train_subset_keeps_yolo_path`: existing behavior — assert the subset branch still calls `YOLO(...).train` (monkeypatch `frigate_learn.training.trainer.YOLO` with a fake recording `.train` kwargs; config with `training.label_space="subset"`). Best/metrics resolution via `fake_results.save_dir`/`results_dict`.
  - Keep `test_evaluate_on_golden...` etc. unchanged.
- Note on construction: `resolve(model_name)` loads candidates (see `training/candidates.py`) — keep the fake model weights path valid by using the existing candidates (e.g. `"yolov8n"` resolves to a `.pt`/`.onnx` literal path; the fake `MaskedDetectionTrainer`/`YOLO` never load it). Check `resolve`'s return for the exact candidate attribute name (`candidate.weights`) in `trainer.py` line 136.

- [ ] **Step 2: Implement** — restructure the training block (after `dry_run` check and right where `model = YOLO(candidate.weights)` is today) into:
```python
if cfg.label_space == "coco80":
    from .masked import MaskedDetectionTrainer
    trainable = config.trainable_class_mask()
    overrides = {...}
    trainer = MaskedDetectionTrainer(overrides=overrides, trainable=trainable)
    trainer.train()
    best = Path(trainer.best)
    metrics = dict(trainer.metrics or {})
else:
    ... existing YOLO branch ...
```
Keep the surrounding `try/except ImportError` as the guard for both branches; `_safe_before_after` runs on success in both.

- [ ] **Step 3: Verify** — `tests/test_trainer.py` green; full suite ≥ 273 passed; ruff clean.

- **Review gate:** coco80 constructs `MaskedDetectionTrainer` with exact overrides (name/imgsz/data/trainable); None-valued overrides omitted; subset path provably unchanged; best/metrs handled identically; `_safe_before_after` still fires.

- **Commit message:** `training: route label_space=coco80 through masked trainer overrides`

---

### Task 6: deploy emits COCO-80 snippet + manifest carries label space

**Files:**
- Modify: `src/frigate_learn/deploy/hailo.py`
- Test: `tests/test_deploy.py`

**Interfaces:**
- Consumes: `config.model_class_names()`.
- Produces: `render_frigate_detector_config(model_name, hef_name, classes, imgsz)` unchanged signature — `deploy()` now passes `config.model_class_names()` instead of `config.classes`, so the snippet `num_classes` is 80 under coco80 (Frigate needs the full head), and stays `len(config.classes)` under subset. The manifest gains two keys: `"label_space": <value>` and `"num_classes": <int>`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_deploy.py`)
  - `test_render_coco80_num_classes_80`: `render_frigate_detector_config("m", "m.hef", config.model_class_names(), imgsz=320)` with default config → `"num_classes: 80"` in snippet.
  - `test_deploy_manifest_label_space_and_num_classes`: reuse the `test_deploy_dry_run...` shape: `deploy(config, db, model_name="yolov8n-coco80", weights=<fake pt>, version="v001", dry_run=True, out_dir=tmp_path/"models"/...)`; assert `manifest["label_space"] == "coco80"` and `manifest["num_classes"] == 80`; snippet has `num_classes: 80`.
  - `test_deploy_subset_compact`: config with `training.label_space="subset"` → `manifest["num_classes"] == len(config.classes)` and `num_classes: <n>` in snippet.

- [ ] **Step 2: Implement** — in `deploy()`, compute `model_classes = config.model_class_names()` once; use it for the snippet; add the two manifest keys. Leave `render_frigate_detector_config` and export/compile untouched.

- [ ] **Step 3: Verify** — `tests/test_deploy.py` green; full suite still green; ruff clean.

- **Review gate:** snippet num_classes reflects the model head (80 under coco80, compact under subset); manifest records label_space + num_classes; no behavior change under default config beyond the two new manifest keys.

- **Commit message:** `deploy: COCO-80 num_classes in Frigate snippet and label_space in manifest`

---

## Final Whole-Branch Review (after Task 6)

- `git log --oneline -8` on `master` shows the six commits (top-down: 6→1) with no merge commits and no interleaved foreign changes.
- Full suite: `.venv/bin/python -m pytest -p no:warnings 2>&1 | tail -1` → `N passed` with `N` ≥ previous baseline + new tests (expect ~305–320).
- `ruff` clean on every changed/new source file.
- Spec conformance walkthrough vs `2026-09-11-frozen-head-finetune-design.md`: label_space default `"coco80"`; dataset/build/deploy consistency; masking-only mechanism (no stage-1 backbone freeze, no weight anchoring, no COCO replay); golden + benchmark untouched; index `i` ⇔ `COCO_80[i]` invariant held everywhere.
- The still-dirty working-tree files remain unmodified by this branch.