# Frozen-Head Fine-Tuning — Design Spec

Status: approved. Plan: `docs/superpowers/plans/2026-09-11-frozen-head-finetune.md`.

## 1. Problem

Fine-tuning a pretrained COCO-80 YOLO on only the 8 configured classes risks degrading the 72 non-configured classes — useful context in the camera frame (e.g. `tv`, `laptop`, `chair`) that Frigate may still report. The model head is 80-wide; simply passing an 8-class `dataset.yaml` rebuilds the whole head as a **new random 8-class head**, destroying the pretrained rows and permanently narrowing the labelmap. The goal: train the 8 camera-relevant rows hard, leave the other 72 rows — and any backbone feedback driven by them — untouched, and keep the deployment a drop-in 80-class model.

## 2. Approach (locked)

- **Label space `"coco80"` (default):** the model output index space is the COCO-80 order (`output index i ≡ COCO_80[i]`). Datasets, `dataset.yaml` (`nc=80`), the trained head, the export/HEF, and the Frigate snippet all stay in that space. The 8 configured classes are the *trainable subset*; indices never reorder.
- **Label space `"subset"`:** historical behavior — compact head over `config.classes`, plain `YOLO().train()`.
- **Frozen-by-loss-mask, not by `requires_grad`:** YOLOv8 cls loss is per-class BCE. Freezing per-class rows is *not* achievable with `freeze` (layer slices) or `requires_grad=False` (per-row). Concretely: zero the frozen columns of `bce_loss` *before* `sum()`, and make `target_scores_sum` sum only trainable columns. Then the frozen rows compute zero loss and exactly zero gradient, and contribute nothing to the shared backbone.
- **No weight anchoring, no stage-1 backbone freeze** for MVP (orthogonal knobs; may compose later). No COCO replay of the other 72 classes for MVP — recorded as deferred work (§9), because replay needs a large external corpus and a licensing/labeling pass; a later change can add it without disturbing this design.
- **Golden set and benchmark unchanged.** Matching is name-based via `result.names`; a coco80-trained model reports the 80 COCO names, so existing benchmark/gate logic needs no edits.
- **Mechanism seam (verified on ultralytics 8.4.146):** `DetectionModel` lazily creates `self.criterion` (a `v8DetectionLoss`) inside `BaseModel.loss`; both the uncompiled (`model(batch)`) and compiled (`model.loss(batch, preds)`) training call sites end at `self.criterion`. Therefore `MaskedDetectionTrainer.get_model` builds the model via `super()` and attaches `model.criterion = MaskedDetectionLoss(model, trainable=...)`. The masked loss subclasses `v8DetectionLoss` and re-expresses only the cls term (`get_assigned_targets_and_loss`) through a pure helper `mask_cls_loss`. (Overriding `self.class_weights` or wrapping `self.bce` is rejected: both are leased to other semantics and would mask state shared with unrelated code.)

## 3. Actors / components

- `src/frigate_learn/classes.py` (new, pure std-lib): `COCO_80`, `coco_index`, `collect_indices`, `trainable_mask`.
- `config.py`: `TrainingSettings.label_space` (`"coco80"`|`"subset"`), `lr0`, `lrf`; `AppConfig.model_class_names()`, `AppConfig.trainable_class_mask()`; load-time validation (coco80 requires all configured classes to be COCO names — `deer` fails fast at load).
- `dataset/builder.py`: class ids = COCO indices under coco80; `dataset.yaml` = 80 names; `build.json` records `label_space` + `trainable` mask.
- `training/masked.py` (new): `mask_cls_loss` (pure torch), `MaskedDetectionLoss(v8DetectionLoss)`, `MaskedDetectionTrainer(DetectionTrainer)`.
- `training/trainer.py`: coco80 branch → `MaskedDetectionTrainer(overrides={...}, trainable=mask)`; subset branch unchanged.
- `deploy/hailo.py`: `num_classes` from `model_class_names()` (80 under coco80); manifest gains `label_space` + `num_classes`.

## 4. Data model / thresholds

No DB or schema changes. Config-only knobs: `training.label_space`, `training.lr0`, `training.lrf` (all optional/None-tolerant).

## 5. API surface

- `classes.coco_index(name) -> int` (ValueError unknown), `classes.collect_indices(names) -> list[int]`, `classes.trainable_mask(nc, names) -> list[bool]`.
- `AppConfig.model_class_names() -> list[str]`, `AppConfig.trainable_class_mask() -> list[bool] | None`.
- `training.masked.mask_cls_loss(bce_loss, target_scores, mask) -> Tensor`.
- `training.masked.MaskedDetectionLoss(model, *, trainable)` — subclass, cls-masked.
- `training.masked.MaskedDetectionTrainer(trainable=None, **kwargs)` — subclass; `get_model` attaches the masked criterion.
- `dataset/builder` build.json keys `label_space`, `trainable`.
- `deploy/hailo.py` manifest keys `label_space`, `num_classes`.

## 6. Failure modes and invariants

- Frozen column must contribute exactly `0` to cls loss and gradient — the provoked-invariant: `mask_cls_loss` zero-gradient test on a frozen positive, plus a real-model `cls_loss == 0.0` when only frozen classes are present.
- Denominator never `0` (the installed `max(..., 1)` guard is preserved).
- Index↔name invariant: no code path may reorder labels; datasets/yaml/head/export/snippet all derive from `model_class_names()`.
- Omission invariant: `trainable_class_mask()` returns `None` for `"subset"` — nothing in the subset path may touch masking.
- Optionality: no `torch`/`ultralytics` import at package import time.

## 7. Test plan

- `test_classes.py`: catalog length/uniqueness, spot indices (0,1,2,3,5,7,15,16,79), unknown-name ValueError, mask shape.
- `test_config.py`: label_space default/parse/invalid; lr0/lrf; `model_class_names`/`trainable_class_mask` both modes; `deer` rejection under coco80.
- `test_dataset_builder.py`: label-file ids are COCO indices; yaml has 80 names; build.json label_space + trainable; non-configured COCO and non-COCO labels dropped; subset stays compact.
- `test_masked.py` (torch-guarded): all-trainable == plain; frozen-zero; frozen-column zero gradient; real model forward `cls_loss == 0` when only frozen classes; trainer attaches criterion; empty/bad-index validation.
- `test_trainer.py`: coco80 constructs MaskedDetectionTrainer with exact overrides; None overrides omitted; lr/freeze forwarded; subset keeps YOLO path.
- `test_deploy.py`: `num_classes: 80` under coco80, compact under subset; manifest carries `label_space`/`num_classes`.
- Regression: full suite green (≥ 273 baseline + new).

## 8. Migration plan

None needed at runtime; the new config keys default to existing behavior (subset) on omission only via explicit `label_space: subset` — default `coco80` changes builder output shape, so **any existing `data/datasets/*` must be rebuilt** when upgrading with an existing config that leaves `label_space` unset (default is coco80). Document in `config.example.yaml` that existing runs should pin `label_space: subset` or rebuild datasets.

## 9. Deferred

- COCO replay / pseudo-labeling of the 72 frozen classes to keep them accurate while improving the 8.
- Weight anchoring (per-row regularization toward the pretrained weights).
- Stage-1 backbone freeze behind `training.freeze` composition with masking.
- VLM-driven head pruning (drop frozen head rows at export and shrink the labelmap).

## 10. Risks and mitigations

- **Ultralytics internal coupling** (the body copy): mitigated by a review gate that diffs the copied method body against the installed 8.4.146 source on the machine, and by pinning the `ml` environment to that version.
- **Frozen rows silent-degrade over time:** acceptable MVP trade-off; deferred §9 items are the remedy and options are recorded per version epoch in `results.csv` only if training continues on later builds.
- **Existing-cache confusion** (built datasets from the pre-change builder): mitigated by §8 migration note and the `label_space` build.json key.