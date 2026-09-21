"""Config loading tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from frigate_learn.config import (
    AppConfig,
    FrigateSettings,
    build_config,
    load_config,
)
from conftest import make_config_file


def test_defaults():
    cfg = AppConfig()
    assert cfg.frigate.base_url == "http://frigate:8971"
    assert cfg.collection.severity == ["detection", "alert"]
    assert cfg.collection.concurrency == 8
    assert cfg.evaluation.golden_dataset == "golden-v001"
    assert len(cfg.classes) >= 8
    assert "person" in cfg.classes
    assert cfg.collection.region_crop is False
    assert cfg.collection.region_crop_height is None
    assert cfg.deployment.max_fp_rate_delta == 0.05
    assert cfg.training.label_space == "coco80"


def test_build_config_maps_sections():
    raw = {
        "classes": ["person", "car"],
        "frigate": {"base_url": "http://cam:8971/", "token": "t"},
        "collection": {"cameras": "front,gate", "severity": ["detection"], "concurrency": 3},
        "deployment": {"max_fp_rate_delta": 0.12},
        "data": {"root": "var"},
    }
    cfg = build_config(raw, base_dir=Path("/tmp/x"))
    assert cfg.frigate.base_url == "http://cam:8971"  # trailing slash stripped
    assert cfg.frigate.token == "t"
    assert cfg.collection.cameras == ["front", "gate"]
    assert cfg.collection.severity == ["detection"]
    assert cfg.collection.concurrency == 3
    assert cfg.deployment.max_fp_rate_delta == 0.12


def test_collection_region_crop_settings():
    raw = {
        "classes": ["person"],
        "collection": {"region_crop": False, "region_crop_height": 512},
        "data": {"root": "var"},
    }
    cfg = build_config(raw, base_dir=Path("/tmp/x"))
    assert cfg.collection.region_crop is False
    assert cfg.collection.region_crop_height == 512
    assert cfg.data.root == "var"
    assert cfg.base_dir == Path("/tmp/x").resolve()


def test_resolve_keeps_consumer_suffixes():
    """Regression: an absolute ``data.root`` must not swallow the consumers
    that follow it.

    Bug: ``resolve()`` returned the first absolute part by itself, so
    ``cfg.resolve(root, "images")`` collapsed to the bare ``root`` directory
    and the database, images, golden, and datasets writers all opened the same
    directory.
    """
    cfg = AppConfig()
    cfg.base_dir = Path("/tmp/x").resolve()
    cfg.data.root = "/abs/root"
    db = cfg.resolve(cfg.data.root, cfg.data.database)
    images = cfg.resolve(cfg.data.root, cfg.data.images)
    golden = cfg.resolve(cfg.data.root, cfg.data.golden)
    assert db == Path("/abs/root") / cfg.data.database
    assert images == Path("/abs/root") / cfg.data.images
    assert golden == Path("/abs/root") / cfg.data.golden
    assert len({db.parent, images, golden}) == 3  # consumers stay distinct


def test_interpolation_and_anchoring(tmp_path, monkeypatch):
    body = """
frigate:
  base_url: "http://frigate:8971"
  token: "${FRIGATE_TOKEN:-fallback}"
  timeout_seconds: 12
collection:
  severity: ["detection"]
data:
  root: "data"
"""
    cfg_path = tmp_path / "config.yaml"
    make_config_file(cfg_path, body)
    monkeypatch.setenv("FRIGATE_TOKEN", "secret")
    cfg = load_config(cfg_path)
    assert cfg.frigate.token == "secret"
    assert cfg.frigate.timeout_seconds == 12.0
    assert cfg.collection.severity == ["detection"]
    # relative paths anchored to the config file's directory
    assert cfg.base_dir == tmp_path.resolve()
    assert cfg.database_path() == tmp_path.resolve() / "data" / "frigate_learn.db"


def test_env_token_default_when_unset(tmp_path):
    body = 'frigate:\n  token: "${FRIGATE_TOKEN:-fallback}"\n'
    cfg_path = tmp_path / "config.yaml"
    make_config_file(cfg_path, body)
    cfg = load_config(cfg_path, overrides={"FRIGATE_TOKEN": ""})
    assert cfg.frigate.token == "fallback"


def test_dotenv_is_used(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text('FRIGATE_TOKEN=fromdotenv\n', encoding="utf-8")
    cfg_path = tmp_path / "config.yaml"
    make_config_file(cfg_path, 'frigate:\n  token: "${FRIGATE_TOKEN}"\n')
    cfg = load_config(cfg_path, env_file=env_file)
    assert cfg.frigate.token == "fromdotenv"


def test_missing_config_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no config.example.yaml fallback here
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_frigate_defaults_dataclass():
    f = FrigateSettings()
    assert f.completed_events_only is True
    assert f.max_retries == 3


def test_training_label_space_default_coco80():
    cfg = AppConfig()
    assert cfg.training.label_space == "coco80"
    assert cfg.training.lr0 is None
    assert cfg.training.lrf is None


def test_build_config_label_space_subset():
    cfg = build_config({"training": {"label_space": "subset"}}, base_dir=Path("/tmp/x"))
    assert cfg.training.label_space == "subset"


def test_build_config_invalid_label_space_raises():
    with pytest.raises(ValueError):
        build_config({"training": {"label_space": "bbox"}}, base_dir=Path("/tmp/x"))


def test_build_config_lr_overrides():
    cfg = build_config(
        {"training": {"lr0": 0.001, "lrf": 0.01}}, base_dir=Path("/tmp/x")
    )
    assert cfg.training.lr0 == 0.001
    assert cfg.training.lrf == 0.01
    cfg = build_config({"training": {"lr0": "", "lrf": ""}}, base_dir=Path("/tmp/x"))
    assert cfg.training.lr0 is None
    assert cfg.training.lrf is None


def test_model_class_names_coco80():
    names = AppConfig().model_class_names()
    assert len(names) == 80
    assert names[0] == "person"
    assert names[2] == "car"
    assert names[15] == "cat"
    assert names[16] == "dog"


def test_model_class_names_subset_matches_classes():
    cfg = build_config(
        {"classes": ["person", "car"], "training": {"label_space": "subset"}},
        base_dir=Path("/tmp/x"),
    )
    assert cfg.model_class_names() == ["person", "car"]


def test_trainable_class_mask_coco80():
    mask = AppConfig().trainable_class_mask()
    assert len(mask) == 80
    assert sum(mask) == 8
    for i in [0, 1, 2, 3, 5, 7, 15, 16]:
        assert mask[i] is True
    for i in [4, 79]:
        assert mask[i] is False


def test_trainable_class_mask_subset_is_none():
    cfg = build_config({"training": {"label_space": "subset"}}, base_dir=Path("/tmp/x"))
    assert cfg.trainable_class_mask() is None


def test_coco80_rejects_non_coco_class():
    with pytest.raises(ValueError, match="deer"):
        build_config({"classes": ["person", "deer"]}, base_dir=Path("/tmp/x"))

def test_audit_sam_remote_defaults():
    cfg = AppConfig()
    assert cfg.audit.sam.backend == "mlx"
    assert cfg.audit.sam.base_url == ""
    assert cfg.audit.sam.api_key == ""
    assert cfg.audit.sam.model == ""
    assert cfg.audit.sam.timeout_seconds == 120.0
    assert cfg.audit.sam.max_retries == 2


def test_build_config_maps_remote_sam_settings():
    raw = {
        "classes": ["person"],
        "audit": {
            "models": {
                "sam": {
                    "backend": "http",
                    "base_url": "http://sam-host:8001/",
                    "api_key": "sekrit",
                    "model": "sam3-1",
                    "timeout_seconds": 90,
                    "max_retries": 4,
                }
            }
        },
    }
    cfg = build_config(raw, base_dir=Path("/tmp/x"))
    assert cfg.audit.sam.backend == "http"
    assert cfg.audit.sam.base_url == "http://sam-host:8001"  # trailing slash stripped
    assert cfg.audit.sam.api_key == "sekrit"
    assert cfg.audit.sam.model == "sam3-1"
    assert cfg.audit.sam.timeout_seconds == 90.0
    assert cfg.audit.sam.max_retries == 4


def test_build_config_rejects_unknown_sam_backend():
    with pytest.raises(ValueError, match="backend"):
        build_config(
            {"audit": {"models": {"sam": {"backend": "tpu"}}}},
            base_dir=Path("/tmp/x"),
        )

