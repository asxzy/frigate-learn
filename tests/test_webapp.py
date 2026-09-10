"""Webapp tests (require the `web` extra: fastapi + uvicorn)."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from frigate_learn.config import build_config
from frigate_learn.webapp.app import create_app


def make_client(tmp_path) -> TestClient:
    cfg = build_config({}, tmp_path)
    return TestClient(create_app(cfg))


def test_health(tmp_path):
    c = make_client(tmp_path)
    r = c.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_serves_spa(tmp_path):
    c = make_client(tmp_path)
    r = c.get("/")
    assert r.status_code == 200
    assert "frigate-learn" in r.text


def test_app_uses_default_config_without_arg(tmp_path, monkeypatch):
    monkeypatch.setenv("FRIGATE_LEARN_CONFIG", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("data:\n  root: data\n", encoding="utf-8")
    client = TestClient(create_app())
    assert client.get("/api/health").status_code == 200


import json

from frigate_learn.db import Database
from frigate_learn.evaluation.benchmark import CandidateResult, results_to_json
from frigate_learn.evaluation.metrics import DetectionMetrics
from frigate_learn.models import Annotation, Deployment, Job, Sample
from frigate_learn.run import PIPELINE
from frigate_learn.webapp import queries

REAL_CSV_HEADER = (
    "epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,"
    "metrics/precision(B),metrics/recall(B),metrics/mAP50(B),"
    "metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss,"
    "lr/pg0,lr/pg1,lr/pg2"
)
CSV_ROW_1 = "1,3.31269,1.83577,4.85717,1.82252,0.5,0.6,0.5,0.3,1.46899,4.50424,1.27657,0.0002,0.0002,0.0002"
CSV_ROW_2 = "2,6.23219,2.26697,5.01956,2.24914,0.6,0.7,0.7,0.4,1.56016,4.48987,1.33118,0.0004,0.0004,0.0004"


def _metrics(map50, recall, latency_ms):
    precision = 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return DetectionMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        map50=map50,
        map50_95=map50 * 0.8,
        small_object_recall=0.0,
        false_positive_rate=0.0,
        latency_ms=latency_ms,
        throughput_fps=1000.0 / latency_ms if latency_ms else None,
    )


def _seed_tree(tmp_path) -> None:
    data = tmp_path / "data"
    (data / "images").mkdir(parents=True)
    (data / "images" / "a.jpg").write_bytes(b"fake-jpeg")

    golden = data / "golden" / "golden-v001"
    (golden / "images").mkdir(parents=True)
    (golden / "labels").mkdir(parents=True)
    (golden / "dataset.yaml").write_text("names:\n  0: person\n  1: car\n", encoding="utf-8")
    (golden / "images" / "a.jpg").write_bytes(b"fake-jpeg")
    (golden / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    ds = data / "datasets" / "v001"
    ds.mkdir(parents=True)
    (ds / "build.json").write_text(
        json.dumps({
            "version": "v001",
            "built_at": "2026-09-09T20:28:15.093812+00:00",
            "classes": ["person", "car", "bicycle"],
            "summary": {
                "images_written": 21,
                "total": 160,
                "verified": 5,
                "skipped_class": 139,
                "skipped_no_box": 0,
                "skipped_cap": 0,
                "split_counts": {"train": 20, "val": 1},
            },
        }, indent=2),
        encoding="utf-8",
    )
    (data / "datasets" / "v002").mkdir(parents=True)

    tr = data / "training" / "yolov8n-v001"
    (tr / "weights").mkdir(parents=True)
    (tr / "weights" / "best.pt").write_bytes(b"model")
    (tr / "results.csv").write_text(
        REAL_CSV_HEADER + "\n" + CSV_ROW_1 + "\n" + CSV_ROW_2 + "\n",
        encoding="utf-8",
    )

    results_to_json(
        [
            CandidateResult(name="yolov8n", metrics=_metrics(0.6, 0.7, 8.0), version="golden-v001"),
            CandidateResult(name="yolov8s", metrics=_metrics(0.75, 0.8, 5.0), version="golden-v001"),
        ],
        data / "benchmark-results.json",
    )


def _seed_db(db, image_path) -> None:
    with db.session() as session:
        session.add_all([
            Sample(
                id="s1",
                camera="front",
                timestamp=300.0,
                image_path=str(image_path),
                frigate_label="person",
                frigate_score=0.9,
                frigate_x1=0.05,
                frigate_y1=0.05,
                frigate_x2=0.6,
                frigate_y2=0.8,
                quality="useful",
                status="collected",
                verified=1,
                created_at="2026-09-09T00:00:00+00:00",
            ),
            Sample(
                id="s2",
                camera="front",
                timestamp=200.0,
                frigate_label="car",
                frigate_score=0.8,
                status="collected",
                verified=0,
                created_at="2026-09-09T00:00:01+00:00",
            ),
            Sample(
                id="s3",
                camera="back",
                timestamp=100.0,
                quality="bad",
                status="reviewed",
                verified=0,
                created_at="2026-09-09T00:00:02+00:00",
            ),
            Deployment(
                id="d1",
                model_name="yolov8n",
                version="golden-v001",
                metrics_json=json.dumps({"map50": 0.6}),
                verdict="PASS",
                reasons_json=json.dumps(["passes gate"]),
                deployed=1,
                deployed_at="2026-09-09T00:00:00+00:00",
                created_at="2026-09-09T00:01:00+00:00",
            ),
            Deployment(
                id="d2",
                model_name="yolov8s",
                version="golden-v001",
                metrics_json=json.dumps({"map50": 0.75}),
                verdict="FAIL",
                reasons_json=None,
                deployed=0,
                deployed_at=None,
                created_at="2026-09-09T00:00:00+00:00",
            ),
            Job(
                id="j1",
                type="build",
                status="finished",
                started_at="2026-09-09T00:00:00+00:00",
                finished_at="2026-09-09T00:01:00+00:00",
                error=None,
            ),
        ])
        session.commit()
    with db.session() as session:
        session.add_all([
            Annotation(
                id="a1",
                sample_id="s1",
                source="vlm",
                label="person",
                x1=0.1,
                y1=0.1,
                x2=0.5,
                y2=0.5,
                confidence=0.95,
                verified=1,
                created_at="2026-09-09T00:00:00+00:00",
            ),
            Annotation(
                id="a2",
                sample_id="s2",
                source="frigate",
                label="car",
                x1=0.2,
                y1=0.2,
                x2=0.6,
                y2=0.6,
                confidence=0.8,
                verified=0,
                created_at="2026-09-09T00:00:00+00:00",
            ),
        ])
        session.commit()


@pytest.fixture
def seeded(tmp_path):
    cfg = build_config({"data": {"root": "data"}}, tmp_path)
    _seed_tree(tmp_path)
    db = Database(cfg.database_path())
    db.init()
    _seed_db(db, tmp_path / "data" / "images" / "a.jpg")
    yield cfg, db
    db.dispose()


def test_overview(seeded):
    cfg, db = seeded
    result = queries.overview(cfg, db)
    assert result["steps"] == PIPELINE
    assert result["auto_enable"] == ["collect", "build"]
    assert result["samples"] == {
        "total": 3,
        "verified": 1,
        "unverified": 2,
        "quality_useful": 1,
        "quality_bad": 1,
        "quality_duplicate": 0,
        "quality_ignore": 0,
    }
    assert result["cameras"] == ["back", "front"]
    assert result["vlm_enabled"] is False
    assert result["disk_free_bytes"] > 0
    assert result["next_build_version"] == "v003"
    assert result["pipeline"][0]["id"] == "j1"
    assert result["pipeline"][0]["status"] == "finished"


def test_benchmark(seeded):
    cfg, _ = seeded
    result = queries.benchmark(cfg)
    assert result["golden"] == "golden-v001"
    assert result["baseline"] == "yolov8l"
    assert result["max_latency_ms"] == 20.0
    assert [r["name"] for r in result["results"]] == ["yolov8n", "yolov8s"]
    assert result["results"][0]["metrics"]["map50"] == 0.6
    assert result["pareto"] == ["yolov8s"]
    assert isinstance(result["results_updated_at"], str)


def test_benchmark_missing_file_returns_empty(tmp_path):
    cfg = build_config({"data": {"root": "data"}}, tmp_path)
    result = queries.benchmark(cfg)
    assert result["results"] == []
    assert result["pareto"] == []
    assert result["results_updated_at"] is None


def test_deployments(seeded):
    _, db = seeded
    result = queries.deployments(db)
    assert [d["model_name"] for d in result["deployments"]] == ["yolov8n", "yolov8s"]
    top = result["deployments"][0]
    assert top["verdict"] == "PASS"
    assert top["metrics"] == {"map50": 0.6}
    assert top["reasons"] == ["passes gate"]
    assert top["deployed"] == 1
    assert top["deployed_at"] == "2026-09-09T00:00:00+00:00"
    bottom = result["deployments"][1]
    assert bottom["verdict"] == "FAIL"
    assert bottom["reasons"] is None
    assert bottom["deployed"] == 0
    assert bottom["deployed_at"] is None


def test_quality_counts(seeded):
    _, db = seeded
    result = queries.quality_counts(db)
    assert result["total"] == 3
    assert result["verified"] == 1
    assert result["unverified"] == 2
    assert result["by_quality"] == {"useful": 1, "bad": 1, "duplicate": 0, "ignore": 0}
    assert result["per_class"] == {"person": 1}
    assert result["one_class_boxes"] == 1
    assert result["problematic"] == 1


def test_datasets(seeded):
    cfg, _ = seeded
    result = queries.datasets(cfg)
    versions = result["versions"]
    assert versions[0] == {"version": "v002", "error": "build.json missing"}
    v1 = versions[1]
    assert v1["version"] == "v001"
    assert v1["built_at"] == "2026-09-09T20:28:15.093812+00:00"
    assert v1["images_written"] == 21
    assert v1["total"] == 160
    assert v1["verified"] == 5
    assert v1["skipped_class"] == 139
    assert v1["split_counts"] == {"train": 20, "val": 1}
    assert v1["classes"] == ["person", "car", "bicycle"]
    assert result["golden"] == {
        "version": "golden-v001",
        "exists": True,
        "classes": ["person", "car"],
        "image_count": 1,
        "per_class": {"person": 1},
    }


def test_training_index(seeded):
    cfg, _ = seeded
    result = queries.training_index(cfg)
    assert len(result["runs"]) == 1
    run = result["runs"][0]
    assert run["run"] == "yolov8n-v001"
    assert run["model"] == "yolov8n"
    assert run["tag"] == "v001"
    assert run["dataset"] == "v001"
    assert isinstance(run["results_mtime"], str)
    assert run["latest_map50"] == 0.7
    assert run["best_map50"] == 0.7
    assert run["has_weights"] is True


def test_training_index_skips_dirs_without_csv(tmp_path):
    cfg = build_config({"data": {"root": "data"}}, tmp_path)
    (tmp_path / "data" / "training" / "yolov8n-v001").mkdir(parents=True)
    assert queries.training_index(cfg)["runs"] == []


def test_training_run(seeded):
    cfg, _ = seeded
    result = queries.training_run(cfg, "yolov8n-v001")
    assert result["run"] == "yolov8n-v001"
    assert result["dataset"] == "v001"
    assert result["columns"][0] == "epoch"
    assert result["epochs"] == 2
    assert result["best_map50"] == 0.7
    assert result["latest_map50"] == 0.7
    assert result["weights_exists"] is True
    assert result["rows"][0]["epoch"] == 1.0
    assert result["rows"][0]["metrics/mAP50(B)"] == 0.5
    assert result["rows"][1]["metrics/recall(B)"] == 0.7


def test_training_run_unknown_returns_none(seeded):
    cfg, _ = seeded
    assert queries.training_run(cfg, "does-not-exist") is None


def test_training_run_missing_map50_column(tmp_path):
    cfg = build_config({"data": {"root": "data"}}, tmp_path)
    run_dir = tmp_path / "data" / "training" / "yolov8n-v999"
    run_dir.mkdir(parents=True)
    (run_dir / "results.csv").write_text(
        "epoch,metrics/recall(B)\n1,0.5\n",
        encoding="utf-8",
    )
    result = queries.training_run(cfg, "yolov8n-v999")
    assert result["best_map50"] is None
    assert result["latest_map50"] is None
    assert result["epochs"] == 1


def test_samples_all(seeded):
    _, db = seeded
    result = queries.samples(db)
    assert result["total"] == 3
    assert result["limit"] == 50
    assert result["offset"] == 0
    assert [s["id"] for s in result["samples"]] == ["s1", "s2", "s3"]
    s1 = result["samples"][0]
    assert s1["camera"] == "front"
    assert s1["timestamp"] == 300.0
    assert s1["label"] == "person"
    assert s1["score"] == 0.9
    assert s1["quality"] == "useful"
    assert s1["status"] == "collected"
    assert s1["verified"] == 1
    assert s1["has_image"] is True
    assert s1["has_annotations"] is True
    assert s1["image_url"] == "/images/s1"
    assert s1["thumb_url"] == "/images/s1/thumb"
    s2 = result["samples"][1]
    assert s2["has_annotations"] is True
    assert s2["verified"] == 0


def test_samples_filters(seeded):
    _, db = seeded
    assert [s["id"] for s in queries.samples(db, verified=1)["samples"]] == ["s1"]
    assert [s["id"] for s in queries.samples(db, quality="bad")["samples"]] == ["s3"]
    assert {s["id"] for s in queries.samples(db, camera="front")["samples"]} == {"s1", "s2"}
    assert [s["id"] for s in queries.samples(db, status="reviewed")["samples"]] == ["s3"]


def test_samples_verified_exact_match(seeded):
    _, db = seeded
    assert [s["id"] for s in queries.samples(db, verified=0)["samples"]] == ["s2", "s3"]
    assert [s["id"] for s in queries.samples(db, verified=1)["samples"]] == ["s1"]


def test_samples_limit_clamp(seeded):
    _, db = seeded
    result = queries.samples(db, limit=0)
    assert result["limit"] == 1
    assert len(result["samples"]) == 1
    result = queries.samples(db, limit=999)
    assert result["limit"] == 200
    assert result["total"] == 3
    assert len(result["samples"]) == 3


def test_sample_detail(seeded):
    _, db = seeded
    result = queries.sample_detail(db, "s1")
    assert result["sample"]["id"] == "s1"
    assert result["sample"]["timestamp"] == 300.0
    assert result["sample"]["created_at"] == "2026-09-09T00:00:00+00:00"
    assert result["sample"]["has_image"] is True
    assert result["sample"]["image_url"] == "/images/s1"
    assert result["image_url"] == "/images/s1"
    assert result["thumb_url"] == "/images/s1/thumb"
    assert result["annotations"] == [
        {
            "source": "vlm",
            "label": "person",
            "box": [0.1, 0.1, 0.5, 0.5],
            "confidence": 0.95,
            "verified": 1,
        }
    ]
    assert result["frigate"] == {
        "label": "person",
        "score": 0.9,
        "box": [0.05, 0.05, 0.6, 0.8],
    }


def test_sample_detail_missing_returns_none(seeded):
    _, db = seeded
    assert queries.sample_detail(db, "nope") is None


def test_sample_detail_no_frigate_no_image(seeded):
    _, db = seeded
    result = queries.sample_detail(db, "s3")
    assert result["sample"]["has_image"] is False
    assert result["sample"]["has_annotations"] is False
    assert "image_url" not in result
    assert "thumb_url" not in result
    assert result["frigate"] is None
    assert result["annotations"] == []


def test_empty_tree_returns_empty_shapes(tmp_path):
    cfg = build_config({"data": {"root": "data"}}, tmp_path)
    db = Database(cfg.database_path())
    db.init()
    assert queries.benchmark(cfg)["results"] == []
    assert queries.training_index(cfg)["runs"] == []
    assert queries.datasets(cfg)["versions"] == []
    assert queries.datasets(cfg)["golden"]["exists"] is False
    assert queries.deployments(db)["deployments"] == []
    assert queries.samples(db) == {"total": 0, "limit": 50, "offset": 0, "samples": []}
    db.dispose()


# --- Image serving tests ---


from PIL import Image as PILImage

from frigate_learn.webapp.serving import resolve_image, resolve_thumb


def _make_jpeg(path: Path, *, w: int = 600, h: int = 400) -> None:
    img = PILImage.new("RGB", (w, h), color=(255, 0, 0))
    img.save(path, quality=90)


def _serving_cfg(tmp_path):
    return build_config({"data": {"root": "data"}}, tmp_path)


def _serve_db(cfg):
    db = Database(cfg.database_path())
    db.init()
    return db


def test_resolve_image_present(tmp_path):
    cfg = _serving_cfg(tmp_path)
    img_dir = cfg.images_dir() / "nested"
    img_dir.mkdir(parents=True)
    img_path = img_dir / "a.jpg"
    _make_jpeg(img_path)
    db = _serve_db(cfg)
    with db.session() as session:
        session.add(Sample(
            id="img1", camera="c", timestamp=1.0,
            image_path=str(img_path), status="collected", verified=0,
            created_at="2026-09-09T00:00:00+00:00",
        ))
        session.commit()
    result = resolve_image(cfg, db, "img1")
    assert result is not None
    assert result == img_path
    assert result.read_bytes() == img_path.read_bytes()
    db.dispose()


def test_resolve_image_missing_sample(tmp_path):
    cfg = _serving_cfg(tmp_path)
    db = _serve_db(cfg)
    assert resolve_image(cfg, db, "nonexistent") is None
    db.dispose()


def test_resolve_image_traversal(tmp_path):
    cfg = _serving_cfg(tmp_path)
    db = _serve_db(cfg)
    with db.session() as session:
        session.add(Sample(
            id="evil", camera="c", timestamp=1.0,
            image_path="/etc/passwd", status="collected", verified=0,
            created_at="2026-09-09T00:00:00+00:00",
        ))
        session.commit()
    assert resolve_image(cfg, db, "evil") is None
    db.dispose()


def test_resolve_thumb_builds(tmp_path):
    cfg = _serving_cfg(tmp_path)
    img_dir = cfg.images_dir()
    img_dir.mkdir(parents=True)
    img_path = img_dir / "big.jpg"
    _make_jpeg(img_path, w=800, h=600)
    db = _serve_db(cfg)
    with db.session() as session:
        session.add(Sample(
            id="t1", camera="c", timestamp=1.0,
            image_path=str(img_path), status="collected", verified=0,
            created_at="2026-09-09T00:00:00+00:00",
        ))
        session.commit()
    result = resolve_thumb(cfg, db, "t1")
    assert result is not None
    assert result.exists()
    assert result.suffix == ".jpg"
    magic = result.read_bytes()[:2]
    assert magic == b"\xff\xd8"
    with PILImage.open(result) as thumb:
        assert thumb.width <= 480
    thumbs_dir = cfg.previews_dir() / "thumbs"
    assert thumbs_dir.is_dir()
    db.dispose()


def test_resolve_thumb_cached(tmp_path):
    cfg = _serving_cfg(tmp_path)
    img_dir = cfg.images_dir()
    img_dir.mkdir(parents=True)
    img_path = img_dir / "cached.jpg"
    _make_jpeg(img_path)
    db = _serve_db(cfg)
    with db.session() as session:
        session.add(Sample(
            id="t2", camera="c", timestamp=1.0,
            image_path=str(img_path), status="collected", verified=0,
            created_at="2026-09-09T00:00:00+00:00",
        ))
        session.commit()
    first = resolve_thumb(cfg, db, "t2")
    assert first is not None
    mtime1 = first.stat().st_mtime_ns
    second = resolve_thumb(cfg, db, "t2")
    assert second == first
    mtime2 = second.stat().st_mtime_ns
    assert mtime1 == mtime2
    db.dispose()


def test_resolve_thumb_stale_returns_none(tmp_path):
    cfg = _serving_cfg(tmp_path)
    thumbs_dir = cfg.previews_dir() / "thumbs"
    thumbs_dir.mkdir(parents=True)
    stale = thumbs_dir / "gone.jpg"
    stale.write_bytes(b"\xff\xd8\xff\xe0")
    db = _serve_db(cfg)
    with db.session() as session:
        session.add(Sample(
            id="gone", camera="c", timestamp=1.0,
            image_path="/nonexistent/path.jpg", status="collected", verified=0,
            created_at="2026-09-09T00:00:00+00:00",
        ))
        session.commit()
    assert resolve_thumb(cfg, db, "gone") is None
    db.dispose()


def test_resolve_thumb_corrupt_image(tmp_path):
    cfg = _serving_cfg(tmp_path)
    img_dir = cfg.images_dir()
    img_dir.mkdir(parents=True)
    bad = img_dir / "bad.jpg"
    bad.write_text("not an image", encoding="utf-8")
    db = _serve_db(cfg)
    with db.session() as session:
        session.add(Sample(
            id="corrupt", camera="c", timestamp=1.0,
            image_path=str(bad), status="collected", verified=0,
            created_at="2026-09-09T00:00:00+00:00",
        ))
        session.commit()
    assert resolve_thumb(cfg, db, "corrupt") is None
    db.dispose()