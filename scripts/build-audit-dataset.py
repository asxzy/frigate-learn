#!/usr/bin/env python3
"""Assemble a frozen training dataset version from the audit training export.

Reads ``<audit.training>/`` (positive/ + hard_negative/ groups written by
``frigate-learn audit run``) and writes ``data/datasets/<version>/`` in the
standard builder layout consumed by ``frigate-learn train``:

    datasets/<version>/
    ├── images/            <split>/<sample_id>.jpg
    ├── labels/            <split>/<sample_id>.txt   (YOLO class cx cy w h)
    ├── train.txt | val.txt | test.txt
    ├── dataset.yaml       (COCO-80 names when label_space=coco80)
    ├── build.json
    └── manifest.jsonl

Class ids from the audit classes.txt ontology are remapped to the training
label space (coco_index for label_space=coco80).  Hard-negative crops (empty
labels) are included in the train split only.  Splits are the same
deterministic 80/10/10 hash assignment the DB builder uses.

Usage: python scripts/build-audit-dataset.py VERSION [--training ROOT] [--overwrite]
"""

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frigate_learn.classes import coco_index  # noqa: E402
from frigate_learn.dataset.manifest import append_record  # noqa: E402
from frigate_learn.dataset.splits import assign_split  # noqa: E402
from frigate_learn.evaluation.golden import write_dataset_yaml  # noqa: E402


def read_provenance(training_root: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    path = training_root / "positive" / "provenance.jsonl"
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        sid = rec.get("sample_id")
        if sid:
            out[sid] = rec
    return out


def remap_line(line: str, ontology: list[str], label_space: str) -> str | None:
    parts = line.split()
    if len(parts) != 5:
        return None
    try:
        cls_id = int(parts[0])
    except ValueError:
        return None
    if cls_id < 0 or cls_id >= len(ontology):
        return None
    name = ontology[cls_id]
    if label_space == "coco80":
        try:
            new_id = coco_index(name)
        except KeyError:
            return None
    else:
        new_id = cls_id
    return f"{new_id} {parts[1]} {parts[2]} {parts[3]} {parts[4]}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("version")
    parser.add_argument("--training", type=Path, default=Path("training_audit"))
    parser.add_argument("--datasets", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.config is not None:
        from frigate_learn.config import load_config

        config = load_config(args.config)
        training_root = config.resolve(config.audit.training)
        datasets_root = config.datasets_dir()
        label_space = config.training.label_space
    else:
        training_root = args.training
        datasets_root = args.datasets or Path("data/datasets")
        label_space = "coco80"

    classes_txt = training_root / "classes.txt"
    if not classes_txt.is_file():
        print(f"missing {classes_txt}", file=sys.stderr)
        return 1
    ontology = [
        name.strip()
        for name in classes_txt.read_text(encoding="utf-8").splitlines()
        if name.strip()
    ]
    if not ontology:
        print(f"empty ontology in {classes_txt}", file=sys.stderr)
        return 1

    target = datasets_root / args.version
    if target.exists() and any(target.iterdir()) and not args.overwrite:
        print(f"dataset version {args.version} already exists at {target}", file=sys.stderr)
        return 1
    if target.exists() and args.overwrite:
        shutil.rmtree(target)

    provenance = read_provenance(training_root)
    image_root = training_root / "positive" / "images"
    label_root = training_root / "positive" / "labels"
    split_counts: dict[str, int] = {}
    written = 0
    dropped = 0
    negatives = 0

    for split in ("train", "val", "test"):
        (target / "images" / split).mkdir(parents=True, exist_ok=True)
        (target / "labels" / split).mkdir(parents=True, exist_ok=True)

    positive_path = training_root / "positive" / "images"
    if positive_path.is_dir():
        for image_path in sorted(positive_path.iterdir()):
            sid = image_path.stem
            label_file = label_root / f"{sid}.txt"
            if not label_file.is_file():
                dropped += 1
                continue
            lines = [
                line
                for line in label_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            remapped = [remap_line(line, ontology, label_space) for line in lines]
            remapped = [line for line in remapped if line is not None]
            if not remapped:
                dropped += 1
                continue
            split = assign_split(sid)
            ext = image_path.suffix.lower() or ".jpg"
            image_dst = target / "images" / split / f"{sid}{ext}"
            shutil.copy2(image_path, image_dst)
            (target / "labels" / split / f"{sid}.txt").write_text(
                "".join(line + "\n" for line in remapped), encoding="utf-8"
            )
            rec = provenance.get(sid, {})
            source = rec.get("source") or {}
            append_record(
                target / "manifest.jsonl",
                {
                    "sample_id": sid,
                    "image": f"{sid}{ext}",
                    "camera": source.get("camera", ""),
                    "timestamp": source.get("timestamp"),
                    "split": split,
                    "source": source.get("type", "audit"),
                    "verified": True,
                    "labels": [line.split()[0] for line in remapped],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            split_counts[split] = split_counts.get(split, 0) + 1
            written += 1

    hard_root = training_root / "hard_negative" / "images"
    if hard_root.is_dir():
        for image_path in sorted(hard_root.iterdir()):
            if image_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                continue
            sid = image_path.stem
            split = "train"
            ext = image_path.suffix.lower() or ".jpg"
            shutil.copy2(image_path, target / "images" / split / f"{sid}{ext}")
            (target / "labels" / split / f"{sid}.txt").write_text("", encoding="utf-8")
            rec = provenance.get(sid, {})
            source = rec.get("source") or {}
            append_record(
                target / "manifest.jsonl",
                {
                    "sample_id": sid,
                    "image": f"{sid}{ext}",
                    "camera": source.get("camera", ""),
                    "timestamp": source.get("timestamp"),
                    "split": split,
                    "source": "audit-hard-negative",
                    "verified": True,
                    "labels": [],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            split_counts[split] = split_counts.get(split, 0) + 1
            written += 1
            negatives += 1

    if written == 0:
        print("nothing to build; audit export empty", file=sys.stderr)
        return 1

    if label_space == "coco80":
        from frigate_learn.classes import COCO_80

        names = list(COCO_80)
    else:
        names = ontology
    write_dataset_yaml(
        target / "dataset.yaml",
        names,
        train="images/train",
        val="images/val",
        test="images/test",
    )
    for split in ("train", "val", "test"):
        files = sorted((target / "images" / split).glob("*"))
        if files:
            (target / f"{split}.txt").write_text(
                "\n".join(
                    f"images/{split}/{p.name}" for p in files if p.is_file()
                )
                + "\n",
                encoding="utf-8",
            )

    payload = {
        "version": args.version,
        "source": "audit",
        "classes": ontology,
        "label_space": label_space,
        "seed": "frigate-learn-v1",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "images_written": written,
            "negatives": negatives,
            "dropped": dropped,
            "split_counts": split_counts,
        },
    }
    (target / "build.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"built {args.version}: positives={written - negatives} "
        f"hard_negatives={negatives} dropped={dropped} "
        f"splits={split_counts}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())