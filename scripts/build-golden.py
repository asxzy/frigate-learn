#!/usr/bin/env python3
"""Build the golden evaluation dataset from audited positives.

Selects audit KEEP samples whose deterministic split assignment is ``test``
(disjoint from the training split by construction), remaps their labels to
``config.classes`` index ids, and writes ``data/golden/golden-v001/`` via the
immutable GoldenDataset API (append-only manifest).

Usage: python scripts/build-golden.py [--config config.yaml] [--split test]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frigate_learn.config import load_config  # noqa: E402
from frigate_learn.dataset.splits import assign_split  # noqa: E402
from frigate_learn.dataset.yolo import YoloLine  # noqa: E402
from frigate_learn.evaluation.golden import GoldenDataset  # noqa: E402


def load_provenance(training_root: Path) -> dict[str, dict]:
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
        source = rec.get("source") or {}
        if isinstance(source, dict) and source.get("sample_id"):
            out.setdefault(str(source["sample_id"]), rec)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config(None)
    classes = list(config.classes)
    training_root = config.resolve(config.audit.training)
    golden_root = config.golden_dir() / config.evaluation.golden_dataset

    classes_txt = training_root / "classes.txt"
    if not classes_txt.is_file():
        print(f"missing {classes_txt}", file=sys.stderr)
        return 1
    ontology = [
        name.strip()
        for name in classes_txt.read_text(encoding="utf-8").splitlines()
        if name.strip()
    ]
    name_to_idx = {name: i for i, name in enumerate(classes)}

    golden = (
        GoldenDataset.create(golden_root, classes)
        if not golden_root.is_dir()
        else GoldenDataset.load(golden_root)
    )
    provenance = load_provenance(training_root)
    image_root = training_root / "positive" / "images"
    label_root = training_root / "positive" / "labels"

    added = 0
    skipped = 0
    existing = {r.get("sample_id") for r in golden.manifest}
    by_class: dict[str, int] = {}
    for image_path in sorted(image_root.iterdir()):
        sid = image_path.stem
        if sid in existing:
            skipped += 1
            continue
        if assign_split(sid) != args.split:
            skipped += 1
            continue
        label_file = label_root / f"{sid}.txt"
        if not label_file.is_file():
            skipped += 1
            continue
        lines = [
            line
            for line in label_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        yolo_lines = []
        ok = True
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                ok = False
                break
            try:
                cls_id = int(parts[0])
            except ValueError:
                ok = False
                break
            if cls_id < 0 or cls_id >= len(ontology):
                ok = False
                break
            name = ontology[cls_id]
            if name not in name_to_idx:
                ok = False
                break
            yolo_lines.append(
                YoloLine(
                    class_id=name_to_idx[name],
                    cx=float(parts[1]),
                    cy=float(parts[2]),
                    w=float(parts[3]),
                    h=float(parts[4]),
                )
            )
        if not ok or not yolo_lines:
            skipped += 1
            continue
        rec = provenance.get(sid, {})
        source = rec.get("source") or {}
        try:
            golden.add_image(
                sid,
                image_path,
                yolo_lines,
                camera=str(source.get("camera", "")),
                timestamp=float(source.get("timestamp") or 0.0),
                source="audit",
            )
        except Exception as exc:
            print(f"skip {sid}: {exc}", file=sys.stderr)
            skipped += 1
            continue
        added += 1
        for yolo_line in yolo_lines:
            by_class[yolo_line.class_id] = by_class.get(yolo_line.class_id, 0) + 1

    problems = golden.validate()
    print(
        f"golden {golden.root.name}: added={added} skipped={skipped} "
        f"total={len(golden.manifest)}"
    )
    if by_class:
        print("by class:", {classes[k]: v for k, v in sorted(by_class.items())})
    if problems:
        print("problems:", problems, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())