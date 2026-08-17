"""Evaluate one checkpoint on untouched and cleaned validation snapshots."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
from ultralytics import YOLO


NAMES = {0: "Empty", 1: "Tweezers", 2: "Needle_driver"}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_yaml(root: Path, output: Path) -> None:
    output.write_text(
        f"path: {root}\ntrain: images/val\nval: images/val\n\nnames:\n"
        "  0: Empty\n  1: Tweezers\n  2: Needle_driver\n"
    )


def metric_dict(metrics) -> dict:
    values = dict(getattr(metrics, "results_dict", {}))
    return {
        "precision": float(values.get("metrics/precision(B)", metrics.box.mp)),
        "recall": float(values.get("metrics/recall(B)", metrics.box.mr)),
        "mAP50": float(values.get("metrics/mAP50(B)", metrics.box.map50)),
        "mAP50-95": float(values.get("metrics/mAP50-95(B)", metrics.box.map)),
        "fitness": float(values.get("fitness", metrics.fitness)),
        "per_class": {
            NAMES.get(class_id, str(class_id)): {
                "precision": float(metrics.box.p[class_id]) if len(metrics.box.p) > class_id else None,
                "recall": float(metrics.box.r[class_id]) if len(metrics.box.r) > class_id else None,
                "mAP50": float(metrics.box.ap50[class_id]) if len(metrics.box.ap50) > class_id else None,
                "mAP50-95": float(metrics.box.maps[class_id]) if len(metrics.box.maps) > class_id else None,
            }
            for class_id in range(len(NAMES))
        },
    }


def training_summary(results_csv: Path | None) -> dict | None:
    if not results_csv:
        return None
    table = pd.read_csv(results_csv)
    table.columns = [column.strip() for column in table.columns]
    fitness = 0.1 * table["metrics/mAP50(B)"] + 0.9 * table["metrics/mAP50-95(B)"]
    row = table.loc[fitness.idxmax()]
    return {
        "results_csv": str(results_csv.resolve()),
        "best_epoch": int(row["epoch"]),
        "train_loss": float(sum(row[column] for column in ("train/box_loss", "train/cls_loss", "train/dfl_loss"))),
        "validation_loss": float(sum(row[column] for column in ("val/box_loss", "val/cls_loss", "val/dfl_loss"))),
        "logged_metrics": {
            "precision": float(row["metrics/precision(B)"]),
            "recall": float(row["metrics/recall(B)"]),
            "mAP50": float(row["metrics/mAP50(B)"]),
            "mAP50-95": float(row["metrics/mAP50-95(B)"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--val-original", required=True, type=Path)
    parser.add_argument("--val-clean", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--training-results", type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        if not args.force:
            raise FileExistsError(f"Refusing to overwrite non-empty evaluation dir: {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    model = YOLO(str(args.weights.resolve()))
    evaluations = {}
    for name, root_arg in (("original", args.val_original), ("clean", args.val_clean)):
        root = root_arg.resolve()
        yaml_path = out / f"val_{name}.yaml"
        write_yaml(root, yaml_path)
        metrics = model.val(
            data=str(yaml_path),
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=str(out / "ultralytics"),
            name=name,
            exist_ok=False,
            plots=True,
            verbose=False,
        )
        evaluations[name] = metric_dict(metrics)

    report = {
        "experiment_id": args.experiment_id,
        "weights": str(args.weights.resolve()),
        "weights_sha256": file_hash(args.weights.resolve()),
        "imgsz": args.imgsz,
        "validation": evaluations,
        "delta_clean_minus_original": {
            metric: evaluations["clean"][metric] - evaluations["original"][metric]
            for metric in ("precision", "recall", "mAP50", "mAP50-95")
        },
        "training": training_summary(args.training_results.resolve() if args.training_results else None),
    }
    (out / "validation_metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
