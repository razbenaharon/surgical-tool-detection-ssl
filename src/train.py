"""Train / fine-tune a YOLO detector for surgical tools.

Used for both the supervised baseline and the semi-supervised refinement
rounds - only the --data yaml and --weights change.

Examples
--------
Baseline (from COCO-pretrained yolo11s):
    python src/train.py --data data/labeled.yaml --name baseline \
        --model yolo11s.pt --epochs 150 --imgsz 640

SSL refine (start from the previous best.pt, train on labeled+pseudo):
    python src/train.py --data data/ssl_id.yaml --name ssl_id \
        --model runs/baseline/weights/best.pt --epochs 120
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset yaml")
    ap.add_argument("--model", default="yolo11s.pt",
                    help="pretrained weights or checkpoint to start from")
    ap.add_argument("--name", required=True, help="run name under project dir")
    ap.add_argument("--project", default="runs")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--device", default="0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--lrf", type=float, default=0.01)
    ap.add_argument("--optimizer", default="AdamW")
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--momentum", type=float, default=0.937)
    ap.add_argument("--warmup-epochs", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save-period", type=int, default=-1)
    ap.add_argument("--exist-ok", action="store_true",
                    help="explicitly allow an existing Ultralytics run directory")
    ap.add_argument("--cos_lr", action="store_true", default=True)
    # augmentation knobs. NOTE: on this tiny dataset, aggressive augmentation
    # (mixup / copy_paste / shear / rotation) + long mosaic made training
    # diverge (cls_loss exploded, model collapsed to conf=1.0 everywhere).
    # Keeping a moderate, stable set: mosaic (closed for the last epochs),
    # colour jitter, left-right flip, mild translate/scale. No vertical flip
    # (surgical orientation is real), no mixup/copy_paste/shear/rotation.
    ap.add_argument("--mosaic", type=float, default=1.0)
    ap.add_argument("--close_mosaic", type=int, default=15)
    ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--copy_paste", type=float, default=0.0)
    ap.add_argument("--degrees", type=float, default=0.0)
    ap.add_argument("--translate", type=float, default=0.1)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--shear", type=float, default=0.0)
    ap.add_argument("--fliplr", type=float, default=0.5)
    ap.add_argument("--flipud", type=float, default=0.0)
    ap.add_argument("--hsv_h", type=float, default=0.015)
    ap.add_argument("--hsv_s", type=float, default=0.7)
    ap.add_argument("--hsv_v", type=float, default=0.4)
    ap.add_argument("--freeze", type=int, default=0,
                    help="freeze first N layers (0 = train all)")
    args = ap.parse_args()

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        pretrained=True,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        warmup_epochs=args.warmup_epochs,
        workers=args.workers,
        save_period=args.save_period,
        cos_lr=args.cos_lr,
        freeze=args.freeze if args.freeze > 0 else None,
        # augmentation
        mosaic=args.mosaic,
        close_mosaic=args.close_mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        fliplr=args.fliplr,
        flipud=args.flipud,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        plots=True,
        val=True,
    )
    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path(args.project) / args.name
    manifest = {
        "experiment_id": args.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "args": vars(args),
        "source_weights": str(Path(args.model).resolve()) if Path(args.model).exists() else args.model,
        "dataset_config": str(Path(args.data).resolve()),
        "git_commit": None,
    }
    try:
        manifest["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    (save_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nDone. Weights + curves in: {save_dir}")
    print(f"best.pt: {save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
