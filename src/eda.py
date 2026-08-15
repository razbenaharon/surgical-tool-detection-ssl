"""Exploratory Data Analysis for the surgical-tool detection dataset.

Produces, under --out:
  * dataset_summary.json  - counts, resolutions, per-class box stats
  * class_distribution.png
  * boxes_per_image.png
  * box_size_distribution.png
  * bbox_center_heatmap.png
  * samples_grid.png       - a few labeled images with their boxes drawn
  * video_metadata.json    - fps / frames / resolution for the ID & OOD videos

Run on the VM inside the venv, e.g.:
  python src/eda.py \
      --data /home/student/hw1/labeled_image_data \
      --videos /datashare/HW1 \
      --out reports/eda
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLASS_NAMES = {0: "Empty", 1: "Tweezers", 2: "Needle_driver"}
CLASS_COLORS = {0: (0, 200, 0), 1: (0, 128, 255), 2: (255, 60, 60)}  # RGB
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def find_image(img_dir: Path, stem: str):
    for ext in IMG_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def read_labels(label_path: Path):
    """Return list of (cls, xc, yc, w, h) in normalized coords."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:5])
        boxes.append((cls, xc, yc, w, h))
    return boxes


def collect_split(data_root: Path, split: str):
    img_dir = data_root / "images" / split
    lbl_dir = data_root / "labels" / split
    records = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        h, w = _img_hw(img_path)
        records.append(
            {
                "image": img_path,
                "label": lbl_path,
                "width": w,
                "height": h,
                "boxes": read_labels(lbl_path),
            }
        )
    return records


def _img_hw(img_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        return (None, None)
    return img.shape[0], img.shape[1]


def analyze(records):
    class_counts = Counter()
    boxes_per_image = []
    resolutions = Counter()
    widths, heights, areas, aspects = [], [], [], []
    centers = []
    images_with_no_box = 0
    for r in records:
        boxes_per_image.append(len(r["boxes"]))
        if len(r["boxes"]) == 0:
            images_with_no_box += 1
        resolutions[(r["width"], r["height"])] += 1
        for cls, xc, yc, bw, bh in r["boxes"]:
            class_counts[cls] += 1
            widths.append(bw)
            heights.append(bh)
            areas.append(bw * bh)
            aspects.append(bw / bh if bh > 0 else 0)
            centers.append((xc, yc))
    return {
        "n_images": len(records),
        "n_boxes": int(sum(class_counts.values())),
        "images_with_no_box": images_with_no_box,
        "class_counts": {CLASS_NAMES.get(k, k): v for k, v in sorted(class_counts.items())},
        "boxes_per_image_mean": float(np.mean(boxes_per_image)) if boxes_per_image else 0,
        "resolutions": {f"{w}x{h}": c for (w, h), c in resolutions.items()},
        "_boxes_per_image": boxes_per_image,
        "_widths": widths,
        "_heights": heights,
        "_areas": areas,
        "_aspects": aspects,
        "_centers": centers,
        "_class_counts_raw": dict(class_counts),
    }


def plot_class_distribution(stats_by_split, out):
    classes = list(CLASS_NAMES.values())
    splits = list(stats_by_split.keys())
    x = np.arange(len(classes))
    width = 0.8 / max(len(splits), 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, sp in enumerate(splits):
        raw = stats_by_split[sp]["_class_counts_raw"]
        vals = [raw.get(c, 0) for c in CLASS_NAMES]
        ax.bar(x + i * width, vals, width, label=sp)
    ax.set_xticks(x + width * (len(splits) - 1) / 2)
    ax.set_xticklabels(classes)
    ax.set_ylabel("# bounding boxes")
    ax.set_title("Class distribution (bounding boxes) by split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "class_distribution.png", dpi=140)
    plt.close(fig)


def plot_boxes_per_image(stats_by_split, out):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for sp, st in stats_by_split.items():
        vals = st["_boxes_per_image"]
        if not vals:
            continue
        bins = np.arange(0, max(vals) + 2) - 0.5
        ax.hist(vals, bins=bins, alpha=0.6, label=f"{sp} (n={len(vals)})")
    ax.set_xlabel("# boxes in image")
    ax.set_ylabel("# images")
    ax.set_title("Boxes per image")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "boxes_per_image.png", dpi=140)
    plt.close(fig)


def plot_box_sizes(stats_by_split, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for sp, st in stats_by_split.items():
        if st["_areas"]:
            axes[0].hist(np.sqrt(st["_areas"]), bins=20, alpha=0.6, label=sp)
            axes[1].scatter(st["_widths"], st["_heights"], s=12, alpha=0.5, label=sp)
    axes[0].set_xlabel("sqrt(box area) [frac of image]")
    axes[0].set_ylabel("# boxes")
    axes[0].set_title("Relative box size")
    axes[0].legend()
    axes[1].set_xlabel("box width [frac]")
    axes[1].set_ylabel("box height [frac]")
    axes[1].set_title("Box width vs height")
    axes[1].plot([0, 1], [0, 1], "k--", lw=0.7)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / "box_size_distribution.png", dpi=140)
    plt.close(fig)


def plot_center_heatmap(stats_by_split, out):
    centers = []
    for st in stats_by_split.values():
        centers.extend(st["_centers"])
    if not centers:
        return
    xs, ys = zip(*centers)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    hb = ax.hist2d(xs, ys, bins=20, range=[[0, 1], [0, 1]], cmap="magma")
    ax.invert_yaxis()  # image coords: y grows downward
    ax.set_xlabel("x_center")
    ax.set_ylabel("y_center")
    ax.set_title("Box-center spatial density (all boxes)")
    fig.colorbar(hb[3], ax=ax, label="# boxes")
    fig.tight_layout()
    fig.savefig(out / "bbox_center_heatmap.png", dpi=140)
    plt.close(fig)


def draw_boxes_grid(records, out, n=9, seed=0):
    random.seed(seed)
    sample = random.sample(records, min(n, len(records)))
    cols = 3
    rows = int(np.ceil(len(sample) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.2))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, r in zip(axes, sample):
        img = cv2.imread(str(r["image"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        for cls, xc, yc, bw, bh in r["boxes"]:
            x1 = int((xc - bw / 2) * W)
            y1 = int((yc - bh / 2) * H)
            x2 = int((xc + bw / 2) * W)
            y2 = int((yc + bh / 2) * H)
            color = CLASS_COLORS.get(cls, (255, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.putText(img, CLASS_NAMES.get(cls, str(cls)), (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        ax.imshow(img)
        ax.set_title(r["image"].name, fontsize=7)
    fig.suptitle("Sample labeled images with ground-truth boxes")
    fig.tight_layout()
    fig.savefig(out / "samples_grid.png", dpi=130)
    plt.close(fig)


def video_metadata(videos_root: Path):
    meta = {}
    for sub in ["id_video_data", "ood_video_data"]:
        d = videos_root / sub
        if not d.exists():
            continue
        for vp in sorted(d.glob("*.mp4")):
            cap = cv2.VideoCapture(str(vp))
            meta[f"{sub}/{vp.name}"] = {
                "fps": round(cap.get(cv2.CAP_PROP_FPS), 3),
                "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "duration_s": round(cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 1), 1),
            }
            cap.release()
    return meta


def strip_private(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/student/hw1/labeled_image_data")
    ap.add_argument("--videos", default="/datashare/HW1")
    ap.add_argument("--out", default="reports/eda")
    args = ap.parse_args()

    data_root = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    stats_by_split = {}
    all_records = []
    for split in ["train", "val"]:
        if not (data_root / "images" / split).exists():
            continue
        recs = collect_split(data_root, split)
        all_records.extend(recs)
        stats_by_split[split] = analyze(recs)

    plot_class_distribution(stats_by_split, out)
    plot_boxes_per_image(stats_by_split, out)
    plot_box_sizes(stats_by_split, out)
    plot_center_heatmap(stats_by_split, out)
    draw_boxes_grid(all_records, out)

    vmeta = video_metadata(Path(args.videos))
    (out / "video_metadata.json").write_text(json.dumps(vmeta, indent=2))

    summary = {sp: strip_private(st) for sp, st in stats_by_split.items()}
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print("\nVideo metadata:")
    print(json.dumps(vmeta, indent=2))
    print(f"\nFigures written to {out}")


if __name__ == "__main__":
    main()
