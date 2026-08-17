"""Characterize real labeled boxes and derive conservative sanity bounds."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CLASS_NAMES = {0: "Empty", 1: "Tweezers", 2: "Needle_driver"}
METRICS = ("width", "height", "area", "aspect_ratio", "x_center", "y_center")


def load_boxes(labels_dir: Path) -> list[dict]:
    boxes = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        for row_number, raw in enumerate(label_path.read_text().splitlines(), 1):
            if not raw.strip():
                continue
            class_id, xc, yc, width, height = raw.split()
            xc, yc, width, height = map(float, (xc, yc, width, height))
            boxes.append(
                {
                    "file": label_path.name,
                    "row": row_number,
                    "class_id": int(class_id),
                    "x_center": xc,
                    "y_center": yc,
                    "width": width,
                    "height": height,
                    "area": width * height,
                    "aspect_ratio": width / height,
                }
            )
    return boxes


def summarize(values: np.ndarray) -> dict:
    quantiles = [0, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 1]
    names = ["min", "q01", "q025", "q05", "q10", "q25", "median", "q75", "q90", "q95", "q975", "q99", "max"]
    return {name: float(value) for name, value in zip(names, np.quantile(values, quantiles))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, type=Path, help="real train label directory")
    parser.add_argument("--out", type=Path, default=Path("reports/box_distribution"))
    args = parser.parse_args()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    boxes = load_boxes(args.labels.resolve())
    if not boxes:
        raise RuntimeError(f"No labels found under {args.labels}")

    arrays = {metric: np.array([box[metric] for box in boxes]) for metric in METRICS}
    summaries = {metric: summarize(values) for metric, values in arrays.items()}
    bounds = {
        "derivation": (
            "Real train only. Lower width/height = 0.75*q01; lower area = 0.5*q01; "
            "upper area = 1.4*q99. Spatial bounds are optional q01/q99 with 0.10 padding."
        ),
        "min_box_width": 0.75 * summaries["width"]["q01"],
        "min_box_height": 0.75 * summaries["height"]["q01"],
        "min_box_area": 0.5 * summaries["area"]["q01"],
        "max_box_area": min(1.0, 1.4 * summaries["area"]["q99"]),
        "optional_spatial_x_min": max(0.0, summaries["x_center"]["q01"] - 0.10),
        "optional_spatial_x_max": min(1.0, summaries["x_center"]["q99"] + 0.10),
        "optional_spatial_y_min": max(0.0, summaries["y_center"]["q01"] - 0.10),
        "optional_spatial_y_max": min(1.0, summaries["y_center"]["q99"] + 0.10),
    }
    class_counts = Counter(box["class_id"] for box in boxes)
    report = {
        "source": str(args.labels.resolve()),
        "box_count": len(boxes),
        "class_counts": {CLASS_NAMES.get(k, str(k)): v for k, v in sorted(class_counts.items())},
        "distributions": summaries,
        "conservative_bounds": bounds,
    }
    (out / "labeled_box_stats.json").write_text(json.dumps(report, indent=2))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for axis, metric in zip(axes.ravel(), METRICS):
        axis.hist(arrays[metric], bins=30, color="#2f6f9f", alpha=0.85)
        axis.axvline(summaries[metric]["q01"], color="#d95f02", linestyle="--", label="q01")
        axis.axvline(summaries[metric]["q99"], color="#1b9e77", linestyle="--", label="q99")
        axis.set(xlabel=metric, ylabel="real labeled boxes")
        axis.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.suptitle("Real labeled training-box distributions (n=%d)" % len(boxes))
    fig.tight_layout()
    fig.savefig(out / "labeled_box_distributions.png", dpi=180)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
