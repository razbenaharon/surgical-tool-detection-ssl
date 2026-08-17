"""Build the two required report figures directly from each run's results.csv:
  * loss_curves.png  - train vs. validation loss per epoch (3 rounds)
  * map_curves.png   - validation mAP@50 / mAP@50-95 per epoch (3 rounds)

Ultralytics only evaluates mAP on the validation split each epoch (there is
no equivalent training-set mAP logged), so the mAP figure shows the
validation curves only; this is noted in the report text rather than
fabricating a training mAP.

Usage:
    python src/make_curve_figures.py --runs <path to runs/detect/runs> --out reports/curves
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = [("baseline", "Baseline (labeled only)"),
        ("ssl_id", "+ ID pseudo-labels"),
        ("ssl_ood", "+ OOD pseudo-labels (final)")]


def load(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    epochs = [int(float(r["epoch"])) for r in rows]
    d = {
        "epoch": epochs,
        "train_loss": [float(r["train/box_loss"]) + float(r["train/cls_loss"]) + float(r["train/dfl_loss"]) for r in rows],
        "val_loss": [],
        "mAP50": [float(r["metrics/mAP50(B)"]) for r in rows],
        "mAP50_95": [float(r["metrics/mAP50-95(B)"]) for r in rows],
    }
    for r in rows:
        try:
            d["val_loss"].append(float(r["val/box_loss"]) + float(r["val/cls_loss"]) + float(r["val/dfl_loss"]))
        except ValueError:
            d["val_loss"].append(float("nan"))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/detect/runs")
    ap.add_argument("--out", default="reports/curves")
    args = ap.parse_args()
    runs_root = Path(args.runs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = {}
    for key, _ in RUNS:
        p = runs_root / key / "results.csv"
        if p.exists():
            data[key] = load(p)

    # ---- Fig: train + valid loss ----
    fig, axes = plt.subplots(1, len(data), figsize=(5.0 * len(data), 3.6), sharey=False)
    if len(data) == 1:
        axes = [axes]
    for ax, (key, label) in zip(axes, RUNS):
        if key not in data:
            continue
        d = data[key]
        ax.plot(d["epoch"], d["train_loss"], label="train loss", color="#1f77b4", lw=1.4)
        ax.plot(d["epoch"], d["val_loss"], label="val loss", color="#d62728", lw=1.4)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("epoch")
        ax.set_ylim(0, 8)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("loss (box + cls + dfl)")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Training and validation loss vs. epoch", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out / "loss_curves.png", dpi=150)
    plt.close(fig)

    # ---- Fig: validation mAP ----
    fig, axes = plt.subplots(1, len(data), figsize=(5.0 * len(data), 3.6), sharey=True)
    if len(data) == 1:
        axes = [axes]
    for ax, (key, label) in zip(axes, RUNS):
        if key not in data:
            continue
        d = data[key]
        ax.plot(d["epoch"], d["mAP50"], label="val mAP@50", color="#2ca02c", lw=1.4)
        ax.plot(d["epoch"], d["mAP50_95"], label="val mAP@50-95", color="#9467bd", lw=1.4)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("epoch")
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("mAP (validation set)")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Validation mAP@50 and mAP@50-95 vs. epoch", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out / "map_curves.png", dpi=150)
    plt.close(fig)

    print(f"wrote {out / 'loss_curves.png'}")
    print(f"wrote {out / 'map_curves.png'}")


if __name__ == "__main__":
    main()
