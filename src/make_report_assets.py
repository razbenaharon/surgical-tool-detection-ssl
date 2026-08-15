"""Build the figures + metrics.json the report needs, using the trained runs.

- Extracts best-epoch val metrics from each run's results.csv (by YOLO fitness
  = 0.1*mAP50 + 0.9*mAP50-95, matching how best.pt is chosen).
- Copies each run's results.png training curve into reports/curves/.
- Runs the FINAL model on evenly-sampled OOD frames -> reports/ood_pred_grid.jpg
- Baseline vs FINAL on the same frames -> reports/ood_compare.jpg
- Writes reports/metrics.json

Usage:
    python src/make_report_assets.py \
        --runs runs/detect/runs \
        --baseline runs/detect/runs/baseline/weights/best.pt \
        --final runs/detect/runs/ssl_ood/weights/best.pt \
        --ood /datashare/HW1/ood_video_data/surg_1.mp4 \
              /datashare/HW1/ood_video_data/4_2_24_A_1.mp4
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

NAMES = {0: "Empty", 1: "Tweezers", 2: "Needle_driver"}
COLORS = {0: (0, 200, 0), 1: (255, 128, 0), 2: (60, 60, 255)}  # BGR


def best_row(csv_path):
    if not Path(csv_path).exists():
        return None
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return None
    def fit(r):
        return 0.1 * float(r["metrics/mAP50(B)"]) + 0.9 * float(r["metrics/mAP50-95(B)"])
    r = max(rows, key=fit)
    return {
        "epoch": int(float(r["epoch"])),
        "precision": float(r["metrics/precision(B)"]),
        "recall": float(r["metrics/recall(B)"]),
        "mAP50": float(r["metrics/mAP50(B)"]),
        "mAP50_95": float(r["metrics/mAP50-95(B)"]),
        "epochs_total": len(rows),
    }


def draw(img, box, cls, conf):
    x1, y1, x2, y2 = map(int, box)
    c = COLORS.get(cls, (255, 255, 255))
    cv2.rectangle(img, (x1, y1), (x2, y2), c, 4)
    txt = f"{NAMES.get(cls, cls)} {conf:.2f}"
    cv2.putText(img, txt, (x1, max(34, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, c, 3)


def sample_frames(videos, n_per_video):
    frames = []
    for v in videos:
        cap = cv2.VideoCapture(v)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = np.linspace(int(total * 0.1), int(total * 0.9), n_per_video).astype(int)
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ret, fr = cap.read()
            if ret:
                frames.append((Path(v).stem, int(i), fr))
        cap.release()
    return frames


def predict_draw(model, frame, conf):
    img = frame.copy()
    r = model.predict(img, imgsz=640, conf=conf, iou=0.6, verbose=False)[0]
    if r.boxes is not None and len(r.boxes):
        for box, c, k in zip(r.boxes.xyxy.cpu().numpy(),
                             r.boxes.conf.cpu().numpy(),
                             r.boxes.cls.cpu().numpy().astype(int)):
            draw(img, box, k, float(c))
    return img


def grid(imgs, cols, cell=(640, 360)):
    tiles = [cv2.resize(im, cell) for im in imgs]
    rows = []
    for i in range(0, len(tiles), cols):
        row = tiles[i:i + cols]
        while len(row) < cols:
            row.append(np.zeros((cell[1], cell[0], 3), np.uint8))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/detect/runs")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--final", required=True)
    ap.add_argument("--ood", nargs="+", required=True)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "curves").mkdir(parents=True, exist_ok=True)

    # metrics
    metrics = {}
    name_map = {"baseline": "baseline", "ssl_id": "ssl_id", "ssl_ood": "ssl_ood"}
    train_imgs = {"baseline": 61, "ssl_id": 763, "ssl_ood": 1430}
    for key in name_map:
        row = best_row(Path(args.runs) / key / "results.csv")
        if row:
            row["train_imgs"] = train_imgs.get(key, "-")
            metrics[key] = row
        cur = Path(args.runs) / key / "results.png"
        if cur.exists():
            shutil.copy(cur, out / "curves" / f"{key}_results.png")

    # pseudo stats
    for tag, pth in [("id", "datasets/pseudo_id/pseudo_stats.json"),
                     ("ood", "datasets/pseudo_ood/pseudo_stats.json")]:
        if Path(pth).exists():
            st = json.loads(Path(pth).read_text())
            metrics.setdefault("pseudo", {})
            metrics["pseudo"].update({
                "stride": st["params"]["stride"],
                "frame_conf": st["params"]["frame_conf"],
                "box_conf": st["params"]["box_conf"],
                f"{tag}_kept": st["frames_kept"],
                f"{tag}_sampled": st["frames_sampled"],
                f"{tag}_keep_rate": st["keep_rate"],
                f"{tag}_boxes": st["pseudo_boxes"],
                f"{tag}_mean_conf": st["mean_box_conf"],
            })

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("metrics:", json.dumps(metrics, indent=2))

    # figures
    final = YOLO(args.final)
    base = YOLO(args.baseline)

    frames = sample_frames(args.ood, n_per_video=3)
    # prediction grid (final model)
    pred_imgs = [predict_draw(final, fr, args.conf) for (_, _, fr) in frames]
    for (stem, idx, _), im in zip(frames, pred_imgs):
        cv2.putText(im, f"{stem} #{idx}", (20, im.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)
    cv2.imwrite(str(out / "ood_pred_grid.jpg"), grid(pred_imgs, cols=3))
    print("wrote ood_pred_grid.jpg")

    # comparison: baseline (top) vs final (bottom) on 3 frames
    cmp_frames = frames[:3] if len(frames) >= 3 else frames
    base_imgs, fin_imgs = [], []
    for (stem, idx, fr) in cmp_frames:
        bi = predict_draw(base, fr, args.conf); cv2.putText(bi, "baseline", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 255), 4)
        fi = predict_draw(final, fr, args.conf); cv2.putText(fi, "SSL final", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 255), 4)
        base_imgs.append(bi); fin_imgs.append(fi)
    top = grid(base_imgs, cols=3)
    bot = grid(fin_imgs, cols=3)
    cv2.imwrite(str(out / "ood_compare.jpg"), np.vstack([top, bot]))
    print("wrote ood_compare.jpg")


if __name__ == "__main__":
    main()
