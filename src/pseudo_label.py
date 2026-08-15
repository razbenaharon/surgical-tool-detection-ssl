"""Generate pseudo-labels for unlabeled surgery videos (SSL step 2).

Samples frames from the given videos, runs the current model, and keeps the
confident predictions as YOLO-format pseudo-labels. Output is written as a
standard YOLO images/ + labels/ pair so it can be merged into a training set.

Heuristics for keeping a frame (all configurable):
  * per-box: keep only boxes with conf >= --box-conf
  * per-frame: keep the frame only if it has >=1 kept box AND its best box
    has conf >= --frame-conf (drops ambiguous frames)
  * temporal spacing: sample one frame every --stride frames (decorrelate)

Example:
    python src/pseudo_label.py \
        --weights runs/baseline/weights/best.pt \
        --videos /datashare/HW1/id_video_data/20_2_24_1.mp4 \
                 /datashare/HW1/id_video_data/4_2_24_B_2.mp4 \
        --out datasets/pseudo_id --stride 15 --box-conf 0.5 --frame-conf 0.7
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

CLASS_NAMES = {0: "Empty", 1: "Tweezers", 2: "Needle_driver"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="output dir (creates images/ labels/)")
    ap.add_argument("--stride", type=int, default=15, help="sample 1 of every N frames")
    ap.add_argument("--box-conf", type=float, default=0.5, help="min conf to keep a box")
    ap.add_argument("--frame-conf", type=float, default=0.7,
                    help="frame kept only if its best box >= this")
    ap.add_argument("--max-frames", type=int, default=100000,
                    help="cap total kept frames per run")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--jpg-quality", type=int, default=95)
    args = ap.parse_args()

    out = Path(args.out)
    img_out = out / "images"
    lbl_out = out / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)

    kept_frames = 0
    seen_frames = 0
    class_counts = Counter()
    conf_sum = 0.0
    n_boxes = 0

    for vid in args.videos:
        vid = Path(vid)
        stem = vid.stem
        cap = cv2.VideoCapture(str(vid))
        if not cap.isOpened():
            print(f"WARN could not open {vid}")
            continue
        idx = -1
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1
            if idx % args.stride != 0:
                continue
            seen_frames += 1
            if kept_frames >= args.max_frames:
                break
            res = model.predict(frame, imgsz=args.imgsz, conf=args.box_conf,
                                device=args.device, verbose=False)[0]
            if res.boxes is None or len(res.boxes) == 0:
                continue
            xywhn = res.boxes.xywhn.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            if confs.max() < args.frame_conf:
                continue
            lines = []
            for (xc, yc, w, h), c, k in zip(xywhn, confs, clss):
                lines.append(f"{k} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                class_counts[k] += 1
                conf_sum += float(c)
                n_boxes += 1
            name = f"{stem}_f{idx:06d}"
            cv2.imwrite(str(img_out / f"{name}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, args.jpg_quality])
            (lbl_out / f"{name}.txt").write_text("\n".join(lines) + "\n")
            kept_frames += 1
        cap.release()

    stats = {
        "videos": [str(v) for v in args.videos],
        "frames_sampled": seen_frames,
        "frames_kept": kept_frames,
        "keep_rate": round(kept_frames / max(seen_frames, 1), 3),
        "pseudo_boxes": n_boxes,
        "mean_box_conf": round(conf_sum / max(n_boxes, 1), 3),
        "class_counts": {CLASS_NAMES.get(k, k): v for k, v in sorted(class_counts.items())},
        "params": {"stride": args.stride, "box_conf": args.box_conf,
                   "frame_conf": args.frame_conf, "imgsz": args.imgsz},
    }
    (out / "pseudo_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"\nPseudo-labels written to {out}")


if __name__ == "__main__":
    main()
