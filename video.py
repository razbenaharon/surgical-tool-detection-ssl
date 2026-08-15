"""video.py - run the surgical-tool detector on a video and write an
annotated output video (bounding boxes + class + confidence) using OpenCV.

Usage:
    python video.py --video path/to/surgery.mp4 --out annotated.mp4
    python video.py --video in.mp4 --weights weights/best.pt --conf 0.3 \
        --out out.mp4 --save-labels labels_dir

Class ids: 0=Empty (empty hand), 1=Tweezers, 2=Needle_driver.
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

CLASS_NAMES = {0: "Empty", 1: "Tweezers", 2: "Needle_driver"}
CLASS_COLORS = {0: (0, 200, 0), 1: (255, 128, 0), 2: (60, 60, 255)}  # BGR


def draw(frame, xyxy, cls, conf):
    x1, y1, x2, y2 = map(int, xyxy)
    color = CLASS_COLORS.get(cls, (255, 255, 255))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{CLASS_NAMES.get(cls, cls)} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y_top = max(y1, th + 6)
    cv2.rectangle(frame, (x1, y_top - th - 6), (x1 + tw + 2, y_top), color, -1)
    cv2.putText(frame, label, (x1 + 1, y_top - 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", default="weights/best.pt")
    ap.add_argument("--out", default="output.mp4")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--save-labels", default=None,
                    help="optional dir to dump per-frame YOLO-format labels")
    args = ap.parse_args()

    model = YOLO(args.weights)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    label_dir = None
    if args.save_labels:
        label_dir = Path(args.save_labels)
        label_dir.mkdir(parents=True, exist_ok=True)

    idx = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        res = model.predict(frame, imgsz=args.imgsz, conf=args.conf,
                            iou=args.iou, device=args.device, verbose=False)[0]
        lines = []
        if res.boxes is not None and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy()
            xywhn = res.boxes.xywhn.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            for box, (xc, yc, w, h), c, k in zip(xyxy, xywhn, confs, clss):
                draw(frame, box, k, float(c))
                lines.append(f"{xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {c:.4f} {k}")
        writer.write(frame)
        if label_dir is not None:
            (label_dir / f"frame_{idx:06d}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""))
        if total and idx % 100 == 0:
            print(f"\r{idx}/{total} frames", end="", flush=True)

    cap.release()
    writer.release()
    print(f"\nWrote annotated video: {args.out}")


if __name__ == "__main__":
    main()
