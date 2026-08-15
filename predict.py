"""predict.py - run the surgical-tool detector on a single image.

Outputs bounding boxes in YOLO format: (x_center, y_center, w, h, conf, class)
with normalized coordinates, and (optionally) saves an annotated image.

Usage:
    python predict.py --image path/to/img.jpg
    python predict.py --image img.jpg --weights weights/best.pt --conf 0.25 \
        --out-txt preds.txt --out-img annotated.jpg

The default weights path is `weights/best.pt` (see README for the download
link). Class ids: 0=Empty (empty hand), 1=Tweezers, 2=Needle_driver.
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

CLASS_NAMES = {0: "Empty", 1: "Tweezers", 2: "Needle_driver"}
CLASS_COLORS = {0: (0, 200, 0), 1: (255, 128, 0), 2: (60, 60, 255)}  # BGR


def draw(img, xyxy, cls, conf):
    x1, y1, x2, y2 = map(int, xyxy)
    color = CLASS_COLORS.get(cls, (255, 255, 255))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    label = f"{CLASS_NAMES.get(cls, cls)} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
    cv2.putText(img, label, (x1 + 1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", default="weights/best.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None, help="e.g. 0 for GPU, cpu for CPU")
    ap.add_argument("--out-txt", default=None, help="write YOLO-format predictions here")
    ap.add_argument("--out-img", default=None, help="write annotated image here")
    args = ap.parse_args()

    model = YOLO(args.weights)
    res = model.predict(args.image, imgsz=args.imgsz, conf=args.conf,
                        iou=args.iou, device=args.device, verbose=False)[0]

    img = cv2.imread(args.image)
    lines = []
    if res.boxes is not None and len(res.boxes):
        xywhn = res.boxes.xywhn.cpu().numpy()
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)
        for (xc, yc, w, h), box, c, k in zip(xywhn, xyxy, confs, clss):
            line = f"{xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {c:.4f} {k}"
            lines.append(line)
            print(line)
            if args.out_img:
                draw(img, box, k, float(c))
    else:
        print("# no detections")

    if args.out_txt:
        Path(args.out_txt).write_text("\n".join(lines) + ("\n" if lines else ""))
        print(f"# wrote {args.out_txt}")
    if args.out_img:
        cv2.imwrite(args.out_img, img)
        print(f"# wrote {args.out_img}")


if __name__ == "__main__":
    main()
