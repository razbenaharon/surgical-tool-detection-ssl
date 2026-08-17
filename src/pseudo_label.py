"""Generate reproducible, precision-oriented pseudo-label datasets from video.

The legacy policy remains expressible, but every selection knob and rejection
count is saved. Outputs are never overwritten unless ``--force`` is supplied.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


CLASS_NAMES = {0: "Empty", 1: "Tweezers", 2: "Needle_driver"}


def parse_class_values(items: list[str], value_type=float) -> dict[int, float]:
    parsed = {}
    for item in items:
        class_id, value = item.split(":", 1)
        parsed[int(class_id)] = value_type(value)
    return parsed


def xywh_iou(a: dict, b: dict) -> float:
    ax1, ay1 = a["xc"] - a["w"] / 2, a["yc"] - a["h"] / 2
    ax2, ay2 = a["xc"] + a["w"] / 2, a["yc"] + a["h"] / 2
    bx1, by1 = b["xc"] - b["w"] / 2, b["yc"] - b["h"] / 2
    bx2, by2 = b["xc"] + b["w"] / 2, b["yc"] + b["h"] / 2
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = a["w"] * a["h"] + b["w"] * b["h"] - intersection
    return intersection / union if union > 0 else 0.0


def distribution_summary(detections: list[dict]) -> dict:
    if not detections:
        return {"count": 0}
    metrics = {
        "confidence": np.array([d["conf"] for d in detections]),
        "width": np.array([d["w"] for d in detections]),
        "height": np.array([d["h"] for d in detections]),
        "area": np.array([d["w"] * d["h"] for d in detections]),
        "aspect_ratio": np.array([d["w"] / d["h"] for d in detections]),
        "x_center": np.array([d["xc"] for d in detections]),
        "y_center": np.array([d["yc"] for d in detections]),
    }
    result = {"count": len(detections)}
    for name, values in metrics.items():
        result[name] = {
            "min": float(values.min()),
            "q05": float(np.quantile(values, 0.05)),
            "median": float(np.median(values)),
            "mean": float(values.mean()),
            "q95": float(np.quantile(values, 0.95)),
            "max": float(values.max()),
        }
    return result


def class_summary(detections: list[dict]) -> dict:
    counts = Counter(d["cls"] for d in detections)
    total = sum(counts.values())
    result = {}
    for class_id, count in sorted(counts.items()):
        confidences = [d["conf"] for d in detections if d["cls"] == class_id]
        result[CLASS_NAMES.get(class_id, str(class_id))] = {
            "count": count,
            "percentage": round(100.0 * count / max(total, 1), 3),
            "mean_confidence": round(float(np.mean(confidences)), 6),
            "median_confidence": round(float(np.median(confidences)), 6),
        }
    return result


def plot_distributions(before: list[dict], after: list[dict], output: Path) -> None:
    metrics = [
        ("w", "normalized width"),
        ("h", "normalized height"),
        ("area", "normalized area"),
        ("aspect", "aspect ratio (w/h)"),
        ("xc", "x center"),
        ("yc", "y center"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for axis, (key, label) in zip(axes.ravel(), metrics):
        def values(items):
            if key == "area":
                return [d["w"] * d["h"] for d in items]
            if key == "aspect":
                return [d["w"] / d["h"] for d in items]
            return [d[key] for d in items]

        if before:
            axis.hist(values(before), bins=35, alpha=0.55, label="before sanity/temporal")
        if after:
            axis.hist(values(after), bins=35, alpha=0.55, label="accepted")
        axis.set_xlabel(label)
        axis.set_ylabel("boxes")
        axis.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_confidence(before: list[dict], after: list[dict], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 31)
    axis.hist([d["conf"] for d in before], bins=bins, alpha=0.55, label="before filters")
    axis.hist([d["conf"] for d in after], bins=bins, alpha=0.55, label="accepted")
    axis.set(xlabel="confidence", ylabel="boxes", title="Pseudo-label confidence")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def reject_reason(det: dict, args, class_conf: dict[int, float]) -> str | None:
    if det["conf"] < class_conf.get(det["cls"], args.box_conf_threshold):
        return "class_or_box_conf_threshold"
    if args.min_box_width is not None and det["w"] < args.min_box_width:
        return "min_box_width"
    if args.min_box_height is not None and det["h"] < args.min_box_height:
        return "min_box_height"
    area = det["w"] * det["h"]
    if args.min_box_area is not None and area < args.min_box_area:
        return "min_box_area"
    if args.max_box_area is not None and area > args.max_box_area:
        return "max_box_area"
    if args.spatial_x_min is not None and det["xc"] < args.spatial_x_min:
        return "spatial_x_min"
    if args.spatial_x_max is not None and det["xc"] > args.spatial_x_max:
        return "spatial_x_max"
    if args.spatial_y_min is not None and det["yc"] < args.spatial_y_min:
        return "spatial_y_min"
    if args.spatial_y_max is not None and det["yc"] > args.spatial_y_max:
        return "spatial_y_max"
    return None


def temporal_filter(records: list[dict], args, rejections: Counter) -> None:
    if args.temporal_min_hits <= 1:
        return
    original_detections = [list(record["detections"]) for record in records]
    filtered_by_record = []
    for position, record in enumerate(records):
        kept = []
        for det in original_detections[position]:
            matched_confidences = [det["conf"]]
            for neighbor_pos in range(
                max(0, position - args.temporal_radius),
                min(len(records), position + args.temporal_radius + 1),
            ):
                if neighbor_pos == position:
                    continue
                candidates = [
                    other
                    for other in original_detections[neighbor_pos]
                    if other["cls"] == det["cls"] and xywh_iou(det, other) >= args.temporal_iou
                ]
                if candidates:
                    matched_confidences.append(max(d["conf"] for d in candidates))
            mean_confidence = float(np.mean(matched_confidences))
            if len(matched_confidences) < args.temporal_min_hits:
                rejections["temporal_min_hits"] += 1
            elif mean_confidence < args.temporal_mean_conf:
                rejections["temporal_mean_conf"] += 1
            else:
                det["temporal_hits"] = len(matched_confidences)
                det["temporal_mean_conf"] = mean_confidence
                kept.append(det)
        filtered_by_record.append(kept)
    for record, kept in zip(records, filtered_by_record):
        record["detections"] = kept


def apply_class_caps(records: list[dict], caps: dict[int, int], rejections: Counter) -> None:
    for class_id, cap in caps.items():
        ranked = sorted(
            (
                (det["conf"], record_index, det_index)
                for record_index, record in enumerate(records)
                for det_index, det in enumerate(record["detections"])
                if det["cls"] == class_id
            ),
            reverse=True,
        )
        keep = {(record_index, det_index) for _, record_index, det_index in ranked[:cap]}
        for record_index, record in enumerate(records):
            retained = []
            for det_index, det in enumerate(record["detections"]):
                if det["cls"] != class_id or (record_index, det_index) in keep:
                    retained.append(det)
                else:
                    rejections[f"class_cap_{class_id}"] += 1
            record["detections"] = retained


def write_selected_frames(records: list[dict], out: Path, jpg_quality: int) -> None:
    by_video = defaultdict(list)
    for record in records:
        by_video[record["video"]].append(record)
    for video, video_records in by_video.items():
        wanted = {record["frame_index"]: record for record in video_records}
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise RuntimeError(f"Could not reopen video for frame export: {video}")
        index = -1
        remaining = set(wanted)
        while remaining:
            ok, frame = cap.read()
            if not ok:
                break
            index += 1
            if index not in remaining:
                continue
            record = wanted[index]
            cv2.imwrite(
                str(out / "images" / f"{record['name']}.jpg"),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, jpg_quality],
            )
            lines = [
                f"{d['cls']} {d['xc']:.6f} {d['yc']:.6f} {d['w']:.6f} {d['h']:.6f}"
                for d in record["detections"]
            ]
            (out / "labels" / f"{record['name']}.txt").write_text("\n".join(lines) + "\n")
            remaining.remove(index)
        cap.release()
        if remaining:
            raise RuntimeError(f"Failed to export frames {sorted(remaining)} from {video}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--box-conf-threshold", "--box-conf", type=float, default=0.5)
    parser.add_argument("--frame-conf-threshold", "--frame-conf", type=float, default=0.7)
    parser.add_argument("--class-conf", action="append", default=[], metavar="CLASS:THRESHOLD")
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--max-frames", type=int, default=100000)
    parser.add_argument("--max-detections-per-frame", type=int, default=300)
    parser.add_argument("--inference-max-detections", type=int, default=300)
    parser.add_argument("--min-box-width", type=float)
    parser.add_argument("--min-box-height", type=float)
    parser.add_argument("--min-box-area", type=float)
    parser.add_argument("--max-box-area", type=float)
    parser.add_argument("--spatial-x-min", type=float)
    parser.add_argument("--spatial-x-max", type=float)
    parser.add_argument("--spatial-y-min", type=float)
    parser.add_argument("--spatial-y-max", type=float)
    parser.add_argument("--temporal-radius", type=int, default=1)
    parser.add_argument("--temporal-min-hits", type=int, default=1)
    parser.add_argument("--temporal-iou", type=float, default=0.3)
    parser.add_argument("--temporal-mean-conf", type=float, default=0.0)
    parser.add_argument("--class-cap", action="append", default=[], metavar="CLASS:CAP")
    parser.add_argument("--nms-iou-threshold", type=float, default=0.6)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--jpg-quality", type=int, default=95)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        if not args.force:
            raise FileExistsError(f"Refusing to overwrite non-empty output: {out}")
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir(parents=True)

    class_conf = parse_class_values(args.class_conf, float)
    class_caps = parse_class_values(args.class_cap, int)
    model = YOLO(args.weights)
    records, before_filters = [], []
    rejections = Counter()
    sampled_frames = 0

    for video_order, video_arg in enumerate(args.videos):
        video = str(Path(video_arg).resolve())
        stem = Path(video).stem
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video}")
        frame_index = -1
        video_records = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            if frame_index % args.stride:
                continue
            sampled_frames += 1
            prediction = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=min(args.box_conf_threshold, min(class_conf.values(), default=1.0)),
                iou=args.nms_iou_threshold,
                max_det=max(args.inference_max_detections, 1),
                device=args.device,
                verbose=False,
            )[0]
            raw = []
            if prediction.boxes is not None and len(prediction.boxes):
                for xywh, conf, class_id in zip(
                    prediction.boxes.xywhn.cpu().numpy(),
                    prediction.boxes.conf.cpu().numpy(),
                    prediction.boxes.cls.cpu().numpy().astype(int),
                ):
                    xc, yc, width, height = map(float, xywh)
                    raw.append({
                        "xc": xc, "yc": yc, "w": width, "h": height,
                        "conf": float(conf), "cls": int(class_id),
                    })
            before_filters.extend(raw)
            filtered = []
            for det in sorted(raw, key=lambda item: item["conf"], reverse=True):
                reason = reject_reason(det, args, class_conf)
                if reason:
                    rejections[reason] += 1
                else:
                    filtered.append(det)
            if len(filtered) > args.max_detections_per_frame:
                rejections["max_detections_per_frame"] += len(filtered) - args.max_detections_per_frame
                filtered = filtered[: args.max_detections_per_frame]
            video_records.append({
                "video": video, "video_order": video_order, "frame_index": frame_index,
                "name": f"{stem}_f{frame_index:06d}", "detections": filtered,
            })
        cap.release()
        temporal_filter(video_records, args, rejections)
        records.extend(video_records)

    apply_class_caps(records, class_caps, rejections)
    accepted_records = []
    rejected_frames = Counter()
    for record in records:
        if not record["detections"]:
            rejected_frames["no_boxes_after_filters"] += 1
            continue
        strongest = max(d["conf"] for d in record["detections"])
        if strongest < args.frame_conf_threshold:
            rejected_frames["frame_conf_threshold"] += 1
            rejections["frame_conf_threshold"] += len(record["detections"])
            continue
        record["strongest_confidence"] = strongest
        accepted_records.append(record)

    if len(accepted_records) > args.max_frames:
        removed = len(accepted_records) - args.max_frames
        accepted_records = sorted(
            accepted_records, key=lambda record: record["strongest_confidence"], reverse=True
        )[: args.max_frames]
        rejected_frames["max_frames"] += removed
    accepted_records.sort(key=lambda record: (record["video_order"], record["frame_index"]))
    accepted_detections = [d for record in accepted_records for d in record["detections"]]
    write_selected_frames(accepted_records, out, args.jpg_quality)

    confidence_histogram, confidence_edges = np.histogram(
        [d["conf"] for d in accepted_detections], bins=np.linspace(0, 1, 21)
    )
    config = vars(args).copy()
    config["out"] = str(out)
    stats = {
        "policy_id": args.policy_id,
        "command": " ".join(sys.argv),
        "weights": str(Path(args.weights).resolve()),
        "videos": [str(Path(video).resolve()) for video in args.videos],
        "frames_sampled": sampled_frames,
        "frames_accepted": len(accepted_records),
        "frames_rejected": sampled_frames - len(accepted_records),
        "frame_rejection_reasons": dict(rejected_frames),
        "keep_rate": round(len(accepted_records) / max(sampled_frames, 1), 6),
        "boxes_before_filters": len(before_filters),
        "boxes_accepted": len(accepted_detections),
        "box_rejection_reasons": dict(rejections),
        "classes": class_summary(accepted_detections),
        "distribution_before_filters": distribution_summary(before_filters),
        "distribution_accepted": distribution_summary(accepted_detections),
        "confidence_histogram": {"bin_edges": confidence_edges.tolist(), "counts": confidence_histogram.tolist()},
        "config": config,
    }
    manifest = [{
        "name": record["name"], "source_video": record["video"],
        "source_frame": record["frame_index"],
        "strongest_confidence": record["strongest_confidence"],
        "detections": record["detections"],
    } for record in accepted_records]
    (out / "pseudo_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    plot_distributions(before_filters, accepted_detections, out / "box_distributions.png")
    plot_confidence(before_filters, accepted_detections, out / "confidence_histogram.png")
    print(json.dumps(stats, indent=2, default=str))
    print(f"Pseudo-labels written to {out}")


if __name__ == "__main__":
    main()
