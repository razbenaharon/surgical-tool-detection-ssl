"""Create identical OOD preview clips and automated stability diagnostics."""
from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


CLASS_NAMES = {0: "Empty", 1: "Tweezers", 2: "Needle_driver"}
CLASS_COLORS = {0: (0, 200, 0), 1: (255, 128, 0), 2: (60, 60, 255)}


def parse_segments(values: list[str], duration: float) -> list[tuple[float, float]]:
    if not values:
        return [(0.0, duration)]
    segments = []
    for value in values:
        start, end = map(float, value.split(":", 1))
        if not (0 <= start < end <= duration + 1e-6):
            raise ValueError(f"Invalid segment {value}; video duration is {duration:.3f}s")
        segments.append((start, min(end, duration)))
    return segments


def iou(a: dict, b: dict) -> float:
    ax1, ay1, ax2, ay2 = a["xyxy"]
    bx1, by1, bx2, by2 = b["xyxy"]
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def draw(frame, detection: dict) -> None:
    x1, y1, x2, y2 = map(int, detection["xyxy"])
    class_id, confidence = detection["cls"], detection["conf"]
    color = CLASS_COLORS.get(class_id, (255, 255, 255))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 7)
    label = f"{CLASS_NAMES.get(class_id, class_id)} {confidence:.2f}"
    font_scale, thickness = 1.8, 5
    (width, height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    top = max(height + 16, y1)
    cv2.rectangle(frame, (x1, top - height - 16), (x1 + width + 10, top), color, -1)
    cv2.putText(
        frame, label, (x1 + 5, top - 9), cv2.FONT_HERSHEY_SIMPLEX,
        font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
    )


class SimpleTracker:
    def __init__(self, match_iou: float):
        self.match_iou = match_iou
        self.active = []
        self.finished = []
        self.next_id = 0
        self.last_frame = None
        self.class_switches = 0
        self.transitions = 0

    def reset_gap(self):
        self.finished.extend(self.active)
        self.active = []

    def update(self, frame_index: int, detections: list[dict]):
        if self.last_frame is not None and frame_index != self.last_frame + 1:
            self.reset_gap()
        self.last_frame = frame_index
        pairs = sorted(
            (
                (iou(track["last"], detection), track_index, det_index)
                for track_index, track in enumerate(self.active)
                for det_index, detection in enumerate(detections)
            ),
            reverse=True,
        )
        used_tracks, used_detections = set(), set()
        for overlap, track_index, det_index in pairs:
            if overlap < self.match_iou:
                break
            if track_index in used_tracks or det_index in used_detections:
                continue
            track, detection = self.active[track_index], detections[det_index]
            self.transitions += 1
            if track["last"]["cls"] != detection["cls"]:
                self.class_switches += 1
            track["length"] += 1
            track["last"] = detection
            track["classes"].append(detection["cls"])
            used_tracks.add(track_index)
            used_detections.add(det_index)
        self.finished.extend(
            track for index, track in enumerate(self.active) if index not in used_tracks
        )
        self.active = [track for index, track in enumerate(self.active) if index in used_tracks]
        for det_index, detection in enumerate(detections):
            if det_index in used_detections:
                continue
            self.active.append(
                {"id": self.next_id, "length": 1, "last": detection, "classes": [detection["cls"]]}
            )
            self.next_id += 1

    def finalize(self):
        self.reset_gap()
        return self.finished


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--out-video", required=True, type=Path)
    parser.add_argument("--out-stats", required=True, type=Path)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--segments", nargs="*", default=[] , metavar="START:END")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-detections", type=int, default=10)
    parser.add_argument("--output-width", type=int, default=1280)
    parser.add_argument("--device", default="0")
    parser.add_argument("--track-iou", type=float, default=0.3)
    parser.add_argument("--tiny-width", type=float, default=0.05865)
    parser.add_argument("--tiny-height", type=float, default=0.13356)
    parser.add_argument("--tiny-area", type=float, default=0.00934)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for output in (args.out_video.resolve(), args.out_stats.resolve()):
        if output.exists() and not args.force:
            raise FileExistsError(f"Refusing to overwrite: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)

    video_path = args.video.resolve()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps
    segments = parse_segments(args.segments, duration)
    output_width = min(source_width, args.output_width) if args.output_width > 0 else source_width
    output_height = round(source_height * output_width / source_width)
    if output_height % 2:
        output_height += 1
    writer = cv2.VideoWriter(
        str(args.out_video.resolve()), cv2.VideoWriter_fourcc(*"mp4v"), fps, (output_width, output_height)
    )
    model = YOLO(str(args.weights.resolve()))
    tracker = SimpleTracker(args.track_iou)
    frame_counts, all_detections = [], []
    processed_frames = 0

    for segment_index, (start, end) in enumerate(segments):
        start_frame, end_frame = round(start * fps), min(total_frames, round(end * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        tracker.reset_gap()
        for frame_index in range(start_frame, end_frame):
            ok, frame = cap.read()
            if not ok:
                break
            result = model.predict(
                frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                max_det=args.max_detections, device=args.device, verbose=False,
            )[0]
            detections = []
            if result.boxes is not None and len(result.boxes):
                for xyxy, xywhn, confidence, class_id in zip(
                    result.boxes.xyxy.cpu().numpy(), result.boxes.xywhn.cpu().numpy(),
                    result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy().astype(int),
                ):
                    xc, yc, width, height = map(float, xywhn)
                    detection = {
                        "frame": frame_index, "segment": segment_index,
                        "xyxy": [float(value) for value in xyxy],
                        "xc": xc, "yc": yc, "w": width, "h": height,
                        "area": width * height, "conf": float(confidence), "cls": int(class_id),
                    }
                    detections.append(detection)
                    draw(frame, detection)
            tracker.update(frame_index, detections)
            frame_counts.append(len(detections))
            all_detections.extend(detections)
            cv2.putText(
                frame,
                f"{args.experiment_id} | conf={args.conf:.2f} | t={frame_index/fps:.1f}s",
                (35, 90), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 6, cv2.LINE_AA,
            )
            if (output_width, output_height) != (source_width, source_height):
                frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
            processed_frames += 1
    cap.release()
    writer.release()
    tracks = tracker.finalize()

    frame_counter = Counter(frame_counts)
    class_counts = Counter(d["cls"] for d in all_detections)
    track_lengths = [track["length"] for track in tracks]
    tiny = [
        d for d in all_detections
        if d["w"] < args.tiny_width or d["h"] < args.tiny_height or d["area"] < args.tiny_area
    ]
    per_class = {}
    for class_id, count in sorted(class_counts.items()):
        confidences = [d["conf"] for d in all_detections if d["cls"] == class_id]
        per_class[CLASS_NAMES.get(class_id, str(class_id))] = {
            "count": count,
            "mean_confidence": float(np.mean(confidences)),
            "median_confidence": float(np.median(confidences)),
        }
    stats = {
        "experiment_id": args.experiment_id,
        "weights": str(args.weights.resolve()),
        "video": str(video_path),
        "segments_seconds": [{"start": start, "end": end} for start, end in segments],
        "config": vars(args) | {"weights": str(args.weights), "video": str(args.video), "out_video": str(args.out_video), "out_stats": str(args.out_stats)},
        "frames": processed_frames,
        "average_detections_per_frame": float(np.mean(frame_counts)) if frame_counts else 0.0,
        "frames_with_0_pct": 100.0 * frame_counter[0] / max(processed_frames, 1),
        "frames_with_1_pct": 100.0 * frame_counter[1] / max(processed_frames, 1),
        "frames_with_2_pct": 100.0 * frame_counter[2] / max(processed_frames, 1),
        "frames_with_more_than_2_pct": 100.0 * sum(v for k, v in frame_counter.items() if k > 2) / max(processed_frames, 1),
        "detections_per_class": per_class,
        "tiny_detections": len(tiny),
        "low_confidence_needle_driver": sum(1 for d in all_detections if d["cls"] == 2 and d["conf"] < 0.5),
        "tracks": len(tracks),
        "one_frame_tracks": sum(length == 1 for length in track_lengths),
        "tracks_lte_2_frames": sum(length <= 2 for length in track_lengths),
        "mean_track_persistence_frames": float(np.mean(track_lengths)) if track_lengths else 0.0,
        "median_track_persistence_frames": float(np.median(track_lengths)) if track_lengths else 0.0,
        "max_track_persistence_frames": max(track_lengths, default=0),
        "class_switches": tracker.class_switches,
        "class_switch_frequency": tracker.class_switches / max(tracker.transitions, 1),
        "box_distribution": {
            metric: {
                "min": float(np.min(values)), "median": float(np.median(values)),
                "mean": float(np.mean(values)), "max": float(np.max(values)),
            }
            for metric, values in {
                "width": [d["w"] for d in all_detections],
                "height": [d["h"] for d in all_detections],
                "area": [d["area"] for d in all_detections],
            }.items() if values
        },
    }
    args.out_stats.resolve().write_text(json.dumps(stats, indent=2, default=str))
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
