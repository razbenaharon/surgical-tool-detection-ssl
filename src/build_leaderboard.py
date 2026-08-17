"""Merge candidate registry, validation metrics, and OOD diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=Path("experiments/candidate_registry.json"))
    parser.add_argument("--evaluations", type=Path, default=Path("experiments/evaluations"))
    parser.add_argument("--ood-diagnostics", type=Path, default=Path("experiments/ood_diagnostics"))
    parser.add_argument("--out-prefix", type=Path, default=Path("experiments/leaderboard"))
    args = parser.parse_args()

    candidates = {item["experiment_id"]: item for item in json.loads(args.registry.read_text())}
    for path in args.evaluations.glob("*/validation_metrics.json"):
        report = json.loads(path.read_text())
        experiment_id = report["experiment_id"]
        candidates.setdefault(experiment_id, {"experiment_id": experiment_id, "manual_visual_rating": ""})
        candidates[experiment_id]["validation_report"] = str(path.resolve())
        candidates[experiment_id]["validation"] = report["validation"]
        candidates[experiment_id]["validation_delta"] = report["delta_clean_minus_original"]
        if report.get("training"):
            candidates[experiment_id]["best_epoch"] = report["training"]["best_epoch"]
            candidates[experiment_id]["train_loss"] = report["training"]["train_loss"]
            candidates[experiment_id]["validation_loss"] = report["training"]["validation_loss"]

    candidate_ids = sorted(candidates, key=len, reverse=True)
    for path in args.ood_diagnostics.glob("*.json"):
        report = json.loads(path.read_text())
        diagnostic_id = report["experiment_id"]
        candidate_id = next((item for item in candidate_ids if diagnostic_id.startswith(item)), None)
        if candidate_id:
            candidates[candidate_id].setdefault("ood_evaluations", []).append(
                report | {"diagnostic_path": str(path.resolve())}
            )

    rows = []
    for candidate in candidates.values():
        variants = candidate.get("ood_evaluations") or [None]
        for ood in variants:
            original = candidate.get("validation", {}).get("original", {})
            clean = candidate.get("validation", {}).get("clean", {})
            row = {
                key: candidate.get(key)
                for key in (
                    "experiment_id", "parent_checkpoint", "training_dataset", "real_image_count",
                    "pseudo_image_count", "pseudo_label_policy", "filtering_method", "imgsz",
                    "model_size", "lr", "batch", "augmentation_config", "best_epoch",
                    "train_loss", "validation_loss", "manual_visual_rating",
                )
            }
            for prefix, values in (("original", original), ("clean", clean)):
                for metric in ("precision", "recall", "mAP50", "mAP50-95"):
                    row[f"{prefix}_{metric}"] = values.get(metric)
            if ood:
                per_class = ood.get("detections_per_class", {})
                video_path = ood["config"]["out_video"]
                is_full_video = "full" in Path(video_path).stem
                row.update(
                    {
                        "inference_confidence": ood["config"]["conf"],
                        "preview_video_path": "" if is_full_video else video_path,
                        "full_video_path": video_path if is_full_video else "",
                        "ood_diagnostic_path": ood["diagnostic_path"],
                        "ood_avg_detections_per_frame": ood["average_detections_per_frame"],
                        "ood_frames_0_pct": ood["frames_with_0_pct"],
                        "ood_frames_1_pct": ood["frames_with_1_pct"],
                        "ood_frames_2_pct": ood["frames_with_2_pct"],
                        "ood_frames_more_than_2_pct": ood["frames_with_more_than_2_pct"],
                        "ood_empty_count": per_class.get("Empty", {}).get("count", 0),
                        "ood_tweezers_count": per_class.get("Tweezers", {}).get("count", 0),
                        "ood_needle_driver_count": per_class.get("Needle_driver", {}).get("count", 0),
                        "ood_one_frame_tracks": ood["one_frame_tracks"],
                        "ood_tracks_lte_2_frames": ood["tracks_lte_2_frames"],
                        "ood_mean_track_persistence_frames": ood["mean_track_persistence_frames"],
                        "ood_class_switches": ood["class_switches"],
                        "ood_class_switch_frequency": ood["class_switch_frequency"],
                        "ood_tiny_detections": ood["tiny_detections"],
                        "ood_low_confidence_needle_driver": ood["low_confidence_needle_driver"],
                    }
                )
            rows.append(row)

    prefix = args.out_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path, json_path, markdown_path = prefix.with_suffix(".csv"), prefix.with_suffix(".json"), prefix.with_suffix(".md")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps({"candidates": list(candidates.values()), "rows": rows}, indent=2, default=str))

    headers = [
        "experiment", "conf", "orig mAP50-95", "clean mAP50-95", "avg det/frame",
        "0 frames %", ">2 frames %", "1-frame", "switches", "manual rating",
    ]
    lines = ["# Candidate leaderboard", "", "| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        lines.append(
            "| " + " | ".join(
                str(value if value is not None else "")
                for value in (
                    row.get("experiment_id"), row.get("inference_confidence"),
                    row.get("original_mAP50-95"), row.get("clean_mAP50-95"),
                    row.get("ood_avg_detections_per_frame"), row.get("ood_frames_0_pct"),
                    row.get("ood_frames_more_than_2_pct"), row.get("ood_one_frame_tracks"),
                    row.get("ood_class_switches"), row.get("manual_visual_rating", ""),
                )
            ) + " |"
        )
    markdown_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {csv_path}, {json_path}, and {markdown_path}")


if __name__ == "__main__":
    main()
