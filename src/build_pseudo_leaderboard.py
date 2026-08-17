"""Build consolidated ID/OOD pseudo-label policy tables from saved statistics."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id-root", type=Path, default=Path("datasets/pseudo_id"))
    parser.add_argument("--ood-root", type=Path, default=Path("datasets/pseudo_ood"))
    parser.add_argument("--out-prefix", type=Path, default=Path("experiments/pseudo_label_leaderboard"))
    args = parser.parse_args()

    rows = []
    for domain, root in (("ID", args.id_root), ("OOD", args.ood_root)):
        for path in sorted(root.glob("*/pseudo_stats.json")):
            stats = json.loads(path.read_text())
            confidence = stats.get("distribution_accepted", {}).get("confidence", {})
            classes = stats.get("classes", {})
            config = stats.get("config", {})
            row = {
                "domain": domain,
                "policy_id": stats["policy_id"],
                "stats_path": str(path.resolve()),
                "frames_sampled": stats["frames_sampled"],
                "frames_accepted": stats["frames_accepted"],
                "frames_rejected": stats["frames_rejected"],
                "keep_rate": stats["keep_rate"],
                "boxes_before_filters": stats["boxes_before_filters"],
                "boxes_accepted": stats["boxes_accepted"],
                "mean_confidence": confidence.get("mean"),
                "median_confidence": confidence.get("median"),
                "empty_count": classes.get("Empty", {}).get("count", 0),
                "tweezers_count": classes.get("Tweezers", {}).get("count", 0),
                "needle_driver_count": classes.get("Needle_driver", {}).get("count", 0),
                "empty_pct": classes.get("Empty", {}).get("percentage", 0),
                "tweezers_pct": classes.get("Tweezers", {}).get("percentage", 0),
                "needle_driver_pct": classes.get("Needle_driver", {}).get("percentage", 0),
                "box_conf_threshold": config.get("box_conf_threshold"),
                "frame_conf_threshold": config.get("frame_conf_threshold"),
                "stride": config.get("stride"),
                "max_detections_per_frame": config.get("max_detections_per_frame"),
                "temporal_min_hits": config.get("temporal_min_hits"),
                "rejection_reasons": json.dumps(stats.get("box_rejection_reasons", {}), sort_keys=True),
            }
            rows.append(row)

    prefix = args.out_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path, json_path, md_path = prefix.with_suffix(".csv"), prefix.with_suffix(".json"), prefix.with_suffix(".md")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    json_path.write_text(json.dumps({"policies": rows}, indent=2))

    headers = ["domain", "policy", "frames", "boxes", "mean conf", "Empty", "Tweezers", "Needle", "rejections"]
    lines = ["# Pseudo-label policy leaderboard", "", "| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        lines.append(
            "| " + " | ".join(
                str(value) for value in (
                    row["domain"], row["policy_id"],
                    f"{row['frames_accepted']}/{row['frames_sampled']}", row["boxes_accepted"],
                    "" if row["mean_confidence"] is None else f"{row['mean_confidence']:.3f}",
                    row["empty_count"], row["tweezers_count"], row["needle_driver_count"],
                    row["rejection_reasons"],
                )
            ) + " |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {csv_path}, {json_path}, and {md_path}")


if __name__ == "__main__":
    main()
