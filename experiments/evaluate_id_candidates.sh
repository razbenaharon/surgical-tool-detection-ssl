#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 IMGSZ EXPERIMENT_ID [EXPERIMENT_ID ...]" >&2
  exit 2
fi

IMGSZ=$1
shift
ROOT=/home/vmadmin/hw1/hw1_repo
source /home/vmadmin/hw1/venv/bin/activate
export YOLO_CONFIG_DIR=/home/vmadmin/hw1/.config/Ultralytics
cd "$ROOT"

for id in "$@"; do
  run="runs/detect/experiments/ssl_id_runs/$id"
  [[ -f "$run/experiment_manifest.json" ]] || { echo "Skipping incomplete run: $id"; continue; }
  python src/evaluate_validation.py \
    --weights "$run/weights/best.pt" \
    --val-original datasets/validation_versions/val_original \
    --val-clean datasets/validation_versions/val_clean \
    --out "experiments/evaluations/$id" --experiment-id "$id" \
    --training-results "$run/results.csv" --imgsz "$IMGSZ" --batch 8 --device 0
  python src/evaluate_ood.py \
    --weights "$run/weights/best.pt" \
    --video /datashare/HW1/ood_video_data/surg_1.mp4 --segments 0:10 \
    --out-video "evaluation_videos/${id}_conf050.mp4" \
    --out-stats "experiments/ood_diagnostics/${id}_conf050.json" \
    --experiment-id "${id}_conf050" --conf 0.50 --iou 0.6 \
    --imgsz "$IMGSZ" --max-detections 10 --output-width 1280 --device 0
done
