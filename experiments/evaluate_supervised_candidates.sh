#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/vmadmin/hw1/hw1_repo
RUN_ROOT="$ROOT/runs/detect/experiments/supervised_runs"
source /home/vmadmin/hw1/venv/bin/activate
export YOLO_CONFIG_DIR=/home/vmadmin/hw1/.config/Ultralytics
cd "$ROOT"

ids=(
  sup_ref_640_auto_clean sup_s_640_lr001_mod sup_s_640_lr003_mod
  sup_s_640_lr010_mod sup_s_960_lr001_mod sup_s_960_lr003_mod
  sup_s_1280_lr003_mod sup_s_960_lr003_weak sup_m_960_lr003_weak
  sup_s_960_lr001_weak sup_s_1280_lr001_mod sup_m_960_lr001_weak
)
sizes=(640 640 640 640 960 960 1280 960 960 960 1280 960)

for index in "${!ids[@]}"; do
  id=${ids[$index]}
  size=${sizes[$index]}
  run="$RUN_ROOT/$id"
  if [[ ! -f "$run/experiment_manifest.json" ]]; then
    echo "Skipping incomplete/failed run: $id"
    continue
  fi
  python src/evaluate_validation.py \
    --weights "$run/weights/best.pt" \
    --val-original datasets/validation_versions/val_original \
    --val-clean datasets/validation_versions/val_clean \
    --out "experiments/evaluations/$id" --experiment-id "$id" \
    --training-results "$run/results.csv" --imgsz "$size" --batch 8 --device 0

  prefix=$(printf "%02d" "$((index + 3))")
  python src/evaluate_ood.py \
    --weights "$run/weights/best.pt" \
    --video /datashare/HW1/ood_video_data/surg_1.mp4 --segments 0:10 \
    --out-video "evaluation_videos/${prefix}_${id}_conf050.mp4" \
    --out-stats "experiments/ood_diagnostics/${id}_conf050.json" \
    --experiment-id "${id}_conf050" --conf 0.50 --iou 0.6 \
    --imgsz "$size" --max-detections 10 --output-width 1280 --device 0
done
