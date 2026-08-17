#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/vmadmin/hw1/hw1_repo
source /home/vmadmin/hw1/venv/bin/activate
export YOLO_CONFIG_DIR=/home/vmadmin/hw1/.config/Ultralytics
cd "$ROOT"

models=(baseline ssl_id ssl_ood)
prefixes=(00 01 02)
for confidence in 0.25 0.40 0.50 0.60 0.70; do
  tag=${confidence/./}
  for index in 0 1 2; do
    model=${models[$index]}
    prefix=${prefixes[$index]}
    experiment="legacy_${model}_conf${tag}"
    python src/evaluate_ood.py \
      --weights "/home/vmadmin/hw1/legacy/${model}/best.pt" \
      --video /datashare/HW1/ood_video_data/surg_1.mp4 \
      --segments 0:10 \
      --out-video "evaluation_videos/${prefix}_${model}_current_conf${tag}.mp4" \
      --out-stats "experiments/ood_diagnostics/${experiment}.json" \
      --experiment-id "$experiment" --conf "$confidence" --iou 0.6 \
      --imgsz 640 --max-detections 10 --output-width 1280 --device 0
  done
done
