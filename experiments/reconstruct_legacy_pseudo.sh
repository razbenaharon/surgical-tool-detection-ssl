#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/vmadmin/hw1/hw1_repo
source /home/vmadmin/hw1/venv/bin/activate
export YOLO_CONFIG_DIR=/home/vmadmin/hw1/.config/Ultralytics
cd "$ROOT"

python src/pseudo_label.py \
  --weights /home/vmadmin/hw1/legacy/baseline/best.pt \
  --videos /datashare/HW1/id_video_data/20_2_24_1.mp4 /datashare/HW1/id_video_data/4_2_24_B_2.mp4 \
  --out datasets/reconstructed_legacy/pseudo_id --policy-id legacy_id_reference \
  --stride 15 --box-conf-threshold 0.5 --frame-conf-threshold 0.7 \
  --max-detections-per-frame 300 --nms-iou-threshold 0.7 --imgsz 640 --device 0

python src/pseudo_label.py \
  --weights /home/vmadmin/hw1/legacy/ssl_id/best.pt \
  --videos /datashare/HW1/ood_video_data/surg_1.mp4 /datashare/HW1/ood_video_data/4_2_24_A_1.mp4 \
  --out datasets/reconstructed_legacy/pseudo_ood --policy-id legacy_ood_reference \
  --stride 8 --box-conf-threshold 0.5 --frame-conf-threshold 0.7 \
  --max-detections-per-frame 300 --nms-iou-threshold 0.7 --imgsz 640 --device 0
