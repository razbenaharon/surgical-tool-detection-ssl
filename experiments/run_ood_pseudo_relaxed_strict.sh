#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 TEACHER_WEIGHTS TEACHER_IMGSZ TEACHER_ID" >&2
  exit 2
fi

WEIGHTS=$1
IMGSZ=$2
TEACHER_ID=$3
ROOT=/home/vmadmin/hw1/hw1_repo
source /home/vmadmin/hw1/venv/bin/activate
export YOLO_CONFIG_DIR=/home/vmadmin/hw1/.config/Ultralytics
cd "$ROOT"

VIDEOS=(/datashare/HW1/ood_video_data/surg_1.mp4 /datashare/HW1/ood_video_data/4_2_24_A_1.mp4)
COMMON=(--weights "$WEIGHTS" --videos "${VIDEOS[@]}" --stride 15 \
  --box-conf-threshold 0.75 --frame-conf-threshold 0.85 \
  --max-detections-per-frame 2 --nms-iou-threshold 0.6 \
  --imgsz "$IMGSZ" --device 0)

python src/pseudo_label.py "${COMMON[@]}" \
  --out "datasets/pseudo_ood/${TEACHER_ID}_relaxed_conf" \
  --policy-id "${TEACHER_ID}_ood_relaxed_conf"

python src/pseudo_label.py "${COMMON[@]}" \
  --out "datasets/pseudo_ood/${TEACHER_ID}_relaxed_sanity" \
  --policy-id "${TEACHER_ID}_ood_relaxed_sanity" \
  --min-box-width 0.05865 --min-box-height 0.13356 \
  --min-box-area 0.00934 --max-box-area 0.17465

python src/pseudo_label.py "${COMMON[@]}" \
  --out "datasets/pseudo_ood/${TEACHER_ID}_relaxed_temporal" \
  --policy-id "${TEACHER_ID}_ood_relaxed_temporal" \
  --temporal-radius 1 --temporal-min-hits 2 --temporal-iou 0.30 \
  --temporal-mean-conf 0.75

python src/pseudo_label.py "${COMMON[@]}" \
  --out "datasets/pseudo_ood/${TEACHER_ID}_relaxed_combined" \
  --policy-id "${TEACHER_ID}_ood_relaxed_combined" \
  --class-conf 2:0.85 --class-cap 2:120 \
  --min-box-width 0.05865 --min-box-height 0.13356 \
  --min-box-area 0.00934 --max-box-area 0.17465 \
  --temporal-radius 1 --temporal-min-hits 2 --temporal-iou 0.30 \
  --temporal-mean-conf 0.75
