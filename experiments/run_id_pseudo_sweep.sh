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

VIDEOS=(/datashare/HW1/id_video_data/20_2_24_1.mp4 /datashare/HW1/id_video_data/4_2_24_B_2.mp4)
policies=(p1 p2 p3 p4)
box_thresholds=(0.50 0.70 0.80 0.90)
frame_thresholds=(0.70 0.80 0.90 0.95)

for index in "${!policies[@]}"; do
  policy=${policies[$index]}
  python src/pseudo_label.py --weights "$WEIGHTS" --videos "${VIDEOS[@]}" \
    --out "datasets/pseudo_id/${TEACHER_ID}_${policy}" \
    --policy-id "${TEACHER_ID}_id_${policy}" --stride 15 \
    --box-conf-threshold "${box_thresholds[$index]}" \
    --frame-conf-threshold "${frame_thresholds[$index]}" \
    --max-detections-per-frame 300 --nms-iou-threshold "$([[ $policy == p1 ]] && echo 0.7 || echo 0.6)" \
    --imgsz "$IMGSZ" --device 0
done

# A precision-first variant grounded in the real train-box distribution.
python src/pseudo_label.py --weights "$WEIGHTS" --videos "${VIDEOS[@]}" \
  --out "datasets/pseudo_id/${TEACHER_ID}_p3_sanity_temporal" \
  --policy-id "${TEACHER_ID}_id_p3_sanity_temporal" --stride 15 \
  --box-conf-threshold 0.80 --frame-conf-threshold 0.90 \
  --max-detections-per-frame 2 --min-box-width 0.05865 \
  --min-box-height 0.13356 --min-box-area 0.00934 --max-box-area 0.17465 \
  --temporal-radius 1 --temporal-min-hits 2 --temporal-iou 0.30 \
  --temporal-mean-conf 0.80 --nms-iou-threshold 0.6 \
  --imgsz "$IMGSZ" --device 0
