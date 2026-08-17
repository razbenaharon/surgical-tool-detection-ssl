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

policies=(p1 p2 p3 p4 p3_sanity_temporal)
repeats=(1 5 5 5 5)
for index in "${!policies[@]}"; do
  policy=${policies[$index]}
  repeat=${repeats[$index]}
  id="ssl_id_${TEACHER_ID}_${policy}_r${repeat}"
  python src/build_ssl_dataset.py \
    --labeled /home/vmadmin/hw1/labeled_image_data \
    --val-root datasets/validation_versions/val_clean \
    --pseudo "datasets/pseudo_id/${TEACHER_ID}_${policy}" \
    --out "datasets/ssl_id/${id}" --yaml "data/${id}.yaml" --real-repeat "$repeat"
  python src/train.py --data "data/${id}.yaml" --name "$id" \
    --project experiments/ssl_id_runs --model "$WEIGHTS" --optimizer AdamW \
    --epochs 100 --patience 25 --imgsz "$IMGSZ" --batch 8 --lr0 0.001 \
    --seed 0 --workers 8 --mosaic 0.2 --close_mosaic 10 \
    --translate 0.05 --scale 0.25 --hsv_s 0.4 --hsv_v 0.25
done
