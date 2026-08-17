#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 TEACHER_WEIGHTS TEACHER_IMGSZ TEACHER_ID POLICY [POLICY ...]" >&2
  exit 2
fi

WEIGHTS=$1
IMGSZ=$2
TEACHER_ID=$3
shift 3
POLICIES=("$@")

ROOT=/home/vmadmin/hw1/hw1_repo
source /home/vmadmin/hw1/venv/bin/activate
export YOLO_CONFIG_DIR=/home/vmadmin/hw1/.config/Ultralytics
cd "$ROOT"
mkdir -p experiments/failures

for policy in "${POLICIES[@]}"; do
  repeat=5
  id="ssl_id_${TEACHER_ID}_${policy}_r${repeat}"
  pseudo="datasets/pseudo_id/${TEACHER_ID}_${policy}"
  if [[ ! -f "$pseudo/pseudo_stats.json" ]]; then
    echo "Missing pseudo-label statistics for $pseudo; skipping" | tee "experiments/failures/${id}.status"
    continue
  fi

  if ! python src/build_ssl_dataset.py \
      --labeled /home/vmadmin/hw1/labeled_image_data \
      --val-root datasets/validation_versions/val_clean \
      --pseudo "$pseudo" --out "datasets/ssl_id/${id}" \
      --yaml "data/${id}.yaml" --real-repeat "$repeat"; then
    echo "$id dataset build failed" | tee "experiments/failures/${id}.status"
    continue
  fi

  if python src/train.py --data "data/${id}.yaml" --name "$id" \
      --project experiments/ssl_id_runs --model "$WEIGHTS" --optimizer AdamW \
      --epochs 80 --patience 18 --imgsz "$IMGSZ" --batch 16 --lr0 0.001 \
      --seed 0 --workers 8 --mosaic 0.2 --close_mosaic 10 \
      --translate 0.05 --scale 0.25 --hsv_s 0.4 --hsv_v 0.25; then
    echo "$id completed" | tee "experiments/failures/${id}.status"
  else
    rc=$?
    echo "$id failed rc=$rc" | tee "experiments/failures/${id}.status"
  fi
done
