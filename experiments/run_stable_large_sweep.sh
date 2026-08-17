#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/vmadmin/hw1/hw1_repo
source /home/vmadmin/hw1/venv/bin/activate
export YOLO_CONFIG_DIR=/home/vmadmin/hw1/.config/Ultralytics
cd "$ROOT"
mkdir -p experiments/failures

run_candidate() {
  local id=$1
  shift
  if python src/train.py "$@" --name "$id"; then
    echo "$id completed" | tee "experiments/failures/${id}.status"
  else
    rc=$?
    echo "$id failed rc=$rc" | tee "experiments/failures/${id}.status"
  fi
}

BASE=(--data data/supervised_clean.yaml --epochs 120 --patience 30 \
  --project experiments/supervised_runs --optimizer AdamW --lr0 0.001 \
  --seed 0 --workers 8 --cos_lr --weight-decay 0.0005)

run_candidate sup_s_960_lr001_weak "${BASE[@]}" --model yolo11s.pt \
  --imgsz 960 --batch 12 --mosaic 0.2 --close_mosaic 10 \
  --translate 0.05 --scale 0.25 --hsv_s 0.4 --hsv_v 0.25

run_candidate sup_s_1280_lr001_mod "${BASE[@]}" --model yolo11s.pt \
  --imgsz 1280 --batch 8

run_candidate sup_m_960_lr001_weak "${BASE[@]}" --model yolo11m.pt \
  --imgsz 960 --batch 8 --mosaic 0.2 --close_mosaic 10 \
  --translate 0.05 --scale 0.25 --hsv_s 0.4 --hsv_v 0.25
