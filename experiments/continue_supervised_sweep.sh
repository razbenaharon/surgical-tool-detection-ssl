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

COMMON=(--data data/supervised_clean.yaml --epochs 120 --patience 30 \
  --project experiments/supervised_runs --model yolo11s.pt --optimizer AdamW \
  --seed 0 --workers 8 --cos_lr --weight-decay 0.0005)

run_candidate sup_s_640_lr010_mod "${COMMON[@]}" --imgsz 640 --batch 16 --lr0 0.010
run_candidate sup_s_960_lr001_mod "${COMMON[@]}" --imgsz 960 --batch 12 --lr0 0.001
run_candidate sup_s_960_lr003_mod "${COMMON[@]}" --imgsz 960 --batch 12 --lr0 0.003
run_candidate sup_s_1280_lr003_mod "${COMMON[@]}" --imgsz 1280 --batch 8 --lr0 0.003
run_candidate sup_s_960_lr003_weak "${COMMON[@]}" --imgsz 960 --batch 12 --lr0 0.003 \
  --mosaic 0.2 --close_mosaic 10 --translate 0.05 --scale 0.25 \
  --hsv_s 0.4 --hsv_v 0.25
run_candidate sup_m_960_lr003_weak \
  --data data/supervised_clean.yaml --epochs 120 --patience 30 \
  --project experiments/supervised_runs --model yolo11m.pt --optimizer AdamW \
  --imgsz 960 --batch 8 --lr0 0.003 --seed 0 --workers 8 --cos_lr \
  --weight-decay 0.0005 --mosaic 0.2 --close_mosaic 10 \
  --translate 0.05 --scale 0.25 --hsv_s 0.4 --hsv_v 0.25
