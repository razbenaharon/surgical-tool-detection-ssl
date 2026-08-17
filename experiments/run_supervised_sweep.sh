#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/vmadmin/hw1/hw1_repo
source /home/vmadmin/hw1/venv/bin/activate
export YOLO_CONFIG_DIR=/home/vmadmin/hw1/.config/Ultralytics
cd "$ROOT"

python src/build_ssl_dataset.py \
  --labeled /home/vmadmin/hw1/labeled_image_data \
  --val-root datasets/validation_versions/val_clean \
  --out datasets/supervised_clean \
  --yaml data/supervised_clean.yaml

# Exact clean-validation counterpart of the legacy baseline. This isolates
# whether the two removed annotations alter early stopping / best epoch.
python src/train.py --data data/supervised_clean.yaml --name sup_ref_640_auto_clean \
  --project experiments/supervised_runs --model yolo11s.pt --optimizer auto \
  --epochs 150 --imgsz 640 --batch 16 --patience 50 --lr0 0.01 --seed 0

COMMON=(--data data/supervised_clean.yaml --epochs 120 --patience 30 \
  --project experiments/supervised_runs --model yolo11s.pt --optimizer AdamW \
  --batch 16 --seed 0 --workers 8 --cos_lr --weight-decay 0.0005)

python src/train.py "${COMMON[@]}" --name sup_s_640_lr001_mod  --imgsz 640  --lr0 0.001
python src/train.py "${COMMON[@]}" --name sup_s_640_lr003_mod  --imgsz 640  --lr0 0.003
python src/train.py "${COMMON[@]}" --name sup_s_640_lr010_mod  --imgsz 640  --lr0 0.010
python src/train.py "${COMMON[@]}" --name sup_s_960_lr001_mod  --imgsz 960  --lr0 0.001 --batch 12
python src/train.py "${COMMON[@]}" --name sup_s_960_lr003_mod  --imgsz 960  --lr0 0.003 --batch 12
python src/train.py "${COMMON[@]}" --name sup_s_1280_lr003_mod --imgsz 1280 --lr0 0.003 --batch 8
python src/train.py "${COMMON[@]}" --name sup_s_960_lr003_weak --imgsz 960 --lr0 0.003 --batch 12 \
  --mosaic 0.2 --close_mosaic 10 --translate 0.05 --scale 0.25 \
  --hsv_s 0.4 --hsv_v 0.25
python src/train.py --data data/supervised_clean.yaml --epochs 120 --patience 30 \
  --project experiments/supervised_runs --name sup_m_960_lr003_weak \
  --model yolo11m.pt --optimizer AdamW --imgsz 960 --batch 8 --lr0 0.003 \
  --seed 0 --workers 8 --cos_lr --weight-decay 0.0005 \
  --mosaic 0.2 --close_mosaic 10 --translate 0.05 --scale 0.25 \
  --hsv_s 0.4 --hsv_v 0.25
