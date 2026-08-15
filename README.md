# Surgical Tool Detection with Semi-Supervised Learning

Object-detection system for surgical **hands and instruments** (classes:
`Empty` = empty hand, `Tweezers`, `Needle_driver`) trained from a very small
labeled set (~70 images) and generalized to **out-of-distribution (OOD)**
surgery video using **semi-supervised learning (SSL) with pseudo-labels**.

Course: Computer Vision — Surgical Applications, HW1.

## Method (overview)

1. **Baseline** — fine-tune a COCO-pretrained YOLO (`yolo11s`) on the 61 labeled
   training images with augmentation.
2. **Pseudo-label ID videos** — run the baseline on the two in-distribution
   videos, keep only high-confidence detections as pseudo-labels.
3. **Refine** — retrain on labeled + ID pseudo-labels (real val set kept for
   honest evaluation).
4. **Generalize to OOD** — pseudo-label the OOD video, refine again.

## Results (real validation set)

| Model | Train imgs | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|
| Baseline (labeled only) | 61 | 0.972 | 0.870 | 0.925 | 0.754 |
| + ID pseudo-labels | 763 | 0.979 | 0.866 | 0.920 | **0.778** |
| + OOD pseudo-labels (final) | 1430 | 0.953 | 0.833 | 0.906 | 0.739 |

ID pseudo-labels improve localization (mAP@50-95 0.754→0.778); the OOD round
trades a little ID accuracy for OOD robustness (clearest qualitatively — the
final model detects hands/tools the baseline misses on the OOD video).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
# PyTorch matching your CUDA driver (A10 course VM = CUDA 12.8):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Model weights

Download the final trained weights and place them at `weights/best.pt`:

> **Weights download:** [best.pt (GitHub Release)](https://github.com/razbenaharon/surgical-tool-detection-ssl/releases/download/v1.0/best.pt)

## Submission artifacts

- [PDF report](reports/HW1_report_final.pdf) (4 pages)
- [Annotated OOD video](https://github.com/razbenaharon/surgical-tool-detection-ssl/releases/download/v1.0/surg_1_annotated.mp4)
- [Final model weights](https://github.com/razbenaharon/surgical-tool-detection-ssl/releases/download/v1.0/best.pt)

## Usage

Predict on a single image (prints YOLO-format `x_center y_center w h conf class`):

```bash
python predict.py --image path/to/image.jpg --weights weights/best.pt \
    --out-img annotated.jpg --out-txt preds.txt
```

Annotate a full video (writes an mp4 with overlaid boxes + class + confidence):

```bash
python video.py --video path/to/surgery.mp4 --weights weights/best.pt \
    --out annotated.mp4
```

## Reproducing training

Data lives on the course server at `/datashare/HW1/`. See `src/` and
`REPRODUCE.md` for the exact commands (EDA → baseline → pseudo-label → refine).

```
src/eda.py                 Exploratory data analysis + figures
src/train.py               Train/fine-tune YOLO (baseline and refinement)
src/pseudo_label.py        Generate high-confidence pseudo-labels from videos
src/build_ssl_dataset.py   Merge labeled + pseudo-labeled data into a training set
```

## Classes

| id | name          | meaning              |
|----|---------------|----------------------|
| 0  | Empty         | hand, no tool        |
| 1  | Tweezers      | tweezers             |
| 2  | Needle_driver | needle driver        |
