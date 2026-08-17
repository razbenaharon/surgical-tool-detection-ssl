# Surgical Tool Detection with Semi-Supervised Learning

Object-detection system for surgical **hands and instruments** (classes:
`Empty` = empty hand, `Tweezers`, `Needle_driver`) trained from a very small
labeled set (~70 images) and generalized to **out-of-distribution (OOD)**
surgery video using **semi-supervised learning (SSL) with pseudo-labels**.

Course: Computer Vision — Surgical Applications, HW1.

Authors: Raz Ben-Aharon and Shalev Manassen (shalevmanassen@gmail.com).

## Method (overview)

1. **Baseline** — fine-tune a COCO-pretrained YOLO (`yolo11s`) on the 61 labeled
   training images with augmentation.
2. **Semi-supervised experiments** — pseudo-label the in-distribution and OOD
   videos with various filtering policies, retrain, and evaluate.
3. **Hyperparameter sweep** — test resolutions (640/960/1280), learning rates
   (0.001/0.003/0.01), and model scales (s/m).
4. **Final selection** — the supervised Baseline at inference confidence 0.60
   was selected as the best OOD detector after quantitative diagnostics and
   manual comparison of three finalist models.

## Results (real validation set)

| Model | Train imgs | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|
| Baseline (labeled only) | 61 | 0.972 | 0.929 | 0.991 | **0.809** |
| + ID pseudo-labels (ssl_id) | 763 | 0.978 | 0.926 | 0.986 | 0.837 |
| + OOD pseudo-labels (ssl_ood) | 1430 | 0.954 | 0.889 | 0.971 | 0.792 |

*Validation metrics above are computed on the cleaned validation set (2 confirmed
annotation artifacts removed; see report §2.1). The supervised Baseline was
selected as the final model because naive pseudo-label self-training degraded
OOD video performance through confirmation bias, despite improving some ID
validation metrics. See the [PDF report](reports/HW1_report_final.pdf) for the
full experimental analysis.*

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

- [PDF report](reports/HW1_report_final.pdf) (5 pages)
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
    --out annotated.mp4 --conf 0.60
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
