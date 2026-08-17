# Reproducing the full pipeline

All commands run on the course GPU VM, from `~/hw1/hw1_repo`, inside the venv:

```bash
cd ~/hw1/hw1_repo
source ~/hw1/venv/bin/activate
```

Data is read-only at `/datashare/HW1/`. A writable copy of the labeled set is
at `/home/student/hw1/labeled_image_data` (so pseudo-labels can be added).

## 0. EDA

```bash
python src/eda.py --data /home/student/hw1/labeled_image_data \
    --videos /datashare/HW1 --out tmp/eda
```

## 1. Supervised baseline

```bash
python src/train.py --data data/labeled.yaml --name baseline \
    --model yolo11s.pt --epochs 200 --imgsz 640 --batch 16 --patience 50
# -> runs/.../baseline/weights/best.pt
```

## 2. Pseudo-label the in-distribution (ID) videos

```bash
python src/pseudo_label.py \
    --weights runs/.../baseline/weights/best.pt \
    --videos /datashare/HW1/id_video_data/20_2_24_1.mp4 \
             /datashare/HW1/id_video_data/4_2_24_B_2.mp4 \
    --out datasets/pseudo_id --stride 15 --box-conf 0.5 --frame-conf 0.7
```

## 3. Refine on labeled + ID pseudo-labels

```bash
python src/build_ssl_dataset.py \
    --labeled /home/student/hw1/labeled_image_data \
    --pseudo datasets/pseudo_id \
    --out datasets/ssl_id --yaml data/ssl_id.yaml

python src/train.py --data data/ssl_id.yaml --name ssl_id \
    --model runs/.../baseline/weights/best.pt --epochs 120 --imgsz 640 \
    --batch 16 --patience 40
```

## 4. Generalize to the out-of-distribution (OOD) video

```bash
python src/pseudo_label.py \
    --weights runs/.../ssl_id/weights/best.pt \
    --videos /datashare/HW1/ood_video_data/surg_1.mp4 \
             /datashare/HW1/ood_video_data/4_2_24_A_1.mp4 \
    --out datasets/pseudo_ood --stride 10 --box-conf 0.5 --frame-conf 0.7

python src/build_ssl_dataset.py \
    --labeled /home/student/hw1/labeled_image_data \
    --pseudo datasets/pseudo_id datasets/pseudo_ood \
    --out datasets/ssl_ood --yaml data/ssl_ood.yaml

python src/train.py --data data/ssl_ood.yaml --name ssl_ood \
    --model runs/.../ssl_id/weights/best.pt --epochs 120 --imgsz 640 \
    --batch 16 --patience 40
```

**Note:** Steps 2–4 document the SSL experiments that were performed. However,
the final evaluation showed that naive pseudo-labeling degraded OOD performance.
The final submitted model uses only the supervised Baseline weights from step 1.

## 5. Deliverables

The final model is the supervised Baseline (step 1), not the ssl_ood model.

```bash
# annotated OOD video (Baseline at conf=0.60)
python video.py --video /datashare/HW1/ood_video_data/surg_1.mp4 \
    --weights weights/best.pt \
    --out deliverables/surg_1_annotated.mp4 --conf 0.60

# single-image prediction (YOLO-format output)
python predict.py --image some_frame.jpg \
    --weights weights/best.pt --out-img pred.jpg --out-txt pred.txt
```
