"""Generate the HW1 PDF report (<= 5 pages) with reportlab.

Revised version — reflects the full experiment campaign and corrected findings:
  * Validation annotation audit
  * Corrected optimizer/LR description
  * SSL failure analysis (confirmation bias)
  * Hyperparameter sweep
  * Pseudo-label policy experiments
  * Finalist comparison and manual selection
  * Final model: supervised Baseline at conf=0.60

Section structure per assignment requirements:
  1. Exploratory Data Analysis
     1.1 Visualization of Some Images
     1.2 Insights from simply "looking" at the data
     1.3 Data distribution analysis
  2. Experiments
     2.1 Data loading, pre-processing and cleaning
     2.2 Training techniques
     2.3 Regularization (if any)
     2.4 Hyper parameter tuning (if any)
     2.5 Train + valid loss graph
     2.6 Train + valid mAP's graphs
  3. Discussion and Conclusions

Usage:
    python src/report_gen.py --root . --out reports/HW1_report_final.pdf
"""
import argparse
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

PAGE_W, PAGE_H = A4
MARGIN = 1.3 * cm
CONTENT_W = PAGE_W - 2 * MARGIN


def styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("TitleBig", parent=ss["Title"], fontSize=17, spaceAfter=2))
    ss.add(ParagraphStyle("Sub", parent=ss["Normal"], fontSize=9.2,
                          textColor=colors.grey, alignment=TA_CENTER, spaceAfter=5))
    ss.add(ParagraphStyle("H", parent=ss["Heading2"], fontSize=12,
                          textColor=colors.HexColor("#1a3d6d"), spaceBefore=7, spaceAfter=3))
    ss.add(ParagraphStyle("H3s", parent=ss["Heading3"], fontSize=9.8,
                          textColor=colors.HexColor("#333333"), spaceBefore=5, spaceAfter=2))
    ss.add(ParagraphStyle("Body", parent=ss["Normal"], fontSize=8.4, leading=10.8,
                          alignment=TA_JUSTIFY, spaceAfter=3))
    ss.add(ParagraphStyle("Cap", parent=ss["Normal"], fontSize=7.3,
                          textColor=colors.grey, alignment=TA_CENTER, spaceAfter=5))
    return ss


def img(path, width, cap=None, S=None, story=None, max_h=None):
    p = Path(path)
    if not p.exists():
        story.append(Paragraph(f"[missing figure: {path}]", S["Cap"]))
        return
    from PIL import Image as PILImage
    with PILImage.open(p) as im:
        w, h = im.size
    ar = h / w
    height = width * ar
    if max_h and height > max_h:
        height = max_h
        width = height / ar
    im_fl = Image(str(p), width=width, height=height)
    story.append(im_fl)
    if cap:
        story.append(Paragraph(cap, S["Cap"]))


def kv_table(rows, colw, S, fontsize=7.2):
    t = Table(rows, colWidths=colw)
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), fontsize),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f8")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9c9c9")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
    ]))
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--metrics", default="reports/metrics.json")
    ap.add_argument("--out", default="reports/HW1_report_final.pdf")
    args = ap.parse_args()
    root = Path(args.root)
    S = styles()

    M = json.loads(Path(args.metrics).read_text()) if Path(args.metrics).exists() else {}
    eda = {}
    eda_path = root / "reports/eda/dataset_summary.json"
    if eda_path.exists():
        eda = json.loads(eda_path.read_text())
    vmeta = {}
    vpath = root / "reports/eda/video_metadata.json"
    if vpath.exists():
        vmeta = json.loads(vpath.read_text())

    story = []
    A = story.append

    tr = eda.get("train", {})
    va = eda.get("val", {})
    def cc(d, k):
        return d.get("class_counts", {}).get(k, 0)

    # ---------- Header ----------
    A(Paragraph("Surgical Tool Detection with Semi-Supervised Learning", S["TitleBig"]))
    A(Paragraph("Computer Vision &mdash; Surgical Applications, HW1", S["Sub"]))
    A(Paragraph("Authors: Raz Ben-Aharon and Shalev Manassen (shalevmanassen@gmail.com)",
               S["Sub"]))

    # ================= 1. Exploratory Data Analysis =================
    A(Paragraph("1. Exploratory Data Analysis", S["H"]))
    A(Paragraph(
        "The labeled set contains only <b>%d training</b> and <b>%d validation</b> images "
        "(&lt;100 total, as stated in the assignment), all native 4K (3840&times;2160). "
        "Each frame comes from a leg-suturing surgery video; every box encloses a gloved hand "
        "together with whatever it holds, and the class label is defined by what the hand "
        "holds: <b>Empty</b> (no tool), <b>Tweezers</b>, or <b>Needle_driver</b>."
        % (tr.get("n_images", 61), va.get("n_images", 10)), S["Body"]))

    A(Paragraph("1.1 Visualization of Some Images", S["H3s"]))
    A(Paragraph(
        "Fig. 1 shows a random sample of labeled training frames with their "
        "ground-truth boxes drawn (green = Empty, blue = Tweezers, red = Needle_driver).",
        S["Body"]))
    img(root / "reports/eda/samples_grid.png", CONTENT_W * 0.86,
        "Fig 1. Nine labeled frames with ground-truth boxes overlaid "
        "(green = Empty, blue = Tweezers, red = Needle_driver).", S, story)

    A(Paragraph("1.2 Insights from simply \u201clooking\u201d at the data", S["H3s"]))
    A(Paragraph(
        "Looking through the labeled frames and the raw videos suggests several practical "
        "properties: (i) every frame is a static top-down camera view of the "
        "same operating field, so the two hands occupy a fairly consistent region; "
        "(ii) boxes are large relative to the image (15&ndash;35% of frame width, "
        "covering the hand-plus-tool, not just the instrument tip); "
        "(iii) the scene is visually difficult: specular highlights off wet tissue "
        "and metal instruments, blood, surgical drapes with similar colour to the gloves, "
        "and partial occlusion; (iv) almost every frame contains exactly one or two "
        "hand+tool boxes (Fig. 2, right) and none of the 71 labeled images has zero boxes; "
        "(v) the OOD video was filmed with different lighting and a different camera "
        "angle/zoom than the ID videos &mdash; the main OOD gap is camera/lighting appearance.",
        S["Body"]))

    A(Paragraph("1.3 Data distribution analysis", S["H3s"]))
    rows = [["Split", "Images", "Boxes", "Empty", "Tweezers", "Needle_driver", "Boxes/img"]]
    for name, d in [("train", tr), ("val", va)]:
        rows.append([name, d.get("n_images", "-"), d.get("n_boxes", "-"),
                     cc(d, "Empty"), cc(d, "Tweezers"), cc(d, "Needle_driver"),
                     f"{d.get('boxes_per_image_mean', 0):.2f}"])
    A(kv_table(rows, [1.7 * cm] + [1.9 * cm] * 4 + [2.1 * cm] + [2.0 * cm], S))
    A(Spacer(1, 3))
    A(Paragraph(
        "<b>Empty is the clear minority class</b> in the labeled set (26/135 &asymp; 19% of "
        "train boxes, 2/22 &asymp; 9% of val boxes), while Tweezers and Needle_driver are "
        "roughly balanced with each other. This class imbalance motivates the pseudo-labeling "
        "experiments in &sect;2 (frames where a hand is empty are common in the raw video "
        "but under-represented in the small labeled sample).", S["Body"]))
    figs2 = [Paragraph("", S["Body"])]
    row2 = []
    cell_w = CONTENT_W / 2 - 4
    from PIL import Image as PILImage
    for pth, cap in [("reports/eda/class_distribution.png",
                      "class counts by split"),
                     ("reports/eda/bbox_center_heatmap.png",
                      "spatial density of box centers")]:
        p = root / pth
        if p.exists():
            with PILImage.open(p) as im:
                w, h = im.size
            row2.append(Image(str(p), width=cell_w, height=cell_w * h / w))
    if row2:
        t = Table([row2], colWidths=[CONTENT_W / 2] * len(row2))
        t.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
        figs2.append(t)
    figs2.append(Paragraph(
        "Fig 2. Left: bounding-box counts per class and split (Empty is the minority class). "
        "Right: spatial density of box centers (x_center, y_center) over all labeled boxes "
        "&mdash; hands are concentrated in the lower-middle of the frame, matching the fixed "
        "top-down camera setup.", S["Cap"]))
    A(KeepTogether(figs2))

    img(root / "reports/eda/box_size_distribution.png", CONTENT_W * 0.92,
        "Fig 3. Left: distribution of relative box size (&radic;area as a fraction of the "
        "frame). Right: box width vs. height (fraction of frame); most boxes lie above the "
        "diagonal, i.e. are taller than they are wide.", S, story,
        max_h=4.2 * cm)

    if vmeta:
        n_id = sum(1 for k in vmeta if k.startswith("id_video_data"))
        n_ood = sum(1 for k in vmeta if k.startswith("ood_video_data"))
        A(Paragraph(
            "In addition to the labeled images, the unlabeled videos on the course server "
            "(<code>/datashare/HW1/</code>) contain %d in-distribution (ID) clips and %d "
            "out-of-distribution (OOD) clips, all 3840&times;2160 at 29.97&nbsp;fps; ID clips "
            "run &asymp;180&nbsp;s each. These are used for pseudo-labeling in &sect;2."
            % (n_id, n_ood), S["Body"]))

    A(PageBreak())

    # ================= 2. Experiments =================
    A(Paragraph("2. Experiments", S["H"]))
    A(Paragraph(
        "<b>Overview.</b> Following the assignment's SSL guideline, we: (1) fine-tuned a "
        "COCO-pretrained <b>YOLO11s</b> detector on the 61 labeled training images; "
        "(2) pseudo-labeled the ID and OOD videos with various filtering policies; "
        "(3) retrained and evaluated each variant. Additionally, we ran a supervised "
        "hyperparameter sweep over resolution, learning rate and model scale. "
        "The result was unexpected: <b>the original supervised Baseline produced the best "
        "OOD behavior</b>, while pseudo-label self-training degraded OOD performance "
        "through confirmation bias (&sect;3). The 10 real labeled validation images were "
        "kept fixed and untouched throughout.", S["Body"]))

    A(Paragraph("2.1 Data loading, pre-processing and cleaning", S["H3s"]))
    A(Paragraph(
        "<b>Loading.</b> Images and YOLO-format label files are read through Ultralytics' "
        "standard detection dataloader from a <code>data/*.yaml</code> config. For the "
        "SSL rounds a custom script (<code>build_ssl_dataset.py</code>) combines labeled and "
        "pseudo-labeled images into one training set while the original 10-image validation "
        "set is passed through unchanged. "
        "<b>Pre-processing.</b> Every image is letterbox-resized to the target resolution "
        "(640&times;640 for most experiments, 960 or 1280 for the higher-resolution sweep) "
        "with aspect ratio preserved and grey padding. "
        "<b>Split.</b> The original train/val split provided with the assignment (61/10 images) "
        "was kept as-is.", S["Body"]))
    A(Paragraph(
        "<b>Validation annotation audit.</b> Manual inspection of the 10 validation images "
        "identified two suspected annotation artifacts in image "
        "<code>ff8c22da-output_0182.png</code>: a 14&times;14-pixel <code>Needle_driver</code> "
        "box on dark background (row 2) and a 10&times;19-pixel <code>Tweezers</code> box at "
        "a glove/background boundary (row 4). These are inconsistent with the dataset "
        "semantics, which define a bounding box as the <i>entire hand plus the held object</i>, "
        "not as a tiny instrument-tip region. With only 10 validation images and &asymp;22 "
        "original boxes, these two artifacts represent &asymp;9% of the evaluation set. To "
        "measure their impact, we created two validation snapshots: <b>val_original</b> "
        "(byte-identical to the provided course data) and <b>val_clean</b> (only those two "
        "confirmed erroneous annotations removed). For the Baseline: original mAP50-95 "
        "= <b>0.754</b>, clean mAP50-95 = <b>0.809</b>. Despite this metric increase, "
        "replay of epoch selection on the clean split still selected <b>epoch 128</b>, "
        "confirming the checkpoint was not affected. We report both metrics throughout to "
        "preserve transparency; the official source data was never overwritten.",
        S["Body"]))

    A(Paragraph("2.2 Training techniques", S["H3s"]))
    A(Paragraph(
        "<b>Transfer learning.</b> All experiments fine-tune <b>YOLO11s</b> "
        "(9.4M parameters) starting from COCO-pretrained weights, with no layers frozen. "
        "<b>Optimizer &amp; schedule.</b> "
        "Ultralytics' <code>optimizer=auto</code> resolved to <b>AdamW</b> (confirmed in "
        "each run's <code>args.yaml</code>: momentum 0.9, weight decay 5e-4) with an "
        "effective initial LR of &asymp;<b>0.001429</b> on a cosine-annealed schedule "
        "(<code>cos_lr=True</code>) and a 3-epoch linear warmup. Note: the reported "
        "<code>lr0=0.01</code> in the configuration was overridden by the auto-optimizer "
        "logic. Batch size 16, image size 640 (or higher in the sweep), mixed-precision "
        "(AMP) training. "
        "<b>Data augmentation</b> (moderate configuration): mosaic (p=1.0, disabled for final "
        "<code>close_mosaic</code> epochs), HSV colour jitter (h=0.015, s=0.7, v=0.4), "
        "horizontal flip (p=0.5), mild translate (0.1) and scale (0.5). No vertical flip, "
        "no mixup/copy-paste/shear/rotation &mdash; aggressive augmentation caused training "
        "divergence on this tiny dataset.", S["Body"]))
    A(Paragraph(
        "<b>Semi-supervised pseudo-labeling.</b> The original SSL pipeline sampled frames at "
        "fixed stride from each video and kept frames where the single most confident box "
        "scored &ge;0.7 and all retained boxes scored &ge;0.5. For ID videos: 702/720 sampled "
        "frames (97.5%) and 1,690 boxes at mean confidence 0.876. For OOD videos: 667/713 "
        "frames (93.5%) and 2,147 boxes at mean confidence 0.732 &mdash; <b>71.7% of the OOD "
        "pseudo-label boxes were classified as Needle_driver</b>, revealing a strong class skew. "
        "Manual inspection additionally showed recurring false-positive Needle_driver "
        "detections on blood-stained gauze and background regions, suggesting that systematic "
        "teacher errors were being reinforced during self-training. Combined with only 61 real "
        "labeled images, the final OOD training set contained 1,369 pseudo-labeled images, "
        "creating a 22:1 ratio of pseudo to real data.", S["Body"]))

    A(Paragraph("2.3 Regularization", S["H3s"]))
    A(Paragraph(
        "The model relies on: (i) <b>weight decay</b> 5e-4 under AdamW; (ii) "
        "<b>early stopping</b> on validation fitness with patience 50 epochs for the "
        "baseline (tightened in later rounds); (iii) the moderate <b>data augmentation</b> "
        "described in &sect;2.2 which acts as an implicit regularizer against the tiny "
        "labeled set; (iv) the pretrained COCO backbone itself as a strong prior. "
        "<b>Dropout is not used</b> (<code>dropout=0.0</code>).", S["Body"]))

    A(Paragraph("2.4 Hyperparameter tuning", S["H3s"]))
    A(Paragraph(
        "A systematic supervised sweep explored: <b>input resolution</b> "
        "(640/960/1280), <b>model scale</b> (YOLO11s/m), <b>learning rate</b> "
        "(0.001/0.003/0.01), and <b>augmentation strength</b> (moderate/weak). "
        "Key findings: (i) LR 0.003 and 0.01 caused training collapse or NaN in every "
        "tested configuration; only LR 0.001 was stable, consistent with the very small "
        "dataset (&asymp;4 steps/epoch). (ii) YOLO11m did not improve over YOLO11s. "
        "(iii) Higher resolution improved ID validation mAP: the supervised 960 candidate "
        "achieved <b>clean mAP50-95 = 0.854</b> (the highest in the campaign). However, "
        "manual inspection of its OOD video showed <b>recurring gauze false positives</b>. "
        "This illustrates an important finding: <i>higher ID validation mAP did not guarantee "
        "better OOD generalization.</i>", S["Body"]))
    A(Paragraph(
        "Beyond supervised tuning, we tested multiple <b>pseudo-label filtering policies</b>: "
        "confidence-only thresholds (P1&ndash;P4), bounding-box sanity constraints "
        "(derived from the real labeled box distributions), per-frame maximum detection "
        "caps, class-specific thresholds/caps, and temporal persistence (requiring "
        "predictions to appear consistently across neighboring frames). The most notable "
        "experiment (P4) retained only 69 frames / 168 boxes at mean confidence 0.938 "
        "&mdash; yet the retrained model still produced substantial OOD false positives, "
        "demonstrating that <i>high model confidence is not equivalent to correct pseudo-labels</i>. "
        "We also re-attempted OOD pseudo-labeling under strict conditions (box 0.85 / frame "
        "0.92), which accepted <b>zero</b> OOD frames, showing the teacher was not "
        "sufficiently reliable on OOD data. A relaxed combined approach retained 123 OOD "
        "frames but made the model too conservative (21% empty OOD frames). "
        "Full policy statistics are documented in the repository "
        "(<code>experiments/pseudo_label_leaderboard.md</code>).", S["Body"]))

    A(Paragraph("2.5 Train + valid loss graph", S["H3s"]))
    A(Paragraph(
        "Fig. 4 shows the training and validation loss (sum of box, classification and DFL "
        "loss terms) as a function of the training epoch. The x-axis is the epoch number; "
        "the y-axis is the summed loss value (clipped at 8 for readability).", S["Body"]))
    img(root / "reports/curves/loss_curves.png", CONTENT_W,
        "Fig 4. Training (blue) and validation (red) loss vs. epoch for the Baseline "
        "and the two original SSL rounds. The Baseline shows an early instability period "
        "(epochs 5\u201331) where the classification loss spikes, then recovers. The SSL rounds "
        "(warm-started) do not show this instability.",
        S, story, max_h=4.8 * cm)
    A(Paragraph(
        "In the Baseline, loss decreases normally for the first 4 epochs, then the "
        "classification loss spikes between roughly epochs 5&ndash;31. It then recovers on "
        "its own: from epoch &asymp;32 both losses decrease steadily. This early instability "
        "is a property of fine-tuning on this specific very small dataset during the LR warmup "
        "phase. From epoch &asymp;50 onward, both losses decrease smoothly with no widening "
        "train/val gap (no overfitting). The ssl_id and ssl_ood rounds, starting from "
        "already fine-tuned checkpoints, do not show this early instability.", S["Body"]))

    A(Paragraph("2.6 Train + valid mAP graphs", S["H3s"]))
    A(Paragraph(
        "Ultralytics evaluates mAP only on the validation split at each epoch end. "
        "The x-axis is the training epoch; the y-axis is the mAP value on the fixed 10-image "
        "validation set.", S["Body"]))
    img(root / "reports/curves/map_curves.png", CONTENT_W,
        "Fig 5. Validation mAP@50 (green) and mAP@50-95 (purple) vs. epoch for the Baseline "
        "and the two original SSL rounds. Note: these are official (original) validation "
        "metrics, not the cleaned validation set.",
        S, story, max_h=4.8 * cm)

    # Results table
    mrows = [["Model", "Epoch", "Imgs", "mAP50-95\norig", "mAP50-95\nclean", "OOD behavior"]]
    mrows.append(["Baseline (final)", "128", "61", "0.754", "0.809", "Best balance"])
    mrows.append(["ssl_id", "40", "763", "0.782", "0.837", "87% >2 det"])
    mrows.append(["ssl_ood", "4", "1430", "0.740", "0.792", "90% >2 det"])
    mrows.append(["Sup. 960", "103", "61", "0.797", "0.854", "Gauze FP"])
    mrows.append(["Sup. 1280", "76", "61", "0.751", "0.805", "Flicker"])
    mrows.append(["ID P4", "11", "130", "0.749", "0.800", "81% >2 det"])
    mrows.append(["ID temporal", "18", "627", "0.767", "0.820", "16% empty"])
    mrows.append(["ID+OOD comb.", "24", "750", "0.721", "0.775", "21% empty"])
    A(kv_table(mrows,
               [2.7*cm, 1.2*cm, 1.1*cm, 1.8*cm, 1.8*cm, CONTENT_W - 8.6*cm], S, fontsize=7.0))
    A(Spacer(1, 2))
    A(Paragraph(
        "<b>Table 1.</b> Key experiment candidates with official (<code>val_original</code>) "
        "and cleaned (<code>val_clean</code>) validation mAP50-95 and OOD video behavior at "
        "comparable inference confidence. &ldquo;>2 det&rdquo; = percentage of OOD frames with "
        "more than two detections (excessive for a two-hand scene). "
        "The full experiment leaderboard is in the repository.", S["Cap"]))

    A(PageBreak())

    # ================= 3. Discussion and Conclusions =================
    A(Paragraph("3. Discussion and Conclusions", S["H"]))
    A(Paragraph(
        "Since the OOD video was filmed with different camera/lighting and has no ground-truth "
        "labels, OOD performance is assessed using automated diagnostics (detection counts, "
        "temporal stability metrics) and manual visual inspection.", S["Body"]))

    # --- SSL failure ---
    A(Paragraph("<b>Original SSL failure.</b>", S["H3s"]))
    A(Paragraph(
        "The original OOD self-training pipeline was intended to improve OOD generalization. "
        "Instead, naive pseudo-label self-training degraded OOD behavior in our experiments. "
        "The OOD pseudo-label boxes were 71.7% classified as <code>Needle_driver</code>, "
        "revealing a strong class skew. Furthermore, 1,369 pseudo-labeled images overwhelmed "
        "61 real labeled images. The old ssl_ood model selected epoch 4, achieved "
        "mAP50-95 = 0.740, and on the long OOD video produced "
        "<b>89.6%</b> of frames with more than two detections (Baseline: 6.6%). The "
        "Baseline outperformed the old ssl_ood model on cleaned validation mAP and, more "
        "importantly, showed substantially better OOD video behavior. "
        "This is consistent with <b>confirmation bias</b> in self-training: the teacher "
        "entered the OOD environment with reduced reliability, generated systematic false "
        "predictions, these were accepted as pseudo-ground-truth, and retraining reinforced "
        "the errors, producing a model even more confident in the same incorrect patterns.",
        S["Body"]))

    # --- Pseudo-label policy findings ---
    A(Paragraph("<b>Pseudo-label policy findings.</b>", S["H3s"]))
    A(Paragraph(
        "To determine whether better pseudo-label filtering could rescue SSL, we tested "
        "multiple policies. (i) <b>Confidence alone is insufficient</b>: policy P4 retained "
        "only 69 frames at mean confidence 0.938, yet the retrained model still produced "
        "81.3% of OOD frames with &gt;2 detections. (ii) <b>Temporal consistency helps</b>: "
        "the P3+sanity+temporal policy, which additionally enforced max 2 detections per frame "
        "and required predictions to persist across neighboring frames, produced a much more "
        "stable model (0.5% of frames with &gt;2 detections, only 18 class switches). However, "
        "it came at a <b>recall cost</b>: 16% of OOD frames were empty, indicating missed "
        "hands. (iii) <b>Strict OOD thresholds rejected everything</b>: at box 0.85/frame "
        "0.92, zero of 380 OOD frames were accepted. A relaxed combined approach retained "
        "123 OOD frames but the retrained model became too conservative (21% empty frames, "
        "mAP50-95 = 0.721).", S["Body"]))

    # --- Finalist selection ---
    A(Paragraph("<b>Finalist comparison and final model selection.</b>", S["H3s"]))
    A(Paragraph(
        "The full experiment campaign was filtered using validation metrics, training-failure "
        "checks, pseudo-label statistics, OOD diagnostics and qualitative preview segments. "
        "Three complementary finalists survived:", S["Body"]))

    frows = [["Finalist", "det/fr", ">2 %", "empty %", "1-frame\ntracks", "class\nswitches"]]
    frows.append(["Baseline / 0.60", "1.70", "6.6", "5.9", "145", "69"])
    frows.append(["ID temporal / 0.40", "1.38", "0.5", "16.0", "92", "18"])
    frows.append(["Sup. 1280 / 0.50", "1.49", "5.1", "10.5", "288", "82"])
    A(kv_table(frows,
               [3.2*cm, 1.4*cm, 1.2*cm, 1.5*cm, 1.6*cm, 1.6*cm], S, fontsize=7.0))
    A(Spacer(1, 2))
    A(Paragraph(
        "<b>Table 2.</b> Automated diagnostics on the full 180-second OOD video (5,395 frames) "
        "at each finalist's selected inference threshold. Note: <code>conf</code> is an "
        "inference threshold, not a training parameter &mdash; all models use their best "
        "training checkpoint.", S["Cap"]))

    A(Paragraph(
        "Manual visual review of identical OOD segments and full 180-second videos focused on "
        "four criteria: (i) <b>false positives</b> &mdash; gauze/background marked as tools; "
        "(ii) <b>missed detections</b> &mdash; real hands left unmarked; "
        "(iii) <b>class stability</b> &mdash; same hand rapidly switching labels without a "
        "real tool change; (iv) <b>temporal stability</b> &mdash; boxes persisting smoothly "
        "vs. flickering. The result:", S["Body"]))
    A(Paragraph(
        "<b>Baseline</b> (conf=0.60): best overall visual behavior. It reliably detected "
        "both hands while avoiding the recurring gauze false-positive failure. Not perfect, "
        "but the most consistent. "
        "<b>ID temporal</b> (conf=0.40): extremely stable and few false positives, but "
        "sometimes failed to mark a real visible hand &mdash; the precision improvement was "
        "not worth the recall loss. "
        "<b>Supervised 1280</b> (conf=0.50): reasonable coverage but visibly more temporal "
        "flicker (288 one-frame tracks vs. 145 for Baseline), with occasional incorrect class "
        "labels on held tools.", S["Body"]))
    A(Paragraph(
        "&rarr; <b>The supervised Baseline at inference confidence 0.60 was selected as the "
        "final model.</b> Its training path is the simplest: COCO-pretrained YOLO11s &rarr; "
        "one supervised fine-tuning stage on the 61 labeled images &rarr; best checkpoint "
        "(epoch 128) &rarr; OOD inference at conf=0.60. The final model uses <i>no</i> "
        "pseudo-labeled data.", S["Body"]))

    # --- Conclusions/lessons ---
    A(Paragraph("<b>Key lessons.</b>", S["H3s"]))
    A(Paragraph(
        "(1) <b>Pseudo-label quality matters more than quantity.</b> Adding 1,369 pseudo images "
        "to 61 real images did not improve performance &mdash; systematic errors were amplified. "
        "(2) <b>Self-training can create confirmation bias.</b> The teacher's repeated "
        "misidentification of gauze as Needle_driver was reinforced during retraining. "
        "(3) <b>Confidence is not correctness.</b> P4 retained pseudo-labels at mean confidence "
        "0.938 yet still produced substantial false positives. "
        "(4) <b>Temporal consistency is useful but insufficient.</b> Requiring predictions to "
        "persist across neighboring frames substantially reduced false positives, but aggressive "
        "filtering also reduced recall. "
        "(5) <b>ID validation mAP does not fully predict OOD quality.</b> Supervised 960 "
        "achieved the strongest clean mAP50-95 (0.854) but was rejected because manual OOD "
        "inspection showed recurring gauze false positives. The 10-image ID validation set "
        "cannot fully measure generalization to the different OOD video. "
        "(6) <b>Small validation sets are sensitive to annotation noise.</b> Removing two "
        "confirmed erroneous boxes changed Baseline mAP50-95 from 0.754 to 0.809 &mdash; "
        "a &asymp;7% swing from only two labels in a 22-box evaluation set. "
        "(7) <b>The simplest model can win.</b> The final winner was not the largest model, "
        "not the highest-resolution variant, and not the deepest SSL pipeline. The experiment "
        "demonstrates why additional data and model complexity must be validated rather than "
        "assumed to improve performance.", S["Body"]))

    A(Paragraph(
        "<b>Conclusion.</b> The supervised Baseline was selected as the final detector "
        "because it produced the best overall OOD video behavior, despite not achieving "
        "the highest validation mAP. Naive pseudo-label self-training introduced "
        "confirmation bias and severe false positives. Stricter confidence thresholds "
        "alone were insufficient, while sanity and temporal filtering improved prediction "
        "stability but reduced recall. The experiments therefore show that higher "
        "validation metrics and larger pseudo-labeled datasets did not necessarily "
        "translate into better OOD generalization.", S["Body"]))

    doc = SimpleDocTemplate(str(root / args.out), pagesize=A4,
                            leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=MARGIN, bottomMargin=MARGIN,
                            title="HW1 Surgical Tool Detection")
    doc.build(story)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
