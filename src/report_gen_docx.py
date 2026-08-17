"""Generate the HW1 DOCX report from the same content as report_gen.py.

Usage:
    python src/report_gen_docx.py --root . --out reports/HW1_report_final.docx
"""
import argparse
import json
from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT


def add_figure(doc, path, caption, width_inches=5.5):
    p = Path(path)
    if not p.exists():
        doc.add_paragraph(f"[missing figure: {path}]")
        return
    doc.add_picture(str(p), width=Inches(width_inches))
    last_paragraph = doc.paragraphs[-1]
    last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap_para = doc.add_paragraph(caption)
    cap_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in cap_para.runs:
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor(128, 128, 128)
        run.font.italic = True


def add_table(doc, headers, rows, col_widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Light Grid Accent 1'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    # Header
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        for p in cell.paragraphs:
            for r in p.runs:
                r.font.bold = True
                r.font.size = Pt(7.5)
    # Data
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = table.rows[ri + 1].cells[ci]
            cell.text = str(val)
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(7.5)
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--metrics", default="reports/metrics.json")
    ap.add_argument("--out", default="reports/HW1_report_final.docx")
    args = ap.parse_args()
    root = Path(args.root)

    M = json.loads(Path(args.metrics).read_text()) if Path(args.metrics).exists() else {}
    eda = {}
    eda_path = root / "reports/eda/dataset_summary.json"
    if eda_path.exists():
        eda = json.loads(eda_path.read_text())

    doc = Document()

    # Set narrow margins
    for section in doc.sections:
        section.top_margin = Cm(1.3)
        section.bottom_margin = Cm(1.3)
        section.left_margin = Cm(1.3)
        section.right_margin = Cm(1.3)

    # Set default font
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Calibri'
    font.size = Pt(9)

    tr = eda.get("train", {})
    va = eda.get("val", {})

    # Title
    title = doc.add_heading('Surgical Tool Detection with Semi-Supervised Learning', level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle = doc.add_paragraph('Computer Vision \u2014 Surgical Applications, HW1')
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    authors = doc.add_paragraph('Authors: Raz Ben-Aharon and Shalev Manassen')
    authors.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # ===== 1. EDA =====
    doc.add_heading('1. Exploratory Data Analysis', level=1)
    doc.add_paragraph(
        f'Our labeled set contained only {tr.get("n_images", 61)} training and '
        f'{va.get("n_images", 10)} validation images (<100 total), all in native 4K. '
        f'Each frame comes from a surgery video, and our objective was to correctly identify '
        f'bounding boxes encompassing a gloved hand along with what it holds: '
        f'Empty, Tweezers, or Needle_driver.')

    doc.add_heading('1.1 Visualization of Some Images', level=2)
    doc.add_paragraph(
        'Fig. 1 shows a random sample of our labeled training frames with ground-truth boxes '
        '(green = Empty, blue = Tweezers, red = Needle_driver).')
    add_figure(doc, root / "reports/eda/samples_grid.png",
               "Fig 1. Nine labeled frames with ground-truth boxes overlaid.", 5.0)

    doc.add_heading('1.2 Insights from our Initial Analysis', level=2)
    doc.add_paragraph(
        'Before training, we manually inspected the frames and videos. We quickly noticed '
        'several challenges: the scene is visually difficult due to specular highlights, blood, '
        'and surgical drapes that blend in with the gloves. Furthermore, almost every frame '
        'contains exactly one or two boxes (Fig. 2, right). Crucially, the OOD video was filmed '
        'with completely different lighting and camera angles, meaning our model would need '
        'strong generalization capabilities, not just high ID accuracy.')

    doc.add_heading('1.3 Data Distribution', level=2)
    add_table(doc,
              ["Split", "Images", "Boxes", "Empty", "Tweezers", "Needle_driver", "Boxes/img"],
              [["train", tr.get("n_images", 61), tr.get("n_boxes", 135),
                tr.get("class_counts", {}).get("Empty", 26),
                tr.get("class_counts", {}).get("Tweezers", 54),
                tr.get("class_counts", {}).get("Needle_driver", 55),
                f'{tr.get("boxes_per_image_mean", 2.21):.2f}'],
               ["val", va.get("n_images", 10), va.get("n_boxes", 22),
                va.get("class_counts", {}).get("Empty", 2),
                va.get("class_counts", {}).get("Tweezers", 10),
                va.get("class_counts", {}).get("Needle_driver", 10),
                f'{va.get("boxes_per_image_mean", 2.2):.2f}']])
    doc.add_paragraph(
        'Empty is the clear minority class in our labeled set (only \u224819% of '
        'train boxes), while Tweezers and Needle_driver are roughly balanced. This class '
        'imbalance strongly motivated our pseudo-labeling strategy, as frames with empty '
        'hands are common in the raw video but scarce in our labeled sample.')
    add_figure(doc, root / "reports/eda/class_distribution.png",
               "Fig 2. Class distribution by split.", 3.0)
    add_figure(doc, root / "reports/eda/box_size_distribution.png",
               "Fig 3. Box size distribution and width vs. height.", 5.0)

    # ===== 2. Experiments =====
    doc.add_page_break()
    doc.add_heading('2. Experiments', level=1)
    doc.add_paragraph(
        'Our goal was to maximize out-of-distribution (OOD) generalization under a very limited '
        'budget of 61 labeled training images. We treated the task as a semi-supervised learning '
        'challenge. Because every simulation (or in our case, training run) can be misleading, '
        'we decided not to rely solely on in-distribution metrics. We evaluated every model '
        'choice by its actual qualitative behavior on the OOD video. Furthermore, we treated '
        'confirmation bias as a significant risk \u2014 naive pseudo-labeling can turn '
        'a minor mistake into a systemic failure. Therefore, simply having more pseudo-labeled '
        'data is a misleading metric; what matters most is the quality of the labels.')

    doc.add_heading('2.1 Data Loading & Validation Audit', level=2)
    doc.add_paragraph(
        'To start, we had to establish a strong supervised Baseline and audit our data. We manually '
        'inspected our validation images and found two erroneous bounding box annotations in '
        'ff8c22da-output_0182.png. Because these artifacts represented \u22489% of '
        'our tiny validation set, we computed \'cleaned\' metrics (val_clean) alongside the official '
        'metrics (val_original) to ensure our hyperparameter choices were based on reality.')

    doc.add_heading('2.2 Hyperparameter Tuning', level=2)
    doc.add_paragraph(
        'We ran a comprehensive hyperparameter sweep, testing different resolutions (640/960/1280), '
        'learning rates (0.001/0.003/0.01), and model sizes (YOLO11s/YOLO11m) to find the most '
        'robust baseline. We found that higher learning rates (0.003+) caused training collapse, '
        'and larger models didn\'t help. Interestingly, our Supervised 960 model achieved '
        'the highest cleaned validation mAP (0.854), but we ultimately rejected it because manual '
        'OOD inspection revealed recurring false positives on surgical gauze.')

    doc.add_heading('2.3 Semi-Supervised Learning & Regularization', level=2)
    doc.add_paragraph(
        'After establishing our baseline, we began building our semi-supervised models. '
        'Our original SSL pipeline sampled frames and kept those where the model was highly '
        'confident (\u22650.7). However, we quickly saw that systematic teacher errors (such as '
        'misclassifying blood-stained gauze as Needle_driver) were being reinforced. '
        'To avoid wasting our model\'s capacity on bad labels, we experimented with sanity '
        'constraints and temporal filtering, requiring predictions to persist across '
        'neighboring frames. While this prevented false positives, it severely hurt our recall. '
        'To regularize, we relied on weight decay (5e-4), early stopping, and moderate data '
        'augmentation (mosaic, color jitter).')

    doc.add_heading('2.4 Train + Valid Loss Graph', level=2)
    add_figure(doc, root / "reports/curves/loss_curves.png",
               "Fig 4. Training (blue) and validation (red) loss vs. epoch. Our Baseline shows an early "
               "instability period during LR warmup, then recovers smoothly without overfitting.", 5.5)

    doc.add_heading('2.5 Train + Valid mAP Graphs', level=2)
    add_figure(doc, root / "reports/curves/map_curves.png",
               "Fig 5. Validation mAP@50 and mAP@50-95 vs. epoch.", 5.5)

    add_table(doc,
              ["Model", "Epoch", "Imgs", "mAP50-95\norig", "mAP50-95\nclean", "OOD behavior"],
              [["Baseline (final)", "128", "61", "0.754", "0.809", "Best balance"],
               ["ssl_id", "40", "763", "0.782", "0.837", "87% >2 det"],
               ["ssl_ood", "4", "1430", "0.740", "0.792", "90% >2 det"],
               ["Sup. 960", "103", "61", "0.797", "0.854", "Gauze FP"],
               ["Sup. 1280", "76", "61", "0.751", "0.805", "Flicker"],
               ["ID P4", "11", "130", "0.749", "0.800", "81% >2 det"],
               ["ID temporal", "18", "627", "0.767", "0.820", "16% empty"],
               ["ID+OOD comb.", "24", "750", "0.721", "0.775", "21% empty"]])
    cap = doc.add_paragraph(
        'Table 1. Key candidates with official and cleaned validation mAP. '
        'Notice that our chosen Baseline doesn\'t have the highest mAP, but rather the best OOD behavior.')
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in cap.runs:
        run.font.size = Pt(8)
        run.font.italic = True

    # ===== 3. Discussion and Conclusions =====
    doc.add_page_break()
    doc.add_heading('3. Discussion and Conclusions', level=1)

    doc.add_heading('The Danger of Confirmation Bias', level=2)
    doc.add_paragraph(
        'Our initial OOD self-training pipeline degraded rather than improved OOD generalization. '
        'We discovered that 71.7% of the OOD pseudo-label boxes were classified as Needle_driver, '
        'revealing a massive class skew. Manual inspection showed recurring false-positive '
        'Needle_driver detections on blood-stained gauze. This is a classic case of '
        'confirmation bias: our teacher model made mistakes on the OOD video, and by blindly '
        'training on those mistakes, the model became even more confident in its incorrect patterns. '
        'We learned that confidence alone is insufficient (our P4 policy retained predictions '
        'at 0.938 confidence but still failed dramatically).')

    doc.add_heading('Finalist Comparison', level=2)
    doc.add_paragraph(
        'To ensure our comparisons were reliable, we evaluated our top candidates using automated '
        'OOD video diagnostics, tracking metrics like frames with >2 detections and single-frame '
        'flickers. We then manually evaluated the finalists on the full 180-second OOD video.')
    add_table(doc,
              ["Finalist", "det/fr", ">2 %", "empty %", "1-frame tracks", "switches"],
              [["Baseline / 0.60", "1.70", "6.6", "5.9", "145", "69"],
               ["ID temporal / 0.40", "1.38", "0.5", "16.0", "92", "18"],
               ["Sup. 1280 / 0.50", "1.49", "5.1", "10.5", "288", "82"]])
    cap2 = doc.add_paragraph('Table 2. Diagnostics on the full 180s OOD video for our 3 finalists.')
    cap2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in cap2.runs:
        run.font.size = Pt(8)
        run.font.italic = True
    doc.add_paragraph(
        'Our ID temporal model was extremely stable (few false positives) but failed to '
        'detect real visible hands too often. Our Supervised 1280 model had reasonable '
        'coverage but suffered from excessive temporal flicker. The Baseline model '
        'offered the best overall balance of precision and recall on the unseen domain.')

    doc.add_heading('Conclusion', level=2)
    doc.add_paragraph(
        'Ultimately, we selected our supervised Baseline (at confidence 0.60) as the final '
        'detector because it produced the best overall OOD video behavior, despite not achieving '
        'the highest validation mAP. Naive pseudo-label self-training introduced confirmation '
        'bias and severe false positives. Stricter confidence thresholds alone were insufficient, '
        'while sanity and temporal filtering improved prediction stability but reduced recall. '
        'Our experiments demonstrate that higher validation metrics and larger pseudo-labeled '
        'datasets do not necessarily translate into better real-world generalization. Sometimes, '
        'a well-regularized baseline is the most robust choice.')

    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
