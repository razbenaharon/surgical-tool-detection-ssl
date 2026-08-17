"""Generate the HW1 PDF report (<= 5 pages) with reportlab.

Revised version — student-level first-person narrative matching the requested style.
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
    A(Paragraph("Authors: Raz Ben-Aharon and Shalev Manassen",
               S["Sub"]))

    # ================= 1. Exploratory Data Analysis =================
    A(Paragraph("1. Exploratory Data Analysis", S["H"]))
    A(Paragraph(
        "Our labeled set contained only <b>%d training</b> and <b>%d validation</b> images "
        "(&lt;100 total), all in native 4K. Each frame comes from a surgery video, "
        "and our objective was to correctly identify bounding boxes encompassing a gloved hand "
        "along with what it holds: <b>Empty</b>, <b>Tweezers</b>, or <b>Needle_driver</b>."
        % (tr.get("n_images", 61), va.get("n_images", 10)), S["Body"]))

    A(Paragraph("1.1 Visualization of Some Images", S["H3s"]))
    A(Paragraph(
        "Fig. 1 shows a random sample of our labeled training frames with ground-truth boxes "
        "(green = Empty, blue = Tweezers, red = Needle_driver).",
        S["Body"]))
    img(root / "reports/eda/samples_grid.png", CONTENT_W * 0.86,
        "Fig 1. Nine labeled frames with ground-truth boxes overlaid.", S, story)

    A(Paragraph("1.2 Insights from our Initial Analysis", S["H3s"]))
    A(Paragraph(
        "Before training, we manually inspected the frames and videos. We quickly noticed "
        "several challenges: the scene is visually difficult due to specular highlights, blood, "
        "and surgical drapes that blend in with the gloves. Furthermore, almost every frame "
        "contains exactly one or two boxes (Fig. 2, right). Crucially, the OOD video was filmed "
        "with completely different lighting and camera angles, meaning our model would need "
        "strong generalization capabilities, not just high ID accuracy.",
        S["Body"]))

    A(Paragraph("1.3 Data Distribution", S["H3s"]))
    rows = [["Split", "Images", "Boxes", "Empty", "Tweezers", "Needle_driver", "Boxes/img"]]
    for name, d in [("train", tr), ("val", va)]:
        rows.append([name, d.get("n_images", "-"), d.get("n_boxes", "-"),
                     cc(d, "Empty"), cc(d, "Tweezers"), cc(d, "Needle_driver"),
                     f"{d.get('boxes_per_image_mean', 0):.2f}"])
    A(kv_table(rows, [1.7 * cm] + [1.9 * cm] * 4 + [2.1 * cm] + [2.0 * cm], S))
    A(Spacer(1, 3))
    A(Paragraph(
        "<b>Empty is the clear minority class</b> in our labeled set (only &asymp;19% of "
        "train boxes), while Tweezers and Needle_driver are roughly balanced. This class "
        "imbalance strongly motivated our pseudo-labeling strategy, as frames with empty "
        "hands are common in the raw video but scarce in our labeled sample.", S["Body"]))
    
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
        "Fig 2. Left: Bounding-box counts showing Empty as the minority class. "
        "Right: Spatial density of box centers, matching the top-down camera setup.", S["Cap"]))
    A(KeepTogether(figs2))

    img(root / "reports/eda/box_size_distribution.png", CONTENT_W * 0.92,
        "Fig 3. Left: Distribution of relative box sizes. Right: Box width vs. height.", S, story,
        max_h=4.2 * cm)

    A(PageBreak())

    # ================= 2. Experiments =================
    A(Paragraph("2. Experiments", S["H"]))
    A(Paragraph(
        "Our goal was to maximize out-of-distribution (OOD) generalization under a very limited "
        "budget of 61 labeled training images. We treated the task as a semi-supervised learning "
        "challenge. Because every simulation (or in our case, training run) can be misleading, "
        "we decided not to rely solely on in-distribution metrics. We evaluated every model "
        "choice by its actual qualitative behavior on the OOD video. Furthermore, we treated "
        "<b>confirmation bias</b> as a significant risk &mdash; naive pseudo-labeling can turn "
        "a minor mistake into a systemic failure. Therefore, simply having more pseudo-labeled "
        "data is a misleading metric; what matters most is the <i>quality</i> of the labels.", S["Body"]))

    A(Paragraph("2.1 Data Loading & Validation Audit", S["H3s"]))
    A(Paragraph(
        "To start, we had to establish a strong supervised Baseline and audit our data. We manually "
        "inspected our validation images and found two erroneous bounding box annotations in "
        "<code>ff8c22da-output_0182.png</code>. Because these artifacts represented &asymp;9% of "
        "our tiny validation set, we computed 'cleaned' metrics (val_clean) alongside the official "
        "metrics (val_original) to ensure our hyperparameter choices were based on reality.", S["Body"]))

    A(Paragraph("2.2 Hyperparameter Tuning", S["H3s"]))
    A(Paragraph(
        "We ran a comprehensive hyperparameter sweep, testing different resolutions (640/960/1280), "
        "learning rates (0.001/0.003/0.01), and model sizes (YOLO11s/YOLO11m) to find the most "
        "robust baseline. We found that higher learning rates (0.003+) caused training collapse, "
        "and larger models didn't help. Interestingly, our <b>Supervised 960</b> model achieved "
        "the highest cleaned validation mAP (0.854), but we ultimately rejected it because manual "
        "OOD inspection revealed recurring false positives on surgical gauze.", S["Body"]))

    A(Paragraph("2.3 Semi-Supervised Learning & Regularization", S["H3s"]))
    A(Paragraph(
        "After establishing our baseline, we began building our semi-supervised models. "
        "Our original SSL pipeline sampled frames and kept those where the model was highly "
        "confident (&ge;0.7). However, we quickly saw that systematic teacher errors (such as "
        "misclassifying blood-stained gauze as <code>Needle_driver</code>) were being reinforced. "
        "To avoid wasting our model's capacity on bad labels, we experimented with <b>sanity "
        "constraints</b> and <b>temporal filtering</b>, requiring predictions to persist across "
        "neighboring frames. While this prevented false positives, it severely hurt our recall. "
        "To regularize, we relied on weight decay (5e-4), early stopping, and moderate data "
        "augmentation (mosaic, color jitter).", S["Body"]))

    A(Paragraph("2.4 Train + Valid Loss Graph", S["H3s"]))
    img(root / "reports/curves/loss_curves.png", CONTENT_W,
        "Fig 4. Training (blue) and validation (red) loss vs. epoch. Our Baseline shows an early "
        "instability period during LR warmup, then recovers smoothly without overfitting.",
        S, story, max_h=4.8 * cm)

    A(Paragraph("2.5 Train + Valid mAP Graphs", S["H3s"]))
    img(root / "reports/curves/map_curves.png", CONTENT_W,
        "Fig 5. Validation mAP@50 (green) and mAP@50-95 (purple) vs. epoch.",
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
        "<b>Table 1.</b> Key candidates with official and cleaned validation mAP. "
        "Notice that our chosen Baseline doesn't have the highest mAP, but rather the best OOD behavior.", S["Cap"]))

    A(PageBreak())

    # ================= 3. Discussion and Conclusions =================
    A(Paragraph("3. Discussion and Conclusions", S["H"]))
    
    A(Paragraph("<b>The Danger of Confirmation Bias</b>", S["H3s"]))
    A(Paragraph(
        "Our initial OOD self-training pipeline degraded rather than improved OOD generalization. "
        "We discovered that 71.7% of the OOD pseudo-label boxes were classified as <code>Needle_driver</code>, "
        "revealing a massive class skew. Manual inspection showed recurring false-positive "
        "<code>Needle_driver</code> detections on blood-stained gauze. This is a classic case of "
        "confirmation bias: our teacher model made mistakes on the OOD video, and by blindly "
        "training on those mistakes, the model became even more confident in its incorrect patterns. "
        "We learned that <b>confidence alone is insufficient</b> (our P4 policy retained predictions "
        "at 0.938 confidence but still failed dramatically).", S["Body"]))

    A(Paragraph("<b>Finalist Comparison</b>", S["H3s"]))
    A(Paragraph(
        "To ensure our comparisons were reliable, we evaluated our top candidates using automated "
        "OOD video diagnostics, tracking metrics like frames with &gt;2 detections and single-frame "
        "flickers. We then manually evaluated the finalists on the full 180-second OOD video.", S["Body"]))

    frows = [["Finalist", "det/fr", ">2 %", "empty %", "1-frame\ntracks", "class\nswitches"]]
    frows.append(["Baseline / 0.60", "1.70", "6.6", "5.9", "145", "69"])
    frows.append(["ID temporal / 0.40", "1.38", "0.5", "16.0", "92", "18"])
    frows.append(["Sup. 1280 / 0.50", "1.49", "5.1", "10.5", "288", "82"])
    A(kv_table(frows,
               [3.2*cm, 1.4*cm, 1.2*cm, 1.5*cm, 1.6*cm, 1.6*cm], S, fontsize=7.0))
    A(Spacer(1, 2))
    A(Paragraph("<b>Table 2.</b> Diagnostics on the full 180s OOD video for our 3 finalists.", S["Cap"]))

    A(Paragraph(
        "Our <b>ID temporal</b> model was extremely stable (few false positives) but failed to "
        "detect real visible hands too often. Our <b>Supervised 1280</b> model had reasonable "
        "coverage but suffered from excessive temporal flicker. The <b>Baseline</b> model "
        "offered the best overall balance of precision and recall on the unseen domain.", S["Body"]))

    A(Paragraph("<b>Conclusion</b>", S["H3s"]))
    A(Paragraph(
        "Ultimately, we selected our supervised Baseline (at confidence 0.60) as the final "
        "detector because it produced the best overall OOD video behavior, despite not achieving "
        "the highest validation mAP. Naive pseudo-label self-training introduced confirmation "
        "bias and severe false positives. Stricter confidence thresholds alone were insufficient, "
        "while sanity and temporal filtering improved prediction stability but reduced recall. "
        "Our experiments demonstrate that higher validation metrics and larger pseudo-labeled "
        "datasets do not necessarily translate into better real-world generalization. Sometimes, "
        "a well-regularized baseline is the most robust choice.", S["Body"]))

    doc = SimpleDocTemplate(str(root / args.out), pagesize=A4,
                            leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=MARGIN, bottomMargin=MARGIN,
                            title="HW1 Surgical Tool Detection")
    doc.build(story)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
