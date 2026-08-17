# HW1 final-experiment redo — pre-report handoff

No final winner is declared here. The `manual_visual_rating` column remains blank until the
comparison videos are reviewed. The report was not edited.

## Main audit findings

- The two tiny boxes in `ff8c22da-output_0182.txt` are annotation artifacts: row 2 is a
  14x14-pixel `Needle_driver` box on dark background and row 4 is a 10x19-pixel
  `Tweezers` box at a glove/background boundary. Only those rows were removed in
  `val_clean`; `val_original` is byte-identical to the official source.
- Removing the two artifacts materially raises recall and mAP. For the legacy baseline,
  mAP50-95 rises from 0.754 to 0.809, but an exact clean-validation replay still selects
  epoch 128, so the selected baseline epoch did not change.
- The legacy run logs show that `optimizer=auto` ignored the reported/requested
  `lr0=0.01` and used AdamW with lr 0.001429 and momentum 0.9.
- Legacy OOD pseudo-labels were 71.7% `Needle_driver`; 1,369 pseudo images overwhelmed
  61 real images. The old OOD branch selected epoch 4 and worsened both validation and
  qualitative OOD behavior.

## Serious candidate results

| candidate | best epoch | original mAP50-95 | clean P | clean R | clean mAP50-95 | short OOD result at conf=0.50 |
|---|---:|---:|---:|---:|---:|---|
| legacy baseline / exact supervised reference | 128 | 0.754 | 0.972 | 0.929 | 0.809 | 2.04 det/frame; 17.7% >2; 0.3% empty |
| legacy ssl_id | 40 | 0.782 | 0.978 | 0.926 | 0.837 | 3.40 det/frame; 87.0% >2 |
| legacy ssl_ood | 4 | 0.740 | 0.954 | 0.889 | 0.792 | 4.05 det/frame; 100% >2 |
| supervised 960/moderate | 103 | 0.797 | 0.981 | 0.963 | 0.854 | repeated gauze false positive; 63.3% >2 |
| supervised 1280/moderate | 76 | 0.751 | 0.978 | 0.960 | 0.805 | 1.71 det/frame; 5.0% >2; 6.0% empty |
| ID P4, real repeat x5 | 11 | 0.751 | 0.795 | 1.000 | 0.800 | failed: 2.74 det/frame; 81.3% >2 |
| ID P3 + sanity + temporal, real repeat x5 | 18 | 0.767 | 0.970 | 0.951 | 0.820 | 1.41 det/frame; 0% >2; 17.3% empty |
| ID + conservative OOD combined, real repeat x5 | 24 | 0.719 | 0.856 | 0.959 | 0.775 | rejected: 0.80 det/frame; 21.0% empty |

The 48-second identical long-video segments reinforce the short-video result: baseline
has 1.84 det/frame, 10.2% >2 and 2.6% empty; supervised-1280 has 1.59, 8.0% and 5.8%;
ID-temporal has 1.35, 0.1% and 16.0%. The legacy ssl_id and ssl_ood models have 76.6%
and 89.6% of frames with more than two detections, respectively.

On the complete 5,395-frame long OOD video at each finalist's tuned inference threshold,
baseline/0.60 has 1.70 det/frame, 6.6% >2 and 5.9% empty; ID-temporal/0.40 has
1.38, 0.5% and 16.0%; supervised-1280/0.50 has 1.49, 5.1% and 10.5%. The 1280 model
also creates 288 one-frame tracks and 82 class switches, versus 145/69 for baseline and
92/18 for ID-temporal.

## Pseudo-label findings

- ID P1 exactly reconstructs the legacy result: 702/720 frames and 1,690 boxes.
- ID P4 is the smallest/highest-confidence set: 69 frames, 168 boxes, mean confidence
  0.938. Its retrained model nevertheless creates many OOD false positives, showing that
  confidence alone is insufficient.
- ID P3 + sanity + temporal keeps 566 frames and 1,020 boxes. It removes 207 boxes through
  max-two-per-frame, 82 through temporal persistence and one through the real-data-derived
  height bound.
- Strict OOD thresholds (box 0.85/frame 0.92) accept zero of 380 frames. The one trained
  relaxed combined branch keeps only 123 frames/129 boxes, including a `Needle_driver`
  cap of 120, but still reduces recall too much.

Full policy statistics and filter rejection counts are in
`experiments/pseudo_label_leaderboard.{csv,json,md}`. Real and pseudo before/after box
distribution plots are stored with each pseudo-label policy.

## Top three for manual review

1. **Legacy baseline at conf=0.60** — best recall/false-positive balance and the current
   recommendation, despite slightly lower clean validation mAP than ID-temporal.
2. **ID P3 + sanity + temporal at conf=0.40** — precision-first option: almost no >2
   detections or class switches, but visibly misses hands in transitions and one-hand frames.
3. **Supervised 1280/moderate at conf=0.50** — middle ground; fewer extras than baseline,
   but more empty frames, short tracks and occasional `Empty` classification on a held tool.

Watch the identical `*_longsegments_conf050.mp4` clips first for an apples-to-apples model
comparison. Then watch the `finalist_*_full.mp4` files at each model's recommended threshold.
The old ssl_id, old ssl_ood, failed ID-P4 and rejected new OOD branch remain in
`evaluation_videos/` for negative controls.

## Reproducible artifact index

- Candidate leaderboard: `experiments/leaderboard.csv`, `.json`, `.md`
- Pseudo-label leaderboard: `experiments/pseudo_label_leaderboard.csv`, `.json`, `.md`
- Validation audit: `reports/validation_audit/audit.json` and debug images
- Real-box distribution: `reports/box_distribution/`
- Validation metrics: `experiments/evaluations/<experiment>/validation_metrics.json`
- OOD diagnostics: `experiments/ood_diagnostics/*.json`
- Comparison videos: `evaluation_videos/`
- Candidate weights: `weights/candidates/`
