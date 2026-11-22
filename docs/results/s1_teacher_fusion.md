# S1-Fusion — Same-resolution complementary detector results

**Completed on CPU and scientifically reviewed.** F0 is original
AGE-YOLO (A0), F1 is AGE-YOLO with P2 (A1), and F2 is equal-weight
weighted boxes fusion. Both detectors supplied frozen, same-resolution
640 predictions from the same 230 validation images and 345 D1 objects.
The reference WBF algorithm agreed across all 230 images. Reconstructing
native TP flags with the pinned label transform gave exact parity with
both archived single-model AP exports. [Protocol and reproducibility](../experiments/s1_teacher_fusion.md)

## Overall and primary matched-budget results

| Metric | F0 · A0 | F1 · A1 | F2 · WBF |
|---|---|---|---|
| mAP50 | 70.01% | 71.00% | 73.08% |
| mAP50–95 | 35.09% | 35.21% | 36.85% |
| Nine-class mAP50 | 74.29% | 70.57% | 75.66% |
| Nine-class mAP50–95 | 37.71% | 35.97% | 38.72% |
| Native precision | 78.59% | 71.31% | 74.95% |
| Native recall | 64.30% | 68.38% | 73.50% |
| Fixed precision | 72.73% | 68.50% | 80.00% |
| Fixed recall | 69.57% | 64.93% | 67.25% |
| TP | 240 | 224 | 232 |
| FP | 90 | 103 | 58 |
| FN | 105 | 121 | 113 |

| FP/image cap | A0 TP / recall | A1 TP / recall | WBF TP / recall |
|---|---|---|---|
| 0.1 | 184 / 53.33% | 173 / 50.14% | 188 / 54.49% |
| 0.25 | 225 / 65.22% | 202 / 58.55% | 231 / 66.96% |
| 0.5 | 246 / 71.30% | 228 / 66.09% | 254 / 73.62% |
| 1.0 | 259 / 75.07% | 246 / 71.30% | 265 / 76.81% |

All fixed-point TP/FP counts and matched-budget TP counts reproduced
the independent exploratory values. At the primary cap of 0.50,
realized FP/image was 0.4957 for A0, 0.4957 for A1 and 0.4870 for WBF.
The primary recall gain over A1 is supported by paired validation-image
sampling. The interval against A0 spans zero. Fusion increases AP
point estimates against both models. The mAP50–95 interval excludes
zero against A1 and spans zero against A0. Fixed-point recall remains
below A0 despite substantially fewer false positives.

| Comparison | Metric | Δ (pp) | 95% interval (pp) | Valid draws |
|---|---|---|---|---|
| F2 − F0 | mAP50 | +3.07 | [−0.99, +5.71] | 4,325 |
| F2 − F0 | mAP50-95 | +1.76 | [−0.17, +3.44] | 4,325 |
| F2 − F0 | nine_class_AP50 | +1.37 | [−1.75, +4.52] | 5,000 |
| F2 − F0 | nine_class_AP50-95 | +1.00 | [−0.60, +2.93] | 5,000 |
| F2 − F0 | fixed_recall | −2.32 | [−5.36, +0.60] | 5,000 |
| F2 − F0 | q1_recall | −6.59 | [−13.51, +0.00] | 5,000 |
| F2 − F0 | short_le16_recall | −8.57 | [−26.32, +4.76] | 5,000 |
| F2 − F0 | primary_budget_recall | +2.32 | [−1.67, +5.31] | 5,000 |
| F2 − F1 | mAP50 | +2.08 | [−0.13, +6.02] | 4,325 |
| F2 − F1 | mAP50-95 | +1.64 | [+0.54, +3.42] | 4,325 |
| F2 − F1 | nine_class_AP50 | +5.08 | [+2.76, +7.42] | 5,000 |
| F2 − F1 | nine_class_AP50-95 | +2.75 | [+1.45, +4.00] | 5,000 |
| F2 − F1 | fixed_recall | +2.32 | [−0.54, +5.22] | 5,000 |
| F2 − F1 | q1_recall | +0.00 | [−5.88, +6.67] | 5,000 |
| F2 − F1 | short_le16_recall | +0.00 | [−11.43, +12.50] | 5,000 |
| F2 − F1 | primary_budget_recall | +7.54 | [+4.37, +10.80] | 5,000 |

Each comparison used the same 5,000 image draws with seed 42. The
0.50 cap was reapplied in each draw. Ten-class AP omitted 675 draws
without all classes. Other metrics retained 5,000. The recalculated
primary intervals were [+4.37, +10.80] pp versus A1 and
[−1.67, +5.31] pp versus A0. Their small endpoint differences from
the exploratory references leave the conclusions unchanged.
Inference outputs and training seeds were fixed.

## Per-class and paired object evidence

| Class | F0 AP50 / AP50–95 | F1 AP50 / AP50–95 | F2 AP50 / AP50–95 |
|---|---|---|---|
| punching_hole | 99.02% / 53.63% | 98.56% / 51.28% | 98.70% / 52.84% |
| welding_line | 92.45% / 39.26% | 87.32% / 35.84% | 92.64% / 39.79% |
| crescent_gap | 92.88% / 64.79% | 93.63% / 63.29% | 94.19% / 65.48% |
| water_spot | 84.63% / 45.07% | 83.55% / 43.97% | 85.23% / 46.91% |
| oil_spot | 72.20% / 36.66% | 69.64% / 35.68% | 73.25% / 37.71% |
| silk_spot | 59.98% / 24.71% | 58.26% / 26.45% | 61.78% / 27.93% |
| inclusion | 48.20% / 13.96% | 34.03% / 13.48% | 49.48% / 15.96% |
| rolled_pit | 31.46% / 11.44% | 74.85% / 28.32% | 49.88% / 20.01% |
| crease | 33.26% / 15.90% | 22.94% / 14.01% | 35.03% / 17.03% |
| waist_folding | 86.00% / 45.43% | 87.23% / 39.74% | 90.61% / 44.81% |

| Fixed-point pair | Retained | Recovered | Lost | Missed by both |
|---|---|---|---|---|
| F2 − F0 | 221 | 11 | 19 | 94 |
| F2 − F1 | 215 | 17 | 9 | 104 |

A0 and A1 shared 207 correct detections at the fixed point. A0
alone detected 33, A1 alone detected 17, and both missed 88.
Fusion had 19 localization errors, seven classification errors, two
duplicates and 19 background false positives. A0 had 27, eight, 18
and 24, respectively. A1 had 28, eight, 21 and 29. WBF had 84
unassigned misses versus 66 and 83. The singleton score penalty
can remove correct single-model detections at the fixed threshold.

| D1 object ID | F0 outcome | F1 outcome | F2 outcome |
|---|---|---|---|
| images/val/img_02_4402329000_00001.jpg:0 | classification | correct | classification |
| images/val/img_02_4402329000_00001.jpg:1 | correct | correct | correct |
| images/val/img_02_4402329000_00001.jpg:2 | missed | correct | correct |
| images/val/img_03_4402329000_00001.jpg:0 | classification_and_localization | localization | classification_and_localization |

Rolled pit has only four objects across two images. The run stores
scores, IoUs, labels and overlays for individual review. Inclusion,
crease, silk spot and oil spot are included in the class table and
per-object diagnostics.

## Geometry and cost

| Short side (px) | Objects / images | A0 recall | A1 recall | WBF recall | WBF recovered / lost vs A0 |
|---|---|---|---|---|---|
| ≤16 | 35 / 25 | 45.71% | 37.14% | 37.14% | 2 / 5 |
| (16, 24] | 76 / 58 | 73.68% | 68.42% | 68.42% | 1 / 5 |
| (24, 32] | 51 / 46 | 84.31% | 78.43% | 80.39% | 0 / 2 |
| (32, 64] | 49 / 41 | 53.06% | 53.06% | 53.06% | 3 / 3 |
| >64 | 134 / 121 | 73.88% | 69.40% | 74.63% | 5 / 4 |

At ≤16 px, WBF detected 13 of 35 objects, matching A1, versus 16
for A0. Full D1 size and aspect-ratio tables are in `size_analysis.csv`.
Median WBF CPU overhead was 0.185 ms/image
(p95 0.731 ms) for preloaded detections.
This excludes both models’ inference and all file I/O.

## Decision for review

The gain at a matched false-positive budget supports investigating
complementary supervision for A1. The evidence against A0 remains
inconclusive. S1-Fusion uses two predictions at inference, while the
proposed S1-KD test would retain the A1 student graph at inference.
Prediction complementarity does not establish a distillation benefit.
Review the [reserved S1-KD protocol](../experiments/s1_teacher_fusion.md#reserved-protocol-for-s1-kd)
before implementing a training loss.

The original run, including `summary.json`, per-class and per-object
CSVs, native AP arrays, figures and source snapshots, remains at
`experiments/s1-teacher-fusion-01/`.
Use `review_examples/recovered-*` for the all-size recovered examples.
Their corrected titles and plotting source are stored alongside the
original figures.
