# A1 — P2 architectural ablation

**Completed — scientifically reviewed mixed result.**
A0 is the matched original AGE-YOLO control. A1 adds P2 and is the current
development architecture. A1 produced complementary detections and improved
rolled-pit AP on four objects across two images. Overall AP gains were
inconclusive, and fixed recall fell. A1 also trailed A0 in the ≤16 px
subset and at every evaluated false-positive budget.

## Training and architecture

| Record | A0 · AGE-YOLO | A1 · AGE-YOLO + P2 |
|---|---|---|
| Training run | `a0-age-yolo-640-b32-02` | `a1-age-yolo-p2-640-b32-01` |
| PBS job | `47977.mgmt01` | `47909.mgmt01` |
| Training resolution | 640 | 640 |
| Physical batch / accumulation | 32 / 2 | 32 / 2 |
| Seed | 42 | 42 |
| Best epoch / completed epochs | 146 / 196 | 160 / 210 |
| Training duration | 2.008 h | 2.666 h |
| Displayed peak reserved training VRAM | 7.94 GiB | 14.00 GiB |
| Unfused parameters | 3,418,721 | 3,446,915 |
| Estimated GFLOPs at 640 | 7.874 | 9.690 |
| Detection strides | 8, 16, 32 | 4, 8, 16, 32 |

Both runs completed normal early stopping. The saved and effective training
configurations agree on the experimental settings, including random
initialization, optimizer, augmentation, AMP and native validation batch 64.
Model YAML, logical device and run paths account for the effective argument
differences. The trainer digest and source HEAD agree across both records.
The captured source HEAD was `5c57509307d7292bc400494a817a360f0be2b51d`
with working-tree changes. The upstream pin was
`99002665ff2f7faa5013c081b1b8f547ae4566ec`.

A1 upsamples the refined neck P3 feature from layer 16, fuses it with the
stride-4 backbone feature from layer 2 using ACF, and reduces the fused feature
to 16 channels with a 1×1 Conv. The original SIFHead consumes P2–P5 with
per-level projections and shared predictors. The original backbone and
P3–P5 routes are preserved. The comparison evaluates the full retrained
network with shared predictions and assignment. It cannot isolate the
causal contribution of every individual A1 detection to the added branch.
[Architecture details](../architecture/age_yolo_p2.md)

| Identity | SHA-256 |
|---|---|
| A0 `best.pt` | `d81967a9ec44d7df0235daf67c1d08fe34b8900714c875bceacd250a7c6aaf88` |
| A1 `best.pt` | `72533a364a7169df6ec7c47352416d30ab529d1607fc8ded79fa260c6bc18082` |
| A0 model YAML | `8001f80267127ee69be729d874caa3edb647efce9a23b0e7ba20c8c535cdc641` |
| A1 model YAML | `46318c44cfd480787f0fc60efb66906b1c3926e390ea146255b1e0776adddfc1` |
| AUG42 training manifest | `c68282dc8c3ac5ed68c46f3a3253eaa6782b0acb921b37e5cbbf24a5587264a7` |
| Original training manifest | `f9d7ca03ea1215117156d2ebc11dff34f71436038b72119abfc7b9ca9f83ab2f` |
| Validation manifest | `c48bbe089cd3813bde9737d7e26672bf1f0ba0d6c9ea5b6839ef8eeb0d443472` |
| D1 object records | `bd023530d3d73565afc94dd1994df538c47180878d63870024bc7836354cdd0f` |

## Standalone validation

PBS job `47980.mgmt01` exited successfully. Both checkpoints used the
same original GC10-DET validation split with 230 images, 345 unique D1
objects and ten classes. The protocol uses full-image inference at 640,
rectangular batch 16, FP32 and native class-aware NMS. Diagnostic matching
uses confidence ≥ 0.25 and correct-match IoU ≥ 0.50. Native AP, precision,
recall and per-class AP were reproduced from the instrumented exports.
[Protocol and registered analysis](../experiments/a1_p2_detection.md)

| Metric | A0 | A1 | A1 − A0 |
|---|---:|---:|---:|
| mAP50 | 70.01% | 71.00% | +0.99 pp |
| mAP50–95 | 35.09% | 35.21% | +0.12 pp |
| Native precision | 78.59% | 71.31% | −7.28 pp |
| Native recall | 64.30% | 68.38% | +4.08 pp |
| Fixed precision | 72.73% | 68.50% | −4.23 pp |
| Fixed recall | 69.57% | 64.93% | −4.64 pp |
| True positives | 240 | 224 | -16 |
| False positives | 90 | 103 | +13 |
| False negatives | 105 | 121 | +16 |
| Localization errors | 27 | 28 | +1 |

Native precision and recall use the fork’s score-sweep operating point.
Fixed metrics use the same declared threshold for both models.

The paired image-level bootstrap used 5,000 draws and seed 42. For
ten-class AP, 675 draws lacked complete class support and were excluded
from both ten-class intervals. Nine-class AP requires support for every
non-rolled-pit class. Recall requires at least one object in the relevant
group. The intervals describe validation-image uncertainty for the two
fixed checkpoints from one seed. They cannot estimate training-seed
variability.

| Metric | A1 − A0 (pp) | 95% interval (pp) | Valid draws |
|---|---:|---:|---:|
| mAP50 | +0.99 | [−6.08, +4.73] | 4,325 |
| mAP50–95 | +0.12 | [−3.07, +2.30] | 4,325 |
| Nine-class AP50 | −3.72 | [−7.84, +0.39] | 5,000 |
| Nine-class AP50–95 | −1.74 | [−4.01, +0.88] | 5,000 |
| Fixed recall | −4.64 | [−8.71, −0.84] | 5,000 |
| Short-side Q1 recall | −6.59 | [−15.71, +1.32] | 5,000 |
| Short side ≤16 px recall | −8.57 | [−27.78, +5.26] | 5,000 |

Both overall AP intervals span zero. The fixed-recall interval lies
below zero. The ≤16 px point estimate also favors A0, with uncertainty
spanning zero for that smaller group.

## Defect size

At ≤16 px, A0 detected 16 of 35 objects and A1 detected 13. The change
was −8.57 percentage points. A1 recovered two and lost five. In this
experiment the P2 extension did not improve overall small-defect recall.

| Group | Objects | Images | Recall A0 / A1 | Δ recall (pp) | Localization A0 / A1 | Median correct IoU A0 / A1 | Recovered / lost |
|---|---:|---:|---:|---:|---:|---:|---:|
| ≤16 | 35 | 25 | 45.71% / 37.14% | −8.57 | 8 / 5 | 0.661 / 0.669 | 2 / 5 |
| (16, 24] | 76 | 58 | 73.68% / 68.42% | −5.26 | 5 / 5 | 0.669 / 0.676 | 3 / 7 |
| (24, 32] | 51 | 46 | 84.31% / 78.43% | −5.88 | 4 / 6 | 0.729 / 0.759 | 0 / 3 |
| (32, 64] | 49 | 41 | 53.06% / 53.06% | +0.00 | 3 / 1 | 0.737 / 0.782 | 5 / 5 |
| >64 | 134 | 121 | 73.88% / 69.40% | −4.48 | 7 / 11 | 0.781 / 0.780 | 7 / 13 |

The D1 area and short-side group boundaries were set from the original
training annotations. Localization counts below are localization-only
assignments. Image counts can overlap between groups.

| Group | Objects | Images | Recall A0 / A1 | Δ recall (pp) | Localization A0 / A1 | Median correct IoU A0 / A1 | Recovered / lost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Area Q1 | 92 | 59 | 60.87% / 54.35% | −6.52 | 15 / 10 | 0.728 / 0.707 | 4 / 10 |
| Area Q2 | 78 | 62 | 61.54% / 58.97% | −2.56 | 6 / 4 | 0.738 / 0.782 | 5 / 7 |
| Area Q3 | 97 | 86 | 78.35% / 74.23% | −4.12 | 2 / 5 | 0.749 / 0.739 | 2 / 6 |
| Area Q4 | 78 | 76 | 76.92% / 71.79% | −5.13 | 4 / 9 | 0.741 / 0.766 | 6 / 10 |
| Short side Q1 | 91 | 60 | 63.74% / 57.14% | −6.59 | 11 / 9 | 0.657 / 0.669 | 5 / 11 |
| Short side Q2 | 81 | 68 | 77.78% / 72.84% | −4.94 | 7 / 7 | 0.729 / 0.717 | 0 / 4 |
| Short side Q3 | 84 | 73 | 60.71% / 57.14% | −3.57 | 5 / 2 | 0.761 / 0.781 | 6 / 9 |
| Short side Q4 | 89 | 85 | 76.40% / 73.03% | −3.37 | 4 / 10 | 0.770 / 0.778 | 6 / 9 |
| Aspect ratio Q1 | 89 | 78 | 62.92% / 56.18% | −6.74 | 12 / 14 | 0.753 / 0.768 | 5 / 11 |
| Aspect ratio Q2 | 72 | 60 | 70.83% / 66.67% | −4.17 | 7 / 2 | 0.729 / 0.769 | 3 / 6 |
| Aspect ratio Q3 | 87 | 77 | 70.11% / 70.11% | +0.00 | 5 / 6 | 0.781 / 0.769 | 6 / 6 |
| Aspect ratio Q4 | 97 | 85 | 74.23% / 67.01% | −7.22 | 3 / 6 | 0.709 / 0.699 | 3 / 10 |

## Detection-level contributions

All 1,303 retained A0 predictions and 1,413 retained A1 predictions at
the NMS score floor received an unambiguous level. Level assignment joins
native NMS output to decoded input-space coordinates, class and score.
The table reports fixed-point detections for A1.

| A1 level | Correct | Correct ≤16 px | Recovered | Background FP |
|---|---:|---:|---:|---:|
| P2 | 61 | 11 | 5 | 15 |
| P3 | 57 | 0 | 6 | 8 |
| P4 | 85 | 2 | 1 | 5 |
| P5 | 21 | 0 | 5 | 1 |

P2 supplied 11 of A1’s 13 correct detections in the ≤16 px group and
15 background false positives. P3–P5 together recovered 12 objects.
The net change reflects all four levels after retraining and NMS.

## Priority classes and paired errors

| Defect | AP50 A0 | AP50 A1 | AP50–95 A0 | AP50–95 A1 |
|---|---:|---:|---:|---:|
| Inclusion | 48.20% | 34.03% | 13.96% | 13.48% |
| Crease | 33.26% | 22.94% | 15.90% | 14.01% |
| Silk spot | 59.98% | 58.26% | 24.71% | 26.45% |
| Oil spot | 72.20% | 69.64% | 36.66% | 35.68% |
| Rolled pit | 31.46% | 74.85% | 11.44% | 28.32% |

Rolled pit went from one correct detection out of four to three. These
four annotations are from two images. At the fixed point, A0 already
assigned a rolled-pit score below 0.25 to overlapping candidates for the
first and third objects. The stronger A1 scores moved them above the
threshold. The second object had better IoU despite lower confidence.
The fourth received the correct class but its IoU remained below 0.50.

| D1 object ID suffix | A0 outcome (IoU, score) | A1 outcome (IoU, score) |
|---|---|---|
| `img_02_4402329000_00001.jpg:0` | Classified as oil spot, P4 (0.664, 0.562) | Correct, P3 (0.649, 0.591) |
| `img_02_4402329000_00001.jpg:1` | Correct, P3 (0.598, 0.727) | Correct, P3 (0.641, 0.391) |
| `img_02_4402329000_00001.jpg:2` | Missed at the fixed point | Correct, P2 (0.786, 0.399) |
| `img_03_4402329000_00001.jpg:0` | Oil spot and poor localization, P4 (0.179, 0.545) | Rolled pit with poor localization, P3 (0.167, 0.379) |

Excluding rolled pit, AP50 was **74.29% for A0 and 70.57% for A1**.
AP50–95 was **37.71% and 35.97%**, respectively. Both nine-class
intervals span zero. The rolled-pit improvement is promising descriptive
evidence. Ten-class AP remains the primary metric.

Of 345 objects, 207 were detected by both checkpoints, 17 recovered by
A1, 33 lost by A1 and 88 missed by both. Among the 207 detected by both,
114 had lower matching IoU under A1. At the fixed point, 47 false
positives were shared, 56 were added and 43 were removed, using
same-class pairing at IoU ≥ 0.50.

| Fixed-point category | A0 | A1 |
|---|---:|---:|
| Missed with no assigned detection | 66 | 83 |
| Localization only | 27 | 28 |
| Classification only | 8 | 8 |
| Classification and localization | 4 | 2 |
| Duplicate | 18 | 21 |
| Background FP | 24 | 29 |
| Overlap with an accounted object | 9 | 15 |

A0 achieved higher recall at all four budgets. Realized FP/image values
differ slightly where tied scores admit discrete sets of detections.

| FP/image budget | A0 recall | A1 recall | A0 realized FP/image | A1 realized FP/image | Δ recall (pp) |
|---|---:|---:|---:|---:|---:|
| 0.10 | 53.33% | 50.14% | 0.096 | 0.100 | −3.19 |
| 0.25 | 65.22% | 58.55% | 0.235 | 0.226 | −6.67 |
| 0.50 | 71.30% | 66.09% | 0.496 | 0.496 | −5.22 |
| 1.00 | 75.07% | 71.30% | 0.991 | 0.965 | −3.77 |

## Measured inference cost

Measurements used native, uninstrumented FP32 timing on `gpu01`,
an NVIDIA H100 80 GB HBM3 at `cuda:6`, physical GPU UUID
`GPU-61f54ca7-3cee-1c93-def3-118523e95241`. The run recorded driver
570.124.06, PyTorch 2.7.1+cu128, CUDA 12.8 and the pinned fork
reporting Ultralytics 8.3.9. Timing order was A0, A1, A1, A0 with
the same 40-image batch-1 sample, five warm-up batches, and all
230 images for the batch-16 throughput passes. Actual NCHW shapes
were 1 × 3 × 352 × 672 and 16 × 3 × 352 × 672 with a six-image final
batch. The phase columns are means per image. End-to-end includes
loading and collation as well as the three reported phases.

| Pass / model | Batch | Pre (ms/img) | Inference (ms/img) | Post (ms/img) | End to end (ms/img) | Batch-1 median / p95 (ms) | Images/s | Peak allocated / reserved (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 / A0 | 1 | 1.092 | 7.111 | 2.104 | 20.559 | 19.26 / 28.25 | 48.64 | 139.8 / 200.0 |
| 1 / A0 | 16 | 0.051 | 0.567 | 0.935 | 2.043 | — | 489.59 | 367.5 / 500.0 |
| 2 / A1 | 1 | 1.336 | 8.030 | 2.258 | 24.740 | 24.54 / 35.20 | 40.42 | 122.3 / 174.0 |
| 2 / A1 | 16 | 0.048 | 0.938 | 0.823 | 2.281 | — | 438.36 | 664.5 / 884.0 |
| 3 / A1 | 1 | 0.966 | 8.176 | 2.279 | 23.907 | 21.08 / 35.56 | 41.83 | 122.3 / 150.0 |
| 3 / A1 | 16 | 0.084 | 0.957 | 1.109 | 2.639 | — | 378.95 | 664.5 / 884.0 |
| 4 / A0 | 1 | 1.479 | 7.063 | 1.663 | 22.463 | 20.38 / 34.87 | 44.52 | 77.4 / 120.0 |
| 4 / A0 | 16 | 0.064 | 0.570 | 0.484 | 1.560 | — | 641.17 | 352.4 / 434.0 |

Mean batch-1 median latency rose from **19.82 ms to 22.81 ms**.
Mean batch-16 throughput fell from **565.38 to 408.66 images/s**.
Batch-16 peak allocated and reserved memory increased in both A1
passes. Batch-1 memory varied across passes. The stride-4 feature
maps increase activation-memory requirements, particularly with
larger batches. The 0.82% parameter increase and 23.05% estimated
GFLOP increase describe model complexity. Measurements in the table
come from the timed inference workflow. Training VRAM and durations
are reported separately above.

## Scientific decision

B1 remains the historical AGE-YOLO baseline. A0 remains the matched
control. Retain A1 as the development architecture for its
complementary detections and descriptive rolled-pit result. The
experiment did not establish an overall AP gain, improved small-defect
recall, or better recall at matched FP budgets. Future architectures must
be compared with both A0 and A1. A2 is a planned context-aware SIFHead
modification starting from A1. Its design and implementation await
scientific review.

## Preserved artifacts

The original run directories remain under `experiments/`. The archives
were compared directly against the original files and verified using the
`SHA256SUMS` files in `artifacts/A0/` and `artifacts/A1/`.
Training archives contain their launcher logs. The evaluation archive
contains its persistent terminal log, native A0/A1 outputs, full
prediction exports, plots and examples. Standalone PBS output files
were unavailable at finalization. The PBS job-record archive preserves
available scheduler status and exit records.

| Archive | SHA-256 |
|---|---:|
| `artifacts/A0/a0-age-yolo-640-b32-02.tar` | `867c6133d75b093de33a3740c09360892bbb78821566f9338e901a321cc40236` |
| `artifacts/A1/a0-a1-pbs-job-records.tar` | `f8e3048e24502bc341708a40c9cbca371c67b2ccc8aed092a4e6881b169b2ca1` |
| `artifacts/A1/a1-age-yolo-p2-640-b32-01.tar` | `67128a268c5cf97c4066c581de275a57355fbb8655999265553228ddf5788a65` |
| `artifacts/A1/a1-architecture-eval-01.tar` | `4a303104f75f3e5cc53fc73e2e5f1ad6ab24215e965095703aa55bb6c7a3d541` |
