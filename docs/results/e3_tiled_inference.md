# E3 fixed tiled inference

**Completed and scientifically reviewed. Negative result.**
Run `e3-tiling-01` evaluated frozen B1 on the original GC10-DET validation
split with 230 images and 345 unique objects. Job `47871.mgmt01` completed
successfully on `gpu02`. Reused A, B and D inputs matched all 12 recorded
SHA-256 digests, and their native AP and fixed-point counts matched E2.
The 460 crop entries verified the two [fixed crop windows](../research_plan.md#e3-protocol)
per image and their 682-pixel overlap.

| Condition | Method | mAP50 | mAP50–95 | Fixed recall | FP |
|---|---|---:|---:|---:|---:|
| A | B1 full image at 640 | 69.53% | 34.12% | 68.12% | 112 |
| B | B1 full image at 960 | 41.10% | 17.24% | 48.99% | 319 |
| D | E2 full image at 960 | 67.99% | 34.23% | 67.83% | 88 |
| T | B1, two crops at 640 | 37.96% | 15.92% | 49.28% | 422 |

## Paired comparisons

Changes and 95% percentile intervals are in percentage points, computed
from 5,000 paired original-image resamples per comparison with seed 42.

| Metric | T–A, primary | T–B | T–D |
|---|---:|---:|---:|
| mAP50 | −31.57 [−36.63, −25.88] | −3.14 [−5.99, −0.44] | −30.02 [−35.13, −24.78] |
| mAP50–95 | −18.20 [−21.97, −14.88] | −1.32 [−2.10, −0.35] | −18.30 [−22.01, −15.11] |
| Fixed recall | −18.84 [−24.78, −12.95] | +0.29 [−1.89, +2.54] | −18.55 [−24.92, −12.47] |
| Q1 short-side recall | −7.69 [−22.35, +5.21] | +2.20 [0.00, +4.92] | −8.79 [−25.00, +5.49] |

These intervals describe validation-image uncertainty with fixed
checkpoints. Training-seed variability remains unmeasured. T–D also
changes the checkpoint and its training settings.

## Crop boundaries and localization

| Ground-truth group | Objects | A recall | T recall | T localization errors |
|---|---:|---:|---:|---:|
| Fully contained in either crop | 305 | 65.57% | 54.75% | 63 |
| Partially visible in both crops | 40 | 87.50% | 7.50% | 35 |
| Largest-area quartile, partially visible in both | 16 | 75.00% | 0.00% | 14 |

| Fixed-point error category | A | T |
|---|---:|---:|
| Localization | 30 | 98 |
| Classification | 8 | 15 |
| Background false positives | 26 | 100 |
| Overlap with accounted objects | 19 | 177 |

For the 98 T localization errors, the median predicted-to-ground-truth
area ratio was **0.47**, versus **0.48** across the 96 localization errors
in E1's B condition. The earlier [E1 report](e1_resolution.md) gives
0.47 for a different subset of 61 objects lost at 960. The comparable
all-error statistics are consistent with partial-defect detection and
incomplete localization. T also lost recall among the 305 fully contained
objects, so clipping alone cannot explain the deterioration. These
groups are descriptive and do not isolate the effects of context,
geometry, or preprocessing. Fixed-point duplicates fell from 43 before
merge to 20 after merge.

## Complementary detections and false-positive budgets

A and T both detected **150** objects. T alone detected **20**, A alone
detected **85**, and both missed **90**. The retrospective oracle union
finds 255 of 345 objects (**73.91% recall**). It uses ground truth after
inference and does not describe a deployable method.

Thirteen of T's 20 recoveries belonged to Q1. The 20 were seven
inclusions, six oil spots, three silk spots, two creases, and two water
spots. T recovered only two of the 77 objects missed by every E2
condition, one crease and one inclusion.

| FP/image budget | A recall | T recall |
|---|---:|---:|
| 0.10 | 45.80% | 16.81% |
| 0.25 | 62.61% | 22.61% |
| 0.50 | 68.99% | 31.59% |
| 1.00 | 75.94% | 40.29% |

T underperformed A at every evaluated FP/image budget. Values are the
best fixed-point matching recall attainable within each shared budget.

## Same-GPU timing

Both conditions used the same H100 80GB HBM3 and the A–T–T–A timing
order. Ranges cover the two measured passes. Batch-1 timings are per
original image, including both serial views for T. End-to-end timing
includes decode, preprocessing, inference, remapping and merging.

| Measurement | A | T |
|---|---:|---:|
| Batch-1 median latency | 14.27–14.30 ms | 21.98–22.14 ms |
| Batch-1 p95 latency | 17.28–18.20 ms | 25.56–26.65 ms |
| Batch-16 original images/s | 394.74–448.68 | 259.72–261.83 |
| Batch-16 crop images/s | — | 519.44–523.66 |
| Batch-16 peak allocated memory | 338.85 MiB | 471.68 MiB |
| Batch-16 peak reserved memory | 464 MiB | 640 MiB |

T increased median batch-1 latency by approximately 54% using the mean
of each condition's two pass medians. Its batch-16 reserved memory was
176 MiB higher and original-image throughput declined. Crop throughput
counts two views per original.

## Decision and artifacts

The tested two-crop method does not qualify as an improvement over B1.
E1A and E3 motivate deferring E4 full-image/tiled prediction fusion and
E5 selective refinement. The next direction awaits scientific review
and maintainer approval. [Research roadmap](../research_plan.md)

The original run remains at `experiments/e3-tiling-01/`. The full run
and PBS output are in `artifacts/E3/e3-tiling-01.tar`, verified against
the original files. Check the archive with
`(cd artifacts/E3 && sha256sum -c SHA256SUMS)`.
