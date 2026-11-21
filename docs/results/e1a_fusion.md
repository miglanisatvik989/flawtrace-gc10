# E1A offline cross-resolution fusion

E1A completed on CPU and has been scientifically reviewed. Neither fusion
rule qualified as an improvement over the 640 control. Fusion recovered
objects at the fixed threshold but reduced AP and produced lower recall at
every evaluated false-positive budget.

| Metric | F0 · 640 | F1 · 960 | F2 · pooled NMS | F3 · 640 priority |
|---|---:|---:|---:|---:|
| mAP50 | 69.53% | 41.10% | 58.58% | 65.33% |
| mAP50–95 | 34.12% | 17.24% | 27.96% | 31.74% |
| Fixed recall | 68.12% | 48.99% | 73.04% | 69.86% |
| True positives | 235 | 169 | 252 | 241 |
| False positives | 112 | 319 | 461 | 242 |

F0 and F1 reproduced the two E1 AP values exactly. At confidence ≥ 0.25
and match IoU ≥ 0.50, F2 recovered 17 objects and F3 recovered six. Both
retained all 235 objects detected by F0. F2 added ten localization errors
relative to F0, and F3 had one fewer. The retrospective union oracle
found 252 of 345 objects (73.04% recall). The oracle uses ground truth
and was excluded from deployable method rankings.

| FP/image budget | F0 recall | F1 recall | F2 recall | F3 recall |
|---|---:|---:|---:|---:|
| 0.10 | 45.80% | 22.03% | 32.75% | 42.61% |
| 0.25 | 62.61% | 30.72% | 46.96% | 51.88% |
| 0.50 | 68.99% | 37.97% | 51.30% | 63.48% |
| 1.00 | 75.94% | 45.22% | 66.38% | 69.57% |

Each value is the best recall at or below the shared budget. Realized
false-positive rates can differ because confidence thresholds admit
discrete sets of predictions.

## Paired bootstrap

Changes versus F0 are in percentage points, with 95% percentile intervals
from 5,000 paired image resamples using seed 42.

| Metric | F2 change [interval] | F3 change [interval] |
|---|---:|---:|
| mAP50 | −10.95 [−13.87, −7.43] | −4.20 [−6.12, −1.93] |
| mAP50–95 | −6.17 [−7.80, −4.21] | −2.39 [−3.39, −1.39] |
| Fixed recall | +4.93 [+2.34, +7.87] | +1.74 [+0.56, +3.21] |
| Q1 short-side recall | +12.09 [+3.90, +20.55] | +4.40 [+0.95, +9.20] |

The intervals describe validation-image sampling for this checkpoint.
They do not measure variation across training seeds. Rare classes such
as rolled pit have too few observations for strong conclusions.

## Latency and evidence

Median / p95 measured CPU fusion cost was 0.081 / 0.380 ms for F2 and
0.060 / 0.278 ms for F3. Estimated two-view latency was 29.76 ms and
29.74 ms, respectively, versus 14.41 ms for F0. The estimates add E1's
batch-1 view medians and median CPU overhead. No live end-to-end fusion
latency was measured.

The original run is at `experiments/e1a-fusion-01/`. The verified archive
at `artifacts/E1A/e1a-fusion-01-review.tar` includes the full run, fusion
source and configuration, and the E1 input predictions and timing table.
Verify with `(cd artifacts/E1A && sha256sum -c SHA256SUMS)`.
[Experiment registry](../experiments.md)
