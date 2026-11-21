# E1 higher-resolution inference

Completed and independently reviewed. PBS job 47768.mgmt01 compared the
frozen B1 checkpoint at 640 and 960 on the same GPU with the same native
settings and 230 manifest-verified validation images. The 640 control
reproduced both D1 AP values exactly.

| Metric | 640 | 960 |
|---|---:|---:|
| mAP50 | 69.53% | 41.10% |
| mAP50–95 | 34.12% | 17.24% |
| Fixed precision | 67.72% | 34.63% |
| Fixed recall | 68.12% | 48.99% |
| True positives | 235 | 169 |
| False positives | 112 | 319 |
| False negatives | 110 | 176 |
| Localization errors | 30 | 96 |

The 960 condition was worse overall. Paired image bootstrap intervals
for 960 minus 640 were −33.54 to −22.96 percentage points for mAP50,
−20.53 to −13.87 for mAP50–95, and −24.63 to −13.74 for fixed recall.
For Q1 recall, the interval was −23.47 to +2.53 points. All intervals
used 5,000 repetitions with seed 42.

## Paired objects and localization

Both resolutions detected 152 objects and missed 93. At 960, 17 were
recovered and 83 were lost. Eleven of the 17 recovered objects were in
the smallest short-side quartile. Of the 83 lost objects, 61 became
localization errors.

For each of those 61 objects, select the highest-IoU same-class 960
prediction with confidence at least 0.25 in the same image. Use the
saved boxes in original image coordinates. Median intersection divided
by ground-truth area was **44.7%**, and median predicted area divided
by ground-truth area was **0.47**. For **36 of 61** objects, intersection
divided by predicted area was at least 0.8 while predicted area was
less than half the ground-truth area.

This pattern is consistent with partial-defect detection at higher
resolution. Reduced contextual coverage and a mismatch between the
training and inference resolutions are possible explanations. This
comparison does not isolate either mechanism. The experiment used one
checkpoint and validation split.

## Artifacts

The original run is preserved at `experiments/e1-resolution-01/`.
The complete run and PBS log are in `artifacts/E1/e1-resolution-01.tar`.
Verify with `(cd artifacts/E1 && sha256sum -c SHA256SUMS)`.
[Protocol](../research_plan.md#e1-protocol)
