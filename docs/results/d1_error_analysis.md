# D1 baseline error analysis

**Completed, scientifically reviewed and accepted.** Run
`d1-baseline-01` used the manifest-verified GC10-DET validation split
with 230 images, 345 objects and 10 classes. PBS job `47767.mgmt01`
finished on `gpu02` with exit status 0. The [D1 protocol](../research_plan.md#d1-protocol)
defines geometry, matching and grouping.

## Validation metrics and parity

| Checkpoint | Native mAP50 | Native mAP50–95 | Fixed-point TP | FP | FN |
|---|---:|---:|---:|---:|---:|
| B0 released weights | 76.27% | 42.02% | 258 | 124 | 87 |
| B1 independent baseline | 69.53% | 34.12% | 235 | 112 | 110 |

Native AP uses a score sweep over retained detections. Diagnostic counts
use confidence ≥ 0.25 and one-to-one correct matching at IoU ≥ 0.50,
with a 0.10 IoU floor for assigning an error to an object. Both models
passed the 0.002 AP-unit parity tolerance against earlier validation.
B0 differed by −0.014 percentage points at AP50 and −0.046 points at
AP50–95. B1 matched the accepted validation metrics exactly.

## Geometry and localization

Quartile cutoffs come from the 2,881 objects in the original training
annotations. Image counts can overlap across quartiles.

| Short-side quartile at 640 | Images | Objects | B0 recall | B1 recall | B1 localization errors |
|---|---:|---:|---:|---:|---:|
| Q1, ≤ 21.25 px | 60 | 91 | 62.64% | 58.24% | 13 |
| Q2, > 21.25 to 38.75 px | 68 | 81 | 77.78% | 76.54% | 5 |
| Q3, > 38.75 to 86.875 px | 73 | 84 | 72.62% | 58.33% | 8 |
| Q4, > 86.875 px | 85 | 89 | 86.52% | 79.78% | 4 |

Both checkpoints recalled smaller-area objects less often. Recall rose
from 58.70% to 87.18% between the smallest and largest area quartiles for B0
and from 56.52% to 79.49% for B1. Both also recalled fewer objects in the
smallest short-side quartile than in the largest. Short-side recall was
nonmonotonic, particularly for B1 in Q3. B1 had 13 of its 30 localization
errors in Q1. Among correctly matched B1 objects, median IoU was 0.657
in Q1 and 0.780 in Q4.

## Error distribution and paired outcomes

| B1 false-negative category | Objects |
|---|---:|
| Missed with no assigned detection | 65 |
| Localization | 30 |
| Classification | 8 |
| Combined classification and localization | 7 |
| Total | 110 |

Inclusion (30), silk spot (31), oil spot (21), and crease (15) accounted
for **97 of 110** B1 false negatives. Training frequency alone is
insufficient to explain the weaknesses. Silk spot had 713 training
objects and 59.21% validation recall, while crescent gap had 219 and
96.15%. Aspect ratio alone is insufficient too. B1 recall ranged from
64.04% to 71.26% across aspect-ratio quartiles without a monotonic pattern.

Both checkpoints detected 218 objects and missed 70. B0 alone detected
40 and B1 alone detected 17. Objects missed by both had a median short
side of 28.59 px and a median area of 1,791 px² at 640, compared with
44.69 px and 9,266 px² for objects detected by both. Median local grayscale
contrast was 1 and 2, respectively, on the 0–255 scale. This proxy can
miss textured defects. Rolled-pit recall was 3/4 for B0 and 2/4 for B1,
with only four objects across two images. Those estimates have limited
statistical reliability.

## Interpretation and provenance

D1 establishes descriptive associations at a single resolution. Area,
short side, class and contrast can covary, and multiple objects can come
from the same image. A causal effect of resizing remains unverified.
E1 is the next approved experiment and will test the hypothesis by changing
inference resolution on validation data.

| Verified input | SHA-256 |
|---|---|
| B0 checkpoint | `2daca14beedf36d6f7f74bca018292239a8eeb398fbe5b1c57516e410a61dc49` |
| B1 checkpoint | `7e61fc3da770652acfd58264c06b31480e04b6ac6440e79e3eae90f648c92275` |
| Original training manifest | `f9d7ca03ea1215117156d2ebc11dff34f71436038b72119abfc7b9ca9f83ab2f` |
| Validation manifest | `c48bbe089cd3813bde9737d7e26672bf1f0ba0d6c9ea5b6839ef8eeb0d443472` |
| D1 archive | `c4d699b595c03a2fab62dbc1fc561dfbe28df125481b6129af8546c6601b3ffd` |

The original run is retained at `experiments/d1-baseline-01/`. The archive
`artifacts/D1/d1-baseline-01.tar` includes that full directory and its PBS
log. Verify the archive and file inventory with
`(cd artifacts/D1 && sha256sum -c SHA256SUMS)`.
