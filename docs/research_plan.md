# FlawTrace research plan

Industrial surface inspection needs to detect defects that can occupy a
narrow strip of an otherwise large image. The baseline sends each image
through AGE-YOLO at a nominal long side of 640 pixels. D1 asks whether
defects with a narrow short side at that scale are missed more often or
localized less accurately. The revised hypothesis is that combining global defect context with
additional spatial detail may improve small-defect recall while preserving
localization of large defects. E1 found that 960 alone degraded overall
accuracy, while recovering some objects missed at 640.
[Reviewed E1 results](results/e1_resolution.md)

The objectives are higher validation recall and AP, especially for narrow
defects, with bounded latency, peak GPU memory, and compute per image.
Inference-only comparisons should reuse the same checkpoint.
Architecture comparisons should train matched controls. Use the same
validation images, evaluator and hardware in each comparison. Include
all preprocessing, inference and postprocessing in end-to-end latency.

| ID | Experiment | Role |
|---|---|---|
| B0 | Official checkpoint verification | Historical reference |
| B1 | Independent AGE-YOLO baseline | Accepted primary baseline |
| D1 | Baseline error analysis | Completed and scientifically reviewed |
| E1 | Higher-resolution inference | Completed and independently reviewed |
| E1A | Offline cross-resolution prediction fusion | Completed and scientifically reviewed. Negative result. |
| E2 | Matched-resolution training at 960 | Completed and scientifically reviewed. Mixed results. |
| E3 | Fixed two-crop tiled inference | Completed and scientifically reviewed. Negative result. |
| E4 | Full-image and tiled prediction fusion | Deferred after E1A and E3 |
| E5 | Selective refinement | Deferred with E4 |
| A0 | Original AGE-YOLO matched architecture control | Completed training and standalone validation. |
| A1 | Stride-4 P2 AGE-YOLO extension | Completed and scientifically reviewed. Mixed result. Current development architecture. |
| A2 | Regression-specific SIFHead context starting from A1 | Completed and independently reviewed. Negative overall result. |
| S1-Fusion | Same-resolution complementary detector analysis | Completed on CPU and scientifically reviewed. |
| S1-KD | Teacher-supervised P2 detector | Reserved. Protocol for review. |
| V1 | Experimental verification | Conditional on identifying a method worth verifying |
| V2 | Final held-out test evaluation | Conditional on V1, frozen choices, and explicit approval |

Use validation alone to develop, select, and tune methods. The recorded
B0 test result is historical. Do not use held-out results to select a
method, confidence threshold, tile policy, or checkpoint.
[Baseline registry](experiments.md)

## Roadmap revision after E1A

The previous sequence moved directly from E1A to tiled inference (E2),
then prediction fusion (E3) and selective tiled inference (E4). E1 found
poor localization when the 640-trained model ran at 960. E1A recovered
some defects but lowered AP and trailed the 640 control under every
shared false-positive budget. E2 now tests training at the target
resolution before introducing the additional variables of tiling.
[Reviewed E1A report](results/e1a_fusion.md)

## Decision after E3

E4 was planned as full-image and tiled prediction fusion, conditional on
E3. E5 was planned as selective refinement, conditional on E4. E1A's
fusion rules reduced AP and failed to improve recall under shared
false-positive budgets. E3 improved neither AP nor recall under those
budgets and increased inference cost. E4 and E5 are now deferred.
[Reviewed E3 results](results/e3_tiled_inference.md)

## Architecture direction after E3

A1 tests whether a stride-4 P2 branch conditioned on P3 improves recall
and localization for small or narrow defects while retaining the
existing P3–P5 routes and full-image 640 context. A0 reruns the original
architecture under matching training settings. B1 remains the historical
reference. Matched standalone validation found inconclusive overall AP
gains. A1 had lower fixed recall, lower observed recall for short sides
≤16 px and lower recall at every evaluated false-positive budget. P2
contributed correct detections in the smallest group, and A1 improved
rolled-pit performance on four objects across two images.
[Matched A0/A1 protocol](experiments/a1_p2_detection.md) · [Reviewed results](results/a1_p2_detection.md)

Retain A1 as the development architecture for its complementary
detections and descriptive rolled-pit result. Future candidates must
compare with both A0 and A1. A2 tested regression-only context starting
from A1. It improved inclusion AP50 but did not improve overall accuracy
or localization counts. The comparison used one seed.
[Reviewed A2 results](results/a2_context_head.md)

## Training-time supervision after A2

A0 and A1 may learn complementary representations of steel defects.
Transferring useful A0 features to A1 during training could improve
detection while preserving single-model P2 inference. S1-Fusion checks
complementarity using archived, same-resolution predictions. At 0.50
FP/image, WBF improved recall over A1 by 7.54 pp. Its 2.32 pp gain over
A0 was inconclusive under paired image resampling. A0 remains the
matched performance control and A1 the development architecture.
[S1-Fusion results](results/s1_teacher_fusion.md)

S1-KD is reserved for a frozen A0 teacher and A1 student. Its proposed
auxiliary loss supervises foreground in the shared P3–P5 features.
The protocol and its gate are recorded separately.
[S1-KD proposal](experiments/s1_teacher_fusion.md#reserved-protocol-for-s1-kd)

## D1 protocol

B1 is `experiments/hpc-full-06/weights/best.pt`. B0 is the pinned upstream
`weights/pretrained_gc10.pt`. B1’s archived validation includes plot
examples and aggregate metrics, which cannot supply full per-image
detections. D1 captures full-precision post-NMS predictions for every
validation image using the existing evaluator and compares its native AP
against the recorded validation results within 0.002 AP units by default.
The evaluator retains its 0.001 score floor, class-aware NMS at 0.7 IoU,
300 detections per image, FP32, batch 16, and rectangular batching at 640.
AP comes from the native score sweep over retained detections. The parity
flag records deviations for review before interpreting diagnostics.

Convert normalized YOLO boxes to original pixel coordinates using the
manifest dimensions. For geometry, multiply widths and heights by
640 / max(image width, image height). This is the nominal aspect-preserving
content scale before stride padding and integer rounding. Translations
from letterboxing do not change box side lengths. Native post-NMS boxes
are reverse-mapped by the validator to original coordinates for IoU
matching. Use the original training annotations to calculate the 25th,
50th and 75th percentiles of area at 640, short side at 640, and long/short
aspect ratio. Values on a boundary go into the lower quartile. Report
empty groups with undefined recall.

The diagnostic operating point is confidence ≥ 0.25, foreground IoU
≥ 0.50, and candidate error IoU ≥ 0.10. These settings are recorded in the
effective YAML.

1. Per image, visit detections in descending confidence, preserving native
   order for equal scores. Match each to the free same-class object with
   the highest IoU ≥ 0.50. Ties use annotation order.
2. Detections with a qualifying same-class IoU whose candidates are
   already correctly assigned are duplicates.
3. For each remaining detection in confidence order, pair it with the
   highest-IoU unassigned object at IoU ≥ 0.10. An incorrect class at IoU
   ≥ 0.50 is a classification error. A correct class at IoU < 0.50 is a
   localization error. An incorrect class below 0.50 is a combined error.
4. Detections below 0.10 IoU with every object are background false
   positives. Residual detections above that floor whose candidates are
   already accounted for are reported separately. Objects with no
   assignment are missed.

Each object and detection participates in at most one correct or error
pair. A duplicate can reference its object without claiming it. TP counts
only correct pairs. Every other retained detection counts as FP, and every
object without a correct pair counts as FN, including objects paired with
classification or localization errors. The categories are inspired by
TIDE and use a fixed operating point with explicit one-to-one pairing.
[TIDE paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480562.pdf)

B0 and B1 are compared using shared image and annotation IDs. Compare
geometry, original training class support and a local contrast proxy
across paired outcomes. Contrast is the absolute difference between
median grayscale intensities inside the original box and in a clipped
surrounding ring, padded by one quarter of its short side (minimum two
pixels). Treat this as a diagnostic proxy. Defects can be textured, and
background, class, and geometry can covary. Rolled-pit results are reported
separately because there are only four instances in two images.

## Execution and outputs

Set the authorized logical CUDA device in `configs/error_analysis.yaml`.
Use a fresh run ID for every submission. From the repository root:

```bash
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate flawtrace-gc10
python scripts/analyze_errors.py --preflight
job=$(qsub submit_analysis.sh)
qstat -f "$job"
tail -F experiments/d1-baseline-01/terminal.log
```

Before Python starts, PBS holds launcher output in its normal job log.
After startup, the per-run terminal log tracks both native validation
passes. The run directory stores CSV diagnostics, a JSON summary, effective
YAML, full detections and fixed-point assignments under `predictions/`,
native validation evidence and environments under `B0/` and `B1/`, PDF
and PNG plots, seeded B1 overlays under `examples/`, and `report.md`.
The completed D1 run passed AP parity. See the [reviewed report](results/d1_error_analysis.md) for the results and limitations.

## E1 protocol

E1 compares the frozen B1 checkpoint at 640 and 960 pixels within one
PBS job on the same authorized CUDA device. Native validation uses
batch 16, eight workers, FP32, rectangular batches, a 0.001 score floor,
class-aware NMS at IoU 0.7, a cap of 300 detections per image, and no test-time
augmentation. The 640 control must pass the D1 AP tolerance before the
challenger runs. Validation staging checks image and label digests. The
experiment preserves D1 object identities and the area, short-side and
aspect-ratio quartiles defined at 640. All fixed-point comparisons use the
D1 matching rules.

The recall–FP sweep keeps tied scores together. For each shared FP/image
budget it reports the greatest recall attainable at or below the budget,
along with the actual FP/image and whether the score sweep spans that
budget. Actual FP/image can differ because thresholds admit discrete
sets of detections. The paired percentile bootstrap draws 230 images
with replacement 5,000 times using seed 42. Both resolutions use the
same multiplicity for each image. AP uses the native per-image TP flags
at the ten IoU thresholds and the fork’s AP calculation on each resample.
Overall and Q1 recall use the D1 fixed operating point. Intervals describe
validation-image sampling for this checkpoint. Rare classes such as
rolled pit are described separately.

Timing uses the native detection loader, FP32 backend and postprocessing.
Each timing condition runs twice in 640, 960, 960, 640 order. CUDA
synchronization brackets each measured phase. Each pass warms up with
five batches. Batch 1 uses the same seeded image sample at both resolutions
and measures synchronous decode, resize, transfer, forward and NMS. The
preprocessing column measures native batch preprocessing after input
loading. End-to-end latency includes loading and collation. Batch 16
measures steady loader throughput over all 230 images with eight requested
workers. Timed paths exclude metric accumulation, plots and artifact writes.
The timing table gives per-pass results, sampled NCHW shapes, and peak
CUDA allocated and reserved bytes. Disk caching and clock variation can
still affect the measurements.

Set DEVICE in `configs/resolution.yaml` to the authorized logical device.
Use a fresh run ID for any retry. From the repository root on the login node:

```bash
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate flawtrace-gc10
python scripts/evaluate_resolution.py --preflight
python scripts/record_source.py
job=$(qsub submit_resolution.sh)
qstat -f "$job"
tail -F experiments/e1-resolution-01/terminal.log
```

PBS startup messages are in `flawtrace-e1.o<job-number>`. Review the
parity flag, paired intervals, class support and accuracy–latency tradeoff
alongside the reviewed E1A results and revised roadmap.

## E3 protocol

E3 asks whether overlapping local crops recover defects missed by B1
full-image inference while holding the network's configured size at 640.
Local crops can improve available spatial detail and can also remove
context or truncate objects at internal boundaries. E3 is complete and
scientifically reviewed. [Reviewed E3 results](results/e3_tiled_inference.md)

Each original 2048 × 1000 validation image provides two crops. Left is
x [0, 1365) and right is x [683, 2048), both spanning y [0, 1000).
The 682-pixel overlap leaves every original pixel covered. Initial
resize maps 1365 × 1000 to 640 × 469, then the pinned rectangular
batching rule pads to 672 × 512 at stride 32. At 640 the crop scale in x
is 640/1365 versus 640/2048 on the full image, approximately 1.50×.
Each prediction is unpadded, divided by the actual per-axis gains from
the rounded resize, clipped to crop bounds and offset into the original
image. Native class-aware NMS runs within each crop. The second
class-aware NMS operates on original coordinates at IoU 0.7 and keeps
at most 300 detections per original. Scores and crop IDs stay attached.
Crop selection and suppression use no labels or full-image predictions.

AP uses the pinned matcher with one set of original D1 ground-truth
objects per image. Compare T with archived A, B and D, with T–A primary.
The paired bootstrap draws original images 5,000 times with seed 42,
and the same draw feeds all four conditions. The intervals describe
image-level uncertainty, with training-seed variation unmeasured.
The A–T oracle is retrospective and excluded from method rankings.

An object is contained when a crop covers its full box. Partial in both
means neither crop contains it. Boundary crossing requires a strict
crossing of x=683 or x=1365. Overlap intersection requires positive
area in the shared strip, and within overlap requires full containment
there. These groups overlap. Maximum visibility is the largest
crop intersection divided by original box area. Tables give all-object
and Q4 area strata with object and image counts, recall and localization
errors. Duplicate counts use the D1 fixed point before and after merge.
Clipping and overlap diagnoses use annotations only after inference.

Measure full-image 640 and tiled 640 on the same GPU with alternating
A, T, T, A passes. End-to-end time includes decode, crop generation,
resize/pad, transfer, model calls, remap and merge. The batch-1 workload
runs both crops serially for each sampled original. The batch-16
workload processes at most 16 views per forward pass. Throughput is
reported separately for originals and crops. Timing excludes digest
verification, metrics, plotting and artifact writes.

For an approved reproduction, choose a fresh run ID and set the authorized
DEVICE in `configs/tiling.yaml` before recording source.

```bash
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate flawtrace-gc10
python scripts/evaluate_tiling.py --preflight
python scripts/record_source.py
job=$(qsub submit_tiling.sh)
qstat -f "$job"
tail -F "flawtrace-e3.o${job%%.*}"  # Per-run terminal.log is under experiments/<new-run-id>.
```
