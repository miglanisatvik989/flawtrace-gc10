# FlawTrace research plan

Industrial surface inspection needs to detect defects that can occupy a
narrow strip of an otherwise large image. The baseline sends each image
through AGE-YOLO at a nominal long side of 640 pixels. D1 asks whether
defects with a narrow short side at that scale are missed more often or
localized less accurately. The refined hypothesis is that increasing the effective spatial
resolution of steel surface defects improves detection recall and
localization, particularly for small-area defects and defects with narrow
spatial dimensions. [Reviewed D1 results](results/d1_error_analysis.md) describe
associations at 640 pixels. E1 is the next approved experiment and will
test the hypothesis with a controlled change in inference resolution.

The objectives are higher validation recall and AP, especially for narrow
defects, with bounded latency, peak GPU memory, and compute per image.
Future comparisons should reuse the same checkpoint, validation images,
and evaluator, report time and memory on the same hardware, and include
all preprocessing, inference and postprocessing in end-to-end latency.

| ID | Experiment | Role |
|---|---|---|
| B0 | Official checkpoint verification | Historical reference |
| B1 | Independent AGE-YOLO baseline | Accepted primary baseline |
| D1 | Baseline error analysis | Completed and scientifically reviewed |
| E1 | Higher-resolution inference | Next approved experiment and first controlled hypothesis test |
| E2 | Tiled inference | Conditional on E1 results and remaining errors |
| E3 | Full-image and tiled prediction fusion | Conditional on complementary detections in E2 |
| E4 | Selective tiled inference | Conditional on evidence that fusion helps and resource costs merit selection |
| V1 | Experimental verification | Conditional on identifying a method worth verifying |
| V2 | Final held-out test evaluation | Conditional on V1, frozen choices, and explicit approval |

Use validation alone to develop, select, and tune methods. The recorded
B0 test result is historical. Do not use held-out results to select a
method, confidence threshold, tile policy, or checkpoint.
[Baseline registry](experiments.md)

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
