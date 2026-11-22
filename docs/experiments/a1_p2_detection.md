# A1 — P2 detection at 640

**Completed — scientifically reviewed mixed result.** A0 is the matched
architectural control and A1 is the current development architecture.
B1 remains the historical baseline. Future architecture changes will
compare against both A0 and A1. [Reviewed results and archives](../results/a1_p2_detection.md)

## Research basis

[D1](../results/d1_error_analysis.md) recorded B1 validation mAP50 of
69.53% and mAP50–95 of 34.12%. In D1's per-object records, B1 detected
11 of 35 objects with nominal short sides ≤16 pixels at 640 (31.43%
recall). That subset is small and the association is descriptive.
[E1](../results/e1_resolution.md) raised localization errors from 30 to
96 using frozen weights at 960. [E1A](../results/e1a_fusion.md) recovered
objects at the fixed threshold but reduced AP and recall at shared FP
budgets. [E2](../results/e2_training_resolution.md) recovered accuracy
by training at 960, with a +0.10 point mAP50–95 change relative to B1 and
a paired interval spanning zero. [E3](../results/e3_tiled_inference.md)
reduced mAP50–95 to 15.92% and increased localization errors and false
positives. These results motivate keeping the full image at 640 and
testing finer features inside the detector.

Adding a stride-4 P2 branch conditioned on P3 may improve recall and
localization of small or narrow defects while preserving full-image
context through the existing P3–P5 routes. The coarser routes retain
their graph and spatial scales. Shared gradients and denser target
assignment can still change their learned features or scores. A1 can
also increase false positives, latency and memory costs. Precision on
large defects and common false-positive budgets helps interpret a gain
on small defects.

P2 detection is an established technique. This is a controlled adaptation
of AGE-YOLO, with a comparison focused on steel surface defects.

- AGE-YOLO defines the detector and industrial inspection setting used
  here. [AGE-YOLO, DOI 10.1016/j.engappai.2026.114690](https://doi.org/10.1016/j.engappai.2026.114690)
- Feature Pyramid Networks established the top-down combination of
  deeper semantic features with finer feature maps.
  [Feature Pyramid Networks for Object Detection, DOI 10.1109/CVPR.2017.106](https://doi.org/10.1109/CVPR.2017.106)
- ATFL-Swin-YOLO describes a shallow P2 detection branch for small
  objects. [ATFL-Swin-YOLO, DOI 10.1038/s41598-026-61060-1](https://doi.org/10.1038/s41598-026-61060-1)

## Matched training

| ID | Configuration | Completed run |
|---|---|---|
| A0 | [Original AGE-YOLO](../../configs/training/age_yolo_a0.yaml) | `a0-age-yolo-640-b32-02` |
| A1 | [AGE-YOLO with P2](../../configs/training/age_yolo_a1_p2.yaml) | `a1-age-yolo-p2-640-b32-01` |

Both use the pinned deployment, AUG42 with 7,336 training images and
11,524 objects, seed 42, random initialization, deterministic training,
batch 32, 640 pixels, SGD with learning rate 0.01, momentum 0.937,
weight decay 0.0005, AMP and eight workers. The cap is 300 epochs with
patience 50. Native validation over 230 images and 345 objects selects
`best.pt` by 0.1 × mAP50 + 0.9 × mAP50–95. Mosaic and all other training
settings match B1. The archived B1 arguments differ in physical batch
(64), device (historical cuda:5), run paths, and model path for A1.
The native nominal batch is 64. Batch 32 yields accumulation 1–2 in
warm-up and 2 afterward, validation batch 64, and unmodified decay
0.0005 in the decayed parameter group. B1 used accumulation 1 and
validation batch 128. Both new configurations retain fixed-size training.

[A0 architecture audit](../architecture/age_yolo.md) ·
[A1 design, verification and memory estimates](../architecture/age_yolo_p2.md)

### Verified training evidence

| Record | A0 | A1 |
|---|---|---|
| PBS job | `47977.mgmt01` | `47909.mgmt01` |
| Detection strides | 8, 16, 32 | 4, 8, 16, 32 |
| Unfused parameters | 3,418,721 | 3,446,915 |
| Estimated GFLOPs | 7.874 | 9.690 |
| CSV best epoch | 146 | 160 |
| Epochs completed | 196 | 210 |
| Native training duration | 2.008 h | 2.666 h |
| Displayed peak reserved training memory | 7.94 GiB | 14.00 GiB |
| Best-epoch CSV mAP50 / mAP50–95 | 69.91% / 35.25% | 70.99% / 35.19% |

Both runs stopped after 50 epochs without improved native validation
fitness. Their recorded effective configurations agree in every
trainer argument besides device, model and run paths. Each selected
checkpoint matches its attempt record and archived model YAML, as
recorded at launch. The common training source commit is
`5c57509307d7292bc400494a817a360f0be2b51d` with captured worktree changes.
Both used the pinned AGE-YOLO revision
`99002665ff2f7faa5013c081b1b8f547ae4566ec`.

| Checkpoint | SHA-256 |
|---|---|
| A0 `best.pt` | `d81967a9ec44d7df0235daf67c1d08fe34b8900714c875bceacd250a7c6aaf88` |
| A1 `best.pt` | `72533a364a7169df6ec7c47352416d30ab529d1607fc8ded79fa260c6bc18082` |

The standalone comparison confirmed the training-time rolled-pit AP50
increase and declines on inclusion, crease, silk spot and oil spot.
Rolled pit has four validation objects across two images. The results
report records the individual assignments and nine-class sensitivity
analysis excluding rolled pit.

## Evaluation after training

Compare A1 against A0 first and report B1 at 640 and E2 at 960 as historical references.
Use each run's validation-selected `best.pt` for full-image 640
inference on the original GC10-DET validation split. Use batch 16,
FP32, rectangular batching, class-aware NMS with a 0.001 score floor
and IoU 0.7, at most 300 detections per image, and no test-time
augmentation. Report the fork's native AP50 and AP50–95.

Pair predictions by D1 image and annotation IDs and retain its original
area, short-side and aspect-ratio groups and fixed-point matcher.
Report recall and counts for short sides ≤16 pixels and the smallest
area group, and performance for inclusion, oil spot, silk spot and
crease. Check recall and localization in the largest area quartile
to assess whether A1 retains large-defect performance. Compare
localization errors, total false positives and recall
at the same FP/image budgets. Report A1–A0 deltas and paired
original-image uncertainty with the single-seed limitation. Report
parameters, GFLOPs and end-to-end latency under the same device,
precision and timing protocol for both models. Include evaluation
overhead for additional detections in postprocessing. Leave the test
split for the separately approved final evaluation.

The [frozen evaluation settings](../../configs/evaluation/age_yolo_a0_a1.yaml)
use D1 short-side cutoffs of 21.25, 38.75 and 86.875 px and area cutoffs
of 1055.859375, 6759.375 and 16768.359375 px². Fixed short-side groups
are ≤16, (16, 24], (24, 32], (32, 64] and >64 px. The ≤16 px group has
35 objects. At the fixed point a false positive is shared when its
same-class A0 counterpart reaches IoU 0.50 under one-to-one matching.
Otherwise it is added or removed. Record any decrease in IoU among
objects both checkpoints detect correctly.

The evaluator reads decoded candidate coordinates and scores and joins
them to native post-NMS rows before rescaling. Rows with an exact
cross-level tie are ambiguous and unmatched rows are recorded as
unattributed. The method leaves native NMS and metric inputs intact and
checks exported AP, precision, recall and per-class AP against the
native results. Timing uses uninstrumented inference in A0, A1, A1,
A0 order with identical sample, warm-up and full batch-16 workloads.
Report measured CUDA peaks separately from historical training peaks.

Bootstrap intervals use 5,000 shared image draws with seed 42 and the
native AP calculation. A ten- or nine-class AP draw requires all
classes in the corresponding mean. A recall draw requires at least
one object in the relevant group. The output records how many draws
had sufficient support. The intervals cover validation-image sampling
for this single training seed.

### Manual submission and packaging

For an approved repeat, set an unused run ID in the evaluation YAML and
set the configured logical device to the PBS-authorized GPU before
recording source on the management node. The evaluation uses the same
CUDA initialization as training and records the selected GPU UUID in
the run metadata.

```bash
cd ~/flawtrace-gc10
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate flawtrace-gc10
cfg=configs/evaluation/age_yolo_a0_a1.yaml
python scripts/record_source.py --architecture-eval --config "$cfg" &&
  PBS_O_WORKDIR="$PWD" bash submit_architecture_eval.sh --preflight &&
  job=$(qsub submit_architecture_eval.sh) &&
  qstat -f "$job"
run=$(python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["RUN_ID"])' "$cfg")
tail -F "experiments/$run/terminal.log"
# Before per-run logging begins:
# tail -F "flawtrace-a1-eval.o${job%%.*}"
```

The original run has been archived and verified. For a future completed
repeat, set `repeat_run_id` to that run ID before running the packaging
commands. Keep the original run and native A0/A1 files in place.

```bash
(
  set -e
  run="${repeat_run_id:?provide the completed repeat run ID}"
  dst=artifacts/A1
  python -c 'import json, sys; assert json.load(open(sys.argv[1]))["status"] == "completed"' \
    "experiments/$run/summary.json"
  mkdir -p "$dst"
  test ! -e "$dst/$run.tar"
  test ! -e "$dst/$run.tar.sha256"
  tmp=$(mktemp -d "$dst/.package.XXXXXXXX")
  trap 'rm -r -- "$tmp"' EXIT
  tar -C experiments -cf "$tmp/$run.tar" "$run"
  tar -C experiments -df "$tmp/$run.tar"
  (cd "$tmp" && sha256sum "$run.tar" > "$run.tar.sha256")
  ln "$tmp/$run.tar" "$dst/$run.tar"
  ln "$tmp/$run.tar.sha256" "$dst/$run.tar.sha256"
  (cd "$dst" && sha256sum -c "$run.tar.sha256")
)
```
