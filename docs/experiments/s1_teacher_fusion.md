# S1-Fusion — Same-resolution complementary detector analysis

**Completed on CPU.** [Reviewed results](../results/s1_teacher_fusion.md)

## Question and protocol

A0 and A1 may learn complementary representations of steel surface
defects. S1-Fusion tests whether the frozen detectors produce useful
complementary predictions before considering training-time supervision.
A0 is the matched original AGE-YOLO control. A1 is the provisional
development architecture with P2. A2 is a completed negative ablation.
[Architecture evidence](../results/a1_p2_detection.md) · [A2 report](../results/a2_context_head.md)

The experiment uses the original GC10-DET validation split with 230
images, ten classes and 345 D1 objects. The [locked YAML](../../configs/evaluation/teacher_fusion.yaml)
specifies F0 as A0, F1 as A1, and F2 as equal-weight weighted boxes
fusion (WBF). Both inputs were evaluated at 640. Fusion runs on CPU
using their archived post-NMS boxes in original image coordinates.
No model inference is required.

WBF groups boxes by class and processes scores in descending order.
A candidate joins the best matching cluster only above IoU 0.55.
Cluster coordinates are a confidence-weighted mean. With two equally
weighted models, default non-overflow `avg` confidence multiplies the
cluster’s mean score by `min(number of boxes, 2) / 2`. A singleton
therefore receives half its score. This reference rule counts boxes
even when two come from the same model. Coordinates are normalized
for the algorithm and restored to original pixels. The input and
final score floors are 0.001 and the final cap is 300 per image.
[Solyovyev et al., Weighted Boxes Fusion](https://arxiv.org/abs/1910.13302) ·
[Reference WBF implementation at the verified revision](https://github.com/ZFTurbo/Weighted-Boxes-Fusion/blob/651884c5c7df13a920d286404b47c6fa86944782/ensemble_boxes/ensemble_boxes_wbf.py)

The evaluator verifies both single-model prediction exports against
the checksummed A1 archive. Native AP uses the pinned validator’s
rectangular label preparation and IoU matcher at all ten thresholds.
The same transform is applied to the labels from the original archive.
D1’s untransformed original-pixel coordinates remain the basis for
fixed-point diagnosis and geometry groups. The original D1 coordinates
alone lowered reconstructed mAP50–95 by 0.379 pp for A0 and 0.294 pp
for A1. Native label preparation reproduced every archived TP flag
for both models. Thus F2 uses the same native AP geometry as the
single-model references.

Fixed-point diagnostics use confidence ≥0.25, class-aware matching at
IoU ≥0.50 and the D1 error rules. The primary comparison is recall at
0.50 false positives per image. Use tied score groups and the best
recall at or below each cap. For uncertainty, the primary cap is
recomputed in each paired image bootstrap sample. Draw 5,000 samples
with seed 42. Report class-support counts for ten- and nine-class AP.
Exclude a draw from an AP metric when a required class is absent.
The intervals describe validation-image uncertainty for frozen
predictions. They leave training-seed uncertainty unmeasured.

## Reproduce and package

Use a new run ID for a reproduction. The original experiment is
`s1-teacher-fusion-01`. The implementation rejects an existing run
directory. Set up the Python environment and run on CPU.

```bash
cd ~/flawtrace-gc10
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate flawtrace-gc10
cfg=$(mktemp --suffix=.yaml)
sed 's/^RUN_ID:.*/RUN_ID: s1-teacher-fusion-02/' configs/evaluation/teacher_fusion.yaml > "$cfg"
CUDA_VISIBLE_DEVICES='' python -u scripts/evaluate_teacher_fusion.py --config "$cfg"
```

To package the completed original run for review, preserve its
directory and create the archive and checksum in a new destination.

```bash
(
  set -e
  cd ~/flawtrace-gc10
  run=s1-teacher-fusion-01
  dst=artifacts/S1
  python -c 'import json, sys; assert json.load(open(sys.argv[1]))["status"] == "completed"' \
    "experiments/$run/summary.json"
  mkdir -p "$dst"
  test ! -e "$dst/$run.tar"
  test ! -e "$dst/$run.tar.sha256"
  temp=$(mktemp -d "$dst/.package.XXXXXXXX")
  trap 'rm -rf -- "$temp"' EXIT
  tar -cf "$temp/$run.tar" "experiments/$run"
  tar -df "$temp/$run.tar"
  (cd "$temp" && sha256sum "$run.tar" > "$run.tar.sha256")
  ln "$temp/$run.tar" "$dst/$run.tar"
  ln "$temp/$run.tar.sha256" "$dst/$run.tar.sha256"
  (cd "$dst" && sha256sum -c "$run.tar.sha256")
)
```

The run archives settings, input and source hashes, a snapshot of the
executed Python sources and YAML, environment, native AP exports,
prediction and fixed-point assignments, paired intervals, figures,
example detections and its terminal log. CPU timing measures only the
WBF operation on preloaded detections.

## Reserved protocol for S1-KD

**Proposed for independent review.** The student is A1 with the original
P2 inference graph. The frozen teacher is A0’s accepted
`a0-age-yolo-640-b32-02/weights/best.pt`, SHA-256
`d81967a9ec44d7df0235daf67c1d08fe34b8900714c875bceacd250a7c6aaf88`.
The teacher stays in evaluation mode with gradients disabled.
Both models receive the same tensors after the student’s native
augmentation, resizing and normalization. Start the student from
random initialization, with the matched A1 settings, AUG42 train,
original validation, 640 px, batch 32, epochs 300, patience 50, seed
42, SGD at 0.01, deterministic mode and AMP. Keep native target
assignment and detection loss. P2 has no direct auxiliary feature loss.

Use neck outputs before SIFHead. Both graphs use the same layer
indices and shapes at the shared levels, so no learned adapters
are needed for the proposed loss.

| Level | A0 and A1 layer | Stride | Shape at 640 |
|---|---:|---:|---|
| P3 | 16 | 8 | `[B, 64, 80, 80]` |
| P4 | 19 | 16 | `[B, 128, 40, 40]` |
| P5 | 22 | 32 | `[B, 256, 20, 20]` |

For each level, mark grid centers that fall inside the union of
annotated boxes in the final augmented tensor. Compute FP32 squared
feature differences against detached teacher features. Divide each
masked squared sum by that level’s foreground-cell count times its
channel count. Average over levels with foreground support, with zero
auxiliary loss if all are empty. Register the auxiliary weight before
training and apply the same batch scaling as the native detection
loss. Keep the teacher out of the student optimizer, EMA and checkpoint.
Record the teacher hash, layer IDs, mask rasterization and weight in
training provenance. On resume, validate the same teacher identity and
restore the original student and optimizer through native recovery.
Evaluation should load the student checkpoint without a teacher.

At batch 32, the stored P3–P5 teacher tensors occupy about 43.75 MiB
at FP16 or 87.50 MiB at FP32. Teacher weights alone occupy about
13.04 MiB at FP32. A forward also needs live intermediates, workspace
and auxiliary loss tensors. Teacher execution under `no_grad`
avoids a second autograd graph. Profile actual peaks before training
and keep the approved batch and accumulation settings. There is no
measured KD cost yet.

Foreground feature imitation is motivated by the spatially selective
detector supervision of Wang et al. and the foreground/background
distinction in Yang et al. The proposed loss uses annotated foreground
and corresponding neck tensors only. It omits anchor-near sampling,
background supervision, feature attention and global relation loss.
[Wang et al., CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Distilling_Object_Detectors_With_Fine-Grained_Feature_Imitation_CVPR_2019_paper.html) ·
[Yang et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_Focal_and_Global_Knowledge_Distillation_for_Detectors_CVPR_2022_paper.html)

The main KD contrast would be the student versus A1, with A0 as an
additional control, using matched standalone validation, per-class
and small-defect analysis, and FP budgets. Offline WBF provides
evidence about prediction complementarity. A separate experiment is
needed to test whether feature transfer improves one-model inference.
