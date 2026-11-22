# A2 — Context-aware regression head

**Completed and independently reviewed. Negative overall result.**
[Results and verified archives](../results/a2_context_head.md) A0 is the
original matched control and A1 is the current development reference.

## Hypothesis and evidence

Regression-specific spatial refinement may improve complete-defect
localization while preserving the fine P2 features seen by the
classifier. [D1](../results/d1_error_analysis.md) linked smaller defects
with reduced recall and described localization errors in the narrowest
quartile. [E1](../results/e1_resolution.md) found partial-defect boxes at
higher inference resolution with frozen weights.
[E2](../results/e2_training_resolution.md) restored much of the lost
accuracy with matched-resolution training. Its overall AP benefit
remained inconclusive. [E3](../results/e3_tiled_inference.md) reported
poor recall and more localization errors for partially visible objects.
These experiments vary image content, input scale or training
conditions and cannot identify the source of errors inside SIFHead.

[A1](../results/a1_p2_detection.md) obtained 71.00% mAP50 and 35.21%
mAP50–95 with 64.93% fixed recall, 28 localization errors and 83
unassigned misses. P2 supplied 61 correct detections, including 11 of
13 at ≤16 px. Rolled-pit AP50 rose from 31.46% to 74.85% across four
validation objects, while inclusion, crease, silk spot and oil spot
remained weak. A2 tests local regression context with the same four
detection levels and full-image resolution.

Task-specific feature pathways and dilated context are established
design ideas. Results from the cited detectors motivate the experiment
without predicting the outcome on GC10-DET.

- Wu et al. examined different classification and localization feature
  requirements in detection. [Rethinking Classification and Localization for Object Detection, CVPR 2020](https://arxiv.org/abs/1904.06493)
- Yu and Koltun used dilated convolutions to expand spatial support
  while preserving feature-map resolution. [Multi-Scale Context Aggregation by Dilated Convolutions, ICLR 2016](https://arxiv.org/abs/1511.07122)
- Luo et al. distinguished theoretical support from effective receptive
  field. [Understanding the Effective Receptive Field in Deep Convolutional Neural Networks, NeurIPS 2016](https://proceedings.neurips.cc/paper/2016/hash/c8067ad1937f728f51288b3eb986afaa-Abstract.html)
- Howard et al. demonstrated depthwise and pointwise factorization as a
  lightweight convolution pattern. [MobileNets, 2017](https://arxiv.org/abs/1704.04861)

## Matched training

Use the [A2 training configuration](../../configs/training/age_yolo_a2_context.yaml)
and [A2 head audit](../architecture/context_sifhead.md).
A2 starts from random initialization with seed 42, deterministic
fixed-size 640 training, batch 32, AMP, eight workers, SGD at
0.01, momentum 0.937, weight decay 0.0005, maximum 300 epochs and
patience 50. It uses the same AUG42 training manifest and unchanged
validation split. All 107 resolved native arguments were compared
against the completed A1 run. The four differences were model path,
logical device, run name and save directory. Nominal batch 64 gives
accumulation 2 after warm-up, native validation batch 64, and
effective decay 0.0005 in the decayed group. Native validation
selects `best.pt`.

A2 completed all 300 epochs in PBS job `47981.mgmt01` on `gpu01`.
Native training took 3.685 hours and displayed a 14.9 GiB reserved
memory peak. Epoch 273 had the highest native validation fitness.
Its training CSV recorded 65.791% mAP50 and 34.447% mAP50–95.
The selected checkpoint SHA-256 is
`d89418563912c8f240d069ec4a666451efd987177ccf69ba2cd1cdedd992db0d`.
The saved model YAML SHA-256 is
`99e46b7bee218d7a939b1c41b2f3231602685d4ea4fcc86106cd41e083231261`.
The archived configuration, source record and AUG42 manifests were
verified. These training-time metrics use the trainer’s validator.
The standalone comparison will test the preliminary class and overall
AP trends using the reviewed A0/A1 settings.

## Standalone validation protocol after training

Use each model’s frozen, validation-selected `best.pt`. The primary
contrast is A2 minus A1. A2 minus A0 is secondary. Follow the
[A0/A1 standalone protocol](a1_p2_detection.md). Use the original
230-image, 345-object validation split at 640, FP32, rectangular
batch 16, eight workers, native class-aware NMS with score floor
0.001, IoU 0.7, and at most 300 detections per image. Disable
test-time augmentation. Preserve D1 object identities, size groups
and fixed-point matching (confidence ≥0.25, IoU ≥0.50 and
background threshold 0.10).

Report native AP50 and AP50–95, per-class AP, fixed precision
and recall, original D1 size quartiles, recall at ≤16 px,
missed and localization errors, recall at shared FP/image budgets,
and detection-level contributions. For partial-box errors, reuse the
E1/E2 approach. For the same ground-truth object and fixed-point
same-class candidates, use the highest-IoU box and report IoU,
intersection / ground-truth area, predicted / ground-truth area,
and intersection / predicted area. Report absent candidates and the
support for each coverage statistic. Compare matched objects and
report coverage changes separately for recovered and lost objects.

Report all four rolled-pit objects and nine-class AP without rolled
pit as a sensitivity analysis. Keep ten-class AP primary. Use 5,000
paired original-image bootstrap draws with seed 42, shared image
multiplicities across conditions and the same class-support rules as
A0/A1. These intervals cover validation-image variation for fixed
single-seed checkpoints. Record same-GPU latency, throughput, and
allocated and reserved VRAM under the existing timing methodology
with repeated, counterbalanced passes for each pair.

Better coverage and fewer localization errors are the expected
signals, alongside preservation of small-defect recall and FP-budget
performance. Dilation can also overextend boxes, add background
responses, or cost latency at the dense P2 grid. If A2 succeeds, a
separately reviewed context-only head without P2 is needed to
isolate the independent effects of the two changes. The reviewed A2 ablation did not justify retaining A2 as the development architecture.

## Manual standalone submission

The [evaluation YAML](../../configs/evaluation/age_yolo_a0_a1_a2.yaml)
pins the three checkpoints and uses `a2-architecture-eval-01`. The
evaluator registers the custom head before loading checkpoints.
The A0/A1 accuracy exports are checked against the reviewed A1
archive and native metrics are recalculated from their AP records.
Only A2 runs accuracy validation. Timing reruns inference for all
three checkpoints on the same GPU in A0, A1, A2, A2, A1, A0 order.
The job records the GPU UUID and separate allocated and reserved
memory peaks. A1 remains the development reference until review.

On the management node, set the YAML device to the authorized logical
index before recording source. The current configuration uses
`cuda:6`, the logical index recorded by the completed A2 training run.

```bash
cd ~/flawtrace-gc10
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate flawtrace-gc10

cfg=configs/evaluation/age_yolo_a0_a1_a2.yaml
python scripts/record_source.py --architecture-eval --config "$cfg" &&
  PBS_O_WORKDIR="$PWD" bash submit_a2_eval.sh --preflight &&
  job=$(qsub submit_a2_eval.sh) &&
  qstat -f "$job"
```

```bash
tail -F "flawtrace-a2-eval.o${job%%.*}"
# After the run creates its persistent log
tail -F experiments/a2-architecture-eval-01/terminal.log
```

Every comparison has separate counts and bootstrap intervals in the
CSV outputs. The run also saves `partial_defect_analysis.csv` and
`partial_defect_per_object.csv`. These report coverage as well as
missing candidate counts, with paired changes restricted to objects
with a candidate under both compared checkpoints.

### Package for review

Run after successful completion. The archive contains the evaluation
directory, including native A2 artifacts, verified A0/A1 references,
plots and prediction exports. This local review package precedes the
formal A2 archive and publication.

```bash
(
  set -e
  cd ~/flawtrace-gc10
  run=a2-architecture-eval-01
  dst=.cache/a2-review
  job=$(python -c 'import json, sys; s=json.load(open(sys.argv[1])); assert s["status"] == "completed"; print(s["device"].get("pbs_job_id") or "")' \
    "experiments/$run/summary.json")
  mkdir -p "$dst"
  test ! -e "$dst/$run.tar"
  test ! -e "$dst/$run.tar.sha256"
  tmp=$(mktemp -d "$dst/.package.XXXXXXXX")
  trap 'rm -r -- "$tmp"' EXIT
  tar -cf "$tmp/$run.tar" "experiments/$run"
  if test -n "${job:-}" && test -f "flawtrace-a2-eval.o${job%%.*}"; then
    tar -rf "$tmp/$run.tar" "flawtrace-a2-eval.o${job%%.*}"
  fi
  tar -df "$tmp/$run.tar"
  (cd "$tmp" && sha256sum "$run.tar" > "$run.tar.sha256")
  ln "$tmp/$run.tar" "$dst/$run.tar"
  ln "$tmp/$run.tar.sha256" "$dst/$run.tar.sha256"
  (cd "$dst" && sha256sum -c "$run.tar.sha256")
)
```
