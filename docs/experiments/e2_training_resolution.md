# E2 matched-resolution training at 960

Training and standalone validation completed and were scientifically reviewed. [Reviewed E2 results](../results/e2_training_resolution.md). E2 asks whether training AGE-YOLO at the
target inference resolution improves its 960 performance and mitigates
the localization degradation observed in E1. The hypothesis is that
reducing the training/inference size mismatch improves fixed-point
small-defect recall and localization. E1A's negative fusion result
motivates testing this change before introducing tiling.
[Reviewed E1](../results/e1_resolution.md) · [Reviewed E1A](../results/e1a_fusion.md)

## Completed training

The preserved directory is `experiments/e2-train-960-b16-02/`. Its name
contains b16, but the saved YAML, effective configuration and native
arguments all show **batch 32**, with **accumulation 2** and native validation
batch 64. B1 used physical batch 64 and accumulation 1. The batch-64 E2
attempts logged CUDA out-of-memory failures, prompting the reduction in
physical batch size. This run started from random weights, with seed 42,
960-pixel training, AMP and patience 50.

PBS job 47802.mgmt01 exited successfully on gpu02. The run stopped at epoch
128 after 50 epochs without improvement, selected epoch 78, and took
approximately 2.47 hours. The epoch-78 training CSV records **68.149%
mAP50** and **34.445% mAP50–95**. Those training-time validation scores
are distinct from the prepared standalone evaluation.

Checkpoint `weights/best.pt` SHA-256
`ed58d685cdc5b2cd5d3f6db800eb5101ecc73b0a960bc7235ac77ddb1be69f33`.
The verified training archive is `artifacts/E2/e2-train-960-b16-02.tar`.
The original directory and earlier attempts remain in place.

Physical batch and accumulation affect the optimization trajectory.
The across-checkpoint comparisons cannot isolate the causal effect of
training resolution. Inference-size comparisons within one checkpoint
hold training fixed. A single training seed limits generalization.

## Original batch-64 plan

Start from the same pinned AGE-YOLO architecture with fresh weights and
seed 42. Use the unchanged AUG42 dataset and validation-based checkpoint
selection. The initial plan used the same trainer, AMP, deterministic mode, batch 64,
eight requested workers, maximum 300 epochs, patience 50, SGD, learning
rate 0.01, momentum 0.937, weight decay 0.0005, and all offline and online
augmentation settings. The nominal batch is 64. The fork yields
gradient accumulation 1, the same scaled weight decay, and validation
batch 128. Retain the same loss and learning-rate defaults.

Comparing the archived B1 effective arguments with the initial E2
configuration resolved by the fork found only image size, logical CUDA
index, and run name/save directory differences. B1 recorded cuda:5 on
its allocation. E2 defaults to cuda:0. Select the authorized logical
index and use one authorized H100. The only intended methodological
change is training image size, from 640 to 960.
[E2 configuration](../../configs/training/age_yolo_e2.yaml)

The pinned loader's resize target, square training canvas, mosaic and
affine geometry, and rectangular validation batch shapes depend on
image size. Their pixel dimensions change with the image-size setting.
The normalized transform settings, mosaic close policy (10 epochs),
disabled multi-scale mode, and schedule defaults are unchanged.
The wrapper's separate 640-pixel AMP compatibility probe is unchanged.
It does not assess 960 training memory.

## Stored data and memory

Training reuses 7,336 images and 11,524 objects. Validation reuses 230
images and 345 objects. The offline augmentation rescaled and re-encoded
source images before training, so some derived inputs were already
downsampled relative to their originals. Both runs use those same bytes.
The pinned loader resizes the stored image relative to its longest side
before augmentation or stride padding.

| Stored split or subset | Images | Stored dimensions |
|---|---:|---|
| Original images in AUG42 train | 1,834 | 2048 × 1000 |
| Offline variants in AUG42 train | 5,502 | Long side 1,638–2,457 px. Short side 800–1,199 px. |
| Validation | 230 | 2048 × 1000 |

Every stored image has a long side above 960. The initial 960 resize is
therefore a downsample from the stored bytes and retains more available
spatial detail than the 640 resize. Some offline variants have already
lost information through their own resize and JPEG encoding. Mosaic
and affine operations can crop or resample the initially resized image
again.

At batch 64, a 960 square has 2.25 times as many pixels as a 640 square.
Expect higher activation memory and compute. Native validation also
requests batch 128 and uses rectangular shapes with AMP, so its peak can
differ from training. B1's progress display peaked at 15.7 GiB reserved.
Multiplying that reading by 2.25 gives about 35.3 GiB as a rough scale
reference. Allocator reservation, kernels, workspace, padding and
validation can vary. This is not a measured E2 peak or an OOM guarantee.
The management host had no accessible CUDA device or authorized H100
session, so no CUDA feasibility run was performed. Preserve batch,
precision and accumulation if the first authorized run hits OOM and
review the failure before another attempt.

## Validation plan and success criteria

| Training checkpoint | Inference at 640 | Inference at 960 |
|---|---|---|
| B1, trained at 640 | Existing E1 control | Existing E1 challenger |
| E2, trained at 960 | C · completed | D · completed |

After training, evaluate the E2 best checkpoint at both resolutions on
the same validation images. Use the existing standalone evaluator and
prediction export path with batch 16, FP32, rectangular batching, eight
workers, confidence floor 0.001, class-aware NMS IoU 0.7, maximum 300
detections, and test-time augmentation disabled. Reuse D1 object IDs
and fixed 640-derived area, short-side and aspect-ratio boundaries. Keep
D1's fixed operating point and compare recall under shared FP/image
budgets. Review localization errors and per-class results separately.
Only validation is in scope.

The primary comparison is E2 at 960 versus B1 at 640. Look for higher
mAP50–95 and Q1 recall, with paired image-bootstrap intervals supporting
the gains and without worse overall localization error counts. Report
false-positive and latency tradeoffs even when the primary comparison
improves. Compare E2 at 960 with B1 at 960 to examine matched-resolution
training, and compare 640 versus 960 within E2 to examine inference
resolution. One seed and one validation split limit generalization.
Training-time validation also changes resolution, so checkpoint
selection can differ even with a shared policy.

## Historical batch-64 setup and provenance

| Input | SHA-256 |
|---|---|
| Frozen B1 configuration | `2c483a2ef6efd519ee05fb9d8c8450333fe4e768fbcc6aa6514b678ade182f3f` |
| Initial E2 YAML | `a51a90c9a8abea41291db0c6b22144f8b2db24ca7f54df66170811c208447e41` |
| AUG42 training manifest | `c68282dc8c3ac5ed68c46f3a3253eaa6782b0acb921b37e5cbbf24a5587264a7` |
| Validation manifest | `c48bbe089cd3813bde9737d7e26672bf1f0ba0d6c9ea5b6839ef8eeb0d443472` |

The original training manifest also verified at
`f9d7ca03ea1215117156d2ebc11dff34f71436038b72119abfc7b9ca9f83ab2f`.

The pinned upstream commit is `99002665ff2f7faa5013c081b1b8f547ae4566ec`.
Run source capture after any configuration edit. The trainer saves the
source commit and working-tree changes, config snapshot, dataset
identities, runtime versions, device name and UUID, job ID, seed,
native arguments and checkpoint digests. The checked environment uses
Python 3.12.14, PyTorch 2.7.1+cu128 and torchvision 0.22.1+cu128.

The commands below describe the earlier batch-64 plan. For the completed checkpoint, use the standalone evaluation commands in the next section.

```bash
cd /home/satvik.14173/flawtrace-gc10
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate flawtrace-gc10
mkdir -p logs
cfg=configs/training/age_yolo_e2.yaml
log=logs/e2-train-960-01-launch.log
python scripts/record_source.py --config "$cfg" &&
python scripts/train.py --config "$cfg" --preflight --check-source &&
job=$(qsub -v "TRAIN_CONFIG=$cfg,TRAIN_LOG=$log" submit.sh) &&
printf '%s\n' "$job" &&
qstat -f "$job"
tail -F "$log"
```

After startup, monitor `experiments/e2-train-960-01/terminal.log`. For
resume, wait until the prior job exits and use a configuration identical
to the actual run snapshot except for RESUME. Verify both checkpoints
before retrying if the earlier failure involved disk I/O. A run without
a usable checkpoint requires a new run ID.

```bash
cd /home/satvik.14173/flawtrace-gc10
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate flawtrace-gc10
run=experiments/e2-train-960-01
cfg=configs/training/age_yolo_e2_resume.yaml
log=logs/e2-train-960-01-launch.log
(test -s "$run/weights/last.pt" || test -s "$run/weights/last.recovery.pt") &&
( set -o noclobber; sed 's/^RESUME: false$/RESUME: true/' "$run/hpc.yaml" > "$cfg" ) &&
python scripts/record_source.py --config "$cfg" &&
python scripts/train.py --config "$cfg" --preflight --check-source &&
job=$(qsub -v "TRAIN_CONFIG=$cfg,TRAIN_LOG=$log" submit.sh) &&
printf '%s\n' "$job" &&
qstat -f "$job"
tail -F "$run/terminal.log"
```

Reuse an existing resume YAML after confirming it matches the snapshot.
The wrapper refuses completed runs and checks native checkpoint settings.

## Standalone evaluation

Run `e2-resolution-eval-01` from the canonical checkout. A and B reuse
E1's 640 and 960 predictions and AP records. C and D run E2's `best.pt`
at 640 and 960 in one PBS job. The job records the physical GPU UUID.
Use E1 timing for direct B1 comparisons only if GPU identity and timing
conditions are comparable.

The primary contrast is D–A. D–B holds inference size at 960, and C–A
holds it at 640. Both across-checkpoint contrasts include the training
batch change. D–C isolates the inference-size change for E2. All four
contrasts use 5,000 paired image bootstrap draws with seed 42. These
intervals do not measure variability across training seeds.

For an authorized repeat, choose a new run ID and set DEVICE in the evaluation YAML to the authorized logical CUDA device.
Run on the login node from the repository root:

```bash
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate flawtrace-gc10
python scripts/evaluate_e2.py --preflight
python scripts/record_source.py
job=$(qsub submit_e2_eval.sh)
qstat -f "$job"
tail -F experiments/e2-resolution-eval-01/terminal.log
```

PBS startup output goes to `flawtrace-e2.o<job-number>`. Both E2 native
runs, per-object coverage, error counts, FP-budget recall, paired
intervals, timing, plots and examples are saved in the evaluation run
directory. A CUDA OOM exits the job with the configured batch and precision.
