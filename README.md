# FlawTrace

## Project overview

FlawTrace studies industrial surface defect detection on GC10-DET using the
authors’ AGE-YOLO fork. B0 verifies the authors’ released checkpoint.
B1 is scientifically accepted as FlawTrace’s hardware-adapted, single-seed
AGE-YOLO baseline for milestone `v0.1-baseline`. The
[experiment registry](docs/experiments.md) records results and artifact identities.

## Repository structure

| Path | Purpose |
|---|---|
| `configs/hpc.yaml`, `configs/baseline/` | Editable execution settings, reference settings and B1 run snapshot |
| `configs/gc10*.yaml`, `data/manifests/` | Native dataset descriptors and versioned split identities |
| `scripts/`, `submit.sh`, `submit_eval.sh` | Data preparation, environment setup, training, evaluation and PBS execution |
| `upstream/AGE-YOLO/` | Pinned author implementation and official weights |
| `docs/` | Methods, evidence, environment and experiment records |
| `experiments/`, `logs/`, `artifacts/` | Locally generated outputs, excluded from Git |

## Environment installation

On the UPES management node, clone once and initialize the recorded submodule.

```bash
git clone --recurse-submodules https://github.com/miglanisatvik989/flawtrace-gc10.git
cd flawtrace-gc10
bash scripts/setup_hpc.sh
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate flawtrace-gc10
```

The setup pins Python 3.12.14, CUDA 12.8 PyTorch wheels and headless OpenCV.
The [environment record](docs/environment.md) describes the portable Linux
wheel lock and the HPC substitution.

## Dataset preparation

```bash
test -d data/gc10-det || python scripts/prepare_data.py
python scripts/check_data.py
test -d data/gc10-det-aug42-x4 || python scripts/augment_data.py
python scripts/augment_data.py --verify
```

The augmented training split has 7,336 images and 11,524 objects. Validation
uses the original 230 images and 345 objects. See the [dataset audit](docs/dataset.md).

## B0 — Official checkpoint verification

From an authorized GPU shell with the Conda environment active, set DEVICE to
the indexed CUDA device assigned within that shell. Choose an unused run ID.

```bash
python scripts/evaluate.py --official --checkpoint upstream/AGE-YOLO/weights/pretrained_gc10.pt \
    --split val --device "${DEVICE:?set assigned logical CUDA device}" --run-id b0-val-repeat-01
```

B0 verifies the released SHA-256 and uses `configs/gc10.yaml`. Each evaluation
saves overall and per-class metrics, plots, timings, environment and terminal
log under `experiments/<run-id>/`. [B0 protocol and results](docs/checkpoint_evaluation.md)

## B1 — Independent baseline reproduction

The [observed B1 configuration](configs/baseline/age_yolo_b1.yaml) records
random initialization, seed 42, batch 64 and deterministic offline augmentation.
The [reference configuration](configs/baseline/age_yolo_reference.yaml) records
batch 16. For another run, copy the B1 snapshot into the editable configuration,
choose a fresh RUN_ID and set DEVICE for the allocation. Keep the original
snapshot available for comparison.

```bash
cp configs/baseline/age_yolo_b1.yaml configs/hpc.yaml
nano configs/hpc.yaml
python scripts/train.py --config configs/hpc.yaml --preflight
python scripts/record_source.py
```

In an authorized interactive allocation use
`python -u scripts/train.py --config configs/hpc.yaml`. Alternatively use PBS
below. Review `results.csv`, the native plots, `args.yaml`, `effective_config.json`
and `weights/{best,last}.pt` in the run directory.

For an authorized repeat of B1 validation, run from the repository root with an unused run ID.

```bash
python scripts/evaluate.py --checkpoint experiments/hpc-full-06/weights/best.pt \
    --device "${DEVICE:?set assigned logical CUDA device}" --split val --run-id b1-val-03
```

For PBS, `qsub submit_eval.sh` reads DEVICE from configs/hpc.yaml and
uses b1-val-03 by default. Set EVAL_RUN_ID to an unused ID for subsequent
repeats. Monitor `_main_eval.log`.

This saves the same evaluation outputs as B0 and records the supplied checkpoint
digest. Test evaluation requires separate approval after model selection.

## Experimental results

| Experiment and measurement | Split | mAP50 | mAP50–95 |
|---|---|---:|---:|
| B0 released weights, standalone evaluation | Validation | 76.28% | 42.07% |
| B0 released weights, approved historical evaluation | Test | 72.76% | 40.93% |
| B1 epoch 84, training CSV | Validation | 69.59% | 34.22% |
| B1 best.pt, accepted standalone evaluation | Validation | 69.5288% | 34.1224% |

B1 selected epoch 84 and stopped after epoch 134. It used batch 64 instead
of the official 16 and a single seed (42). The exact launch-time source
revision is unknown. Independent B1 test evaluation remains outstanding.
[Experiment registry](docs/experiments.md)

D1 is complete and scientifically reviewed. Both checkpoints recalled
smaller-area defects less often on validation, and B1 had 13 localization
errors in the narrowest short-side quartile.
[Reviewed D1 report](docs/results/d1_error_analysis.md) · [Research plan](docs/research_plan.md)

E1 is complete and independently reviewed. At 960, validation mAP50–95
fell from 34.12% to 17.24%. E1A is also complete and scientifically
reviewed. Both fusion rules reduced AP and trailed the 640 control at every
evaluated false-positive budget. E2 matched-resolution training restored E1’s 960 degradation but yielded
only a +0.10-point mAP50–95 gain versus B1 at 640. E3 fixed tiled
inference is the next approved diagnostic.
[Reviewed E1 report](docs/results/e1_resolution.md) · [Reviewed E1A report](docs/results/e1a_fusion.md) · [Reviewed E2 report](docs/results/e2_training_resolution.md)

## HPC execution and recovery

After recording the current source, submit from the repository root.

```bash
job=$(qsub submit.sh)
qstat -f "$job"
tail -F _main_train.log
```

Per-run logs persist in `experiments/<run-id>/terminal.log`. After an
interruption, keep the run ID and experiment settings, change RESUME to true,
run `python scripts/record_source.py` again and submit. The trainer first tries
last.pt, then the recovery copy if the primary file is unreadable. See the
[HPC guide](docs/hpc.md) and [provenance workflow](docs/reproducibility.md).

## Citation and licensing

Credit the [AGE-YOLO paper](https://doi.org/10.1016/j.engappai.2026.114690),
[author implementation](https://github.com/gnmtdt/AGE-YOLO) and
[GC10 dataset release](https://huggingface.co/datasets/YYYHHHH993/AGE-YOLO).
The fork supplies an AGPL-3.0 license, and the release lists CC BY-NC 4.0 for
the dataset. FlawTrace authorship metadata and licensing for original
contributions await maintainer approval. [Attribution and provenance](docs/age_yolo_reference.md)
