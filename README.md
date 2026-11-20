# FlawTrace

FlawTrace is a research project on industrial surface defect detection using
**GC10-DET**, with **AGE-YOLO** as the reference methodology. Proposed improvements
will be determined after baseline validation, within limited development time
and computational resources.

## Current status

The official AGE-YOLO implementation is pinned as a Git submodule at
`upstream/AGE-YOLO` (commit `99002665ff2f7faa5013c081b1b8f547ae4566ec`).
The authors' GC10-DET release has been audited with its original splits. The
official AGE-YOLO checkpoint scored **76.28% mAP50 on validation** and
**72.76% on test**, with mAP50–95 of **42.07%** and **40.93%**, respectively.
These are checkpoint verification results. Independent training reproduction
remains pending. The independent baseline is prepared for a manually run
benchmark. [Evaluation report](docs/checkpoint_evaluation.md)

Initialize the submodule from the repository root:

```sh
git submodule update --init --checkout -- upstream/AGE-YOLO
git submodule status -- upstream/AGE-YOLO
```

The first command checks out the commit recorded by FlawTrace. The submodule
includes the checkpoint files and demo images tracked by upstream. See the
[AGE-YOLO reference notes](docs/age_yolo_reference.md) for their provenance,
licensing questions, and upstream script discrepancies.

## CUDA environment

Use CPython 3.12.14 and uv (tested with 0.12.5) on Linux x86-64 with an NVIDIA
GPU and a driver compatible with CUDA 12.8. After initializing the submodule,
run these commands from the repository root:

```sh
uv venv --python 3.12.14 --no-python-downloads .venv
uv pip sync --python .venv/bin/python \
  --index https://download.pytorch.org/whl/cu128 \
  --default-index https://pypi.org/simple \
  --index-strategy unsafe-first-match requirements.txt
uv pip check --python .venv/bin/python
.venv/bin/python scripts/check_env.py
```

The checker imports Ultralytics from the pinned submodule and uses the official
model YAML and a synthetic input. See the [environment record](docs/environment.md)
for the tested host, dependency choices, and smoke-test results.

## GC10-DET setup

After initializing the submodule and installing the Python environment above,
make `bsdtar` available on PATH (`libarchive-tools` on Ubuntu), then run
from the repository root:

```sh
.venv/bin/python scripts/prepare_data.py
.venv/bin/python scripts/check_data.py
```

The preparation script fetches only the GC10 portion of the authors' pinned
archive. See the [dataset record](docs/dataset.md) for usage terms, provenance,
and integrity findings.

## Official checkpoint evaluation

From the repository root, verify the environment, manifests, and bundled
checkpoint. The checkpoint smoke check uses a single synthetic input.

```sh
source .venv/bin/activate
python scripts/check_env.py
python scripts/check_data.py
checkpoint=upstream/AGE-YOLO/weights/pretrained_gc10.pt
python scripts/evaluate.py --checkpoint "$checkpoint" --smoke-test
```

Run the requested split manually. Use validation for development and reserve
the test command for a separately approved final check. Choose a new run ID
for each attempt, including retries of failed runs.

```sh
python -u scripts/evaluate.py --checkpoint "$checkpoint" --split val --run-id age-yolo-gc10-val-01
# When test evaluation is approved:
python -u scripts/evaluate.py --checkpoint "$checkpoint" --split test --run-id age-yolo-gc10-test-01
```

Monitor GPU usage in a separate terminal:

```sh
nvidia-smi -l 1
```

The wrapper verifies the official checkpoint SHA-256 before loading it.
It uses the bundled validator on `cuda:0` with image size 640, FP32,
rectangular batching, batch size 16, eight workers, confidence threshold
0.001, NMS IoU 0.7, at most 300 detections per image, and no test-time
augmentation. Those settings come from the pinned `test.py` and the fork's
validation defaults. The wrapper selects the split explicitly and uses a copy
of `configs/gc10.yaml` in a temporary tree with the manifest-verified images
and labels for that split. This keeps the validator's caches and any JPEG
repair away from the original data. Native plots use a local Matplotlib font.

Each run writes live progress to the terminal and `terminal.log`:

```text
experiments/<run-id>/
├── terminal.log
├── run.json                      # checkpoint and dataset identities, run status
├── environment.json
├── effective_config.json         # actual arguments from the validator
├── gc10.yaml
├── gc10-provenance.json
├── gc10-audit.json
├── gc10-<split>.csv
├── metrics.json
├── per_class.csv
├── timing_ms_per_image.json
└── plots/                        # native PR/confidence curves and confusion matrices
```

Review the validation run from the repository root:

```sh
run=experiments/age-yolo-gc10-val-01
cat "$run/run.json" "$run/metrics.json" "$run/timing_ms_per_image.json"
cat "$run/per_class.csv"
ls "$run/plots"
tail -n 30 "$run/terminal.log"
```

Use `experiments/age-yolo-gc10-test-01` to review the test run. Empty AP
cells mean the fork did not return a class score. If it finds no true
positives, it may omit the PR curve. Both completed runs produced the PR curve
and confusion matrices. Use validation for future tuning and model selection.
The recorded test run is a fixed checkpoint verification result.
[Reviewed metrics and limitations](docs/checkpoint_evaluation.md)

## Independent AGE-YOLO training

From the repository root, verify the environment and original data. Prepare
the derived dataset once, then verify its byte reproducibility:

```sh
source .venv/bin/activate
python scripts/check_env.py
python scripts/check_data.py
test -d data/gc10-det-aug42-x4 || python -u scripts/augment_data.py
python -u scripts/augment_data.py --verify
python scripts/train.py --mode check-amp
```

Run the short benchmark manually after separate authorization. Use a new ID
for each attempt. This run stops after 60 training batches.

```sh
python -u scripts/train.py --mode benchmark --run-id age-yolo-gc10-bench-01
cat experiments/benchmarks/age-yolo-gc10-bench-01/benchmark.json
tail -n 10 experiments/benchmarks/age-yolo-gc10-bench-01/gpu.csv
```

The [benchmark diagnostics](docs/benchmark_diagnostics.md) document the
original run, an A/B synchronization check, and the warmed native
batch-32 validation memory probe.

Monitor the GPU in another terminal:

```sh
nvidia-smi -l 1
```

After reviewing benchmark results and authorizing full training, launch from
your local terminal without an active AI session:

```sh
python -u scripts/train.py --mode full --run-id age-yolo-gc10-baseline-01
```

If interrupted after a checkpoint was saved, resume the same run ID:

```sh
python -u scripts/train.py --mode full --run-id age-yolo-gc10-baseline-01 --resume
```

Review the saved validation metrics, provenance, and checkpoints:

```sh
run=experiments/age-yolo-gc10-baseline-01
cat "$run/run.json" "$run/effective_config.json"
tail -n 6 "$run/results.csv"
tail -n 30 "$run/terminal.log"
ls -lh "$run/weights/best.pt" "$run/weights/last.pt"
```

The [baseline method](docs/baseline.md) records the augmentation assumptions,
training settings, AMP check, and the remaining hardware checks. Use
validation alone for checkpoint and model selection.

## Planned reproducibility workflow

1. Verify dataset provenance, terms, integrity, class mapping, and split protocol.
2. Initialize the pinned upstream implementation and review its reference notes.
3. Recreate the pinned environment and run the CUDA checker.
4. Version the baseline configuration and validation protocol before execution.
5. Train only with explicit authorization and available CUDA, preserving each run
   in a unique directory with its configuration, logs, and provenance.
6. Validate the baseline before selecting improvements; record actual outcomes,
   limitations, and deviations.

See [project rules](AGENTS.md), [data policy](data/README.md), and the
[reproducibility checklist](docs/reproducibility.md).

## Layout

- `data/`: data policy and audit manifests. Local dataset payloads are ignored by Git.
- `docs/`: reproducibility requirements and the AGE-YOLO source review.
- `upstream/AGE-YOLO`: pinned official reference implementation.
- `scripts/`: environment and dataset checks, augmentation, evaluation, and training.
- `configs/`: GC10-DET and AGE-YOLO baseline configuration.

Keep local datasets, additional weights, generated experiment outputs, caches,
virtual environments, and credentials out of FlawTrace's Git history.
