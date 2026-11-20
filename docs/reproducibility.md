# Reproducibility checklist

This checklist describes the workflow for future FlawTrace experiments.

## Before implementation

- Verify GC10-DET provenance and usage terms as described in the
  [data policy](../data/README.md).
- Review the [pinned AGE-YOLO reference](age_yolo_reference.md) for the paper,
  repository commit, license, checkpoints, and implementation discrepancies.
  Record any approved deviations and the checksum of any weights used.
- Establish hardware and software requirements before choosing dependency
  versions. Record Python, framework, dependencies, OS, GPU, driver, and CUDA
  versions in the eventual environment record.

## Before an authorized baseline run

- Version actual settings under `configs/`: dataset and split identities,
  model, initialization, seed, image size, batch size, epochs, optimizer,
  learning rate, augmentations, and other relevant upstream settings.
- Define validation metrics and protocol, preserve test separation, and record
  differences from the reference methodology before execution.
- Confirm CUDA and mixed precision compatibility with the environment checker.
  Explicitly select the CUDA device and verify model parameters, training
  images, and loss inputs are on that device when used. Fail clearly if CUDA
  is unavailable.
- Obtain explicit authorization before launching training or evaluation.

## Preparing and running experiments

The user starts and resumes full training runs manually in their local terminal
with no active AI session. Codex prepares the commands and does not launch full
training runs.

For each prepared experiment, provide exact, copyable commands for:

1. Activating the environment.
2. Checking CUDA availability.
3. Starting training with the approved configuration and run directory.
4. Monitoring GPU usage from another terminal with `nvidia-smi -l 1`.
5. Resuming an interrupted run from its last checkpoint.
6. Reviewing the saved metrics, logs, and artifacts after completion.

Use standard Python and Ultralytics functionality and standard terminal tools.
Keep live epoch and batch progress visible, including losses, GPU memory usage,
and validation metrics. Capture terminal output, including standard error, in
a persistent log without suppressing live progress. Preserve the training
process's exit status when logging through a shell pipeline.

Save the native Ultralytics last checkpoint for resume. Resume the same
interrupted run with the framework's native resume functionality so it can
restore training state. Append to the log or create a separate resume log in
the same run directory. Keep completed runs intact.

Training script implementation belongs to the authorized training milestone.
Use the standard terminal output and GPU tools for monitoring.

## GPU performance before full training

Preserve the official AGE-YOLO training settings as the reference. Assess
data loader worker counts against available CPU cores, host RAM, shared
memory, and GPU utilization. If hardware requires a change, record the
original and revised settings, the constraint, and the benchmark evidence.

During training preparation, obtain separate authorization for a short
training benchmark. Under the proposed full-run settings, record sampled GPU
utilization, peak VRAM, images processed per second, and wall-clock batch time
over a steady window after warmup. Estimate training time per epoch from
batch time and batches per epoch. State whether the estimate includes
validation and checkpoint time. Record the measurement method and effective
training settings, including worker count and AMP status.

Use native PyTorch and Ultralytics features and standard CLI tools. Resolve
the upstream AMP check's checkpoint requirements during training preparation
before any run. See the [environment record](environment.md) for current
hardware and compatibility evidence.

## For each future experiment

- Use a unique `experiments/<run_id>/` directory and refuse to overwrite a
  completed run. Generated contents are ignored by Git.
- Save the repository commit and working-tree changes, upstream commit,
  environment details, exact command, configuration, dataset manifest identity,
  logs, checkpoints, metrics, and failure or deviation records.
- Retain concise, reviewed provenance and result summaries in version-controlled
  documentation, with artifact locations and checksums where appropriate.
- Validate the baseline from actual evidence before choosing improvements.
  Distinguish observations, hypotheses, and unresolved questions.

## Decisions still required

Confirm the dataset source and split protocol, CUDA hardware and compatible
environment, baseline configuration, validation criteria, and durable storage
for ignored experiment artifacts. Verify checkpoint provenance if a checkpoint
will be used.

Create `scripts/` and `configs/` only when approved implementation work needs
them. Do not add placeholder code, speculative dependency pins, or infrastructure.
