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
- Confirm CUDA availability. Training must fail clearly if CUDA is unavailable;
  CPU fallback is forbidden.
- Obtain explicit authorization before launching training or evaluation.

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
