# FlawTrace project rules

- Use simple, readable Python and standard coding conventions.
- Prefer official upstream implementations.
- Avoid unnecessary abstractions and dependencies.
- Keep changes narrowly scoped to the assigned task.
- Record dataset provenance, upstream commits, and configurations.
- Require CUDA for model training. Fail clearly if CUDA is unavailable; never silently fall back to CPU during training.
- Never overwrite completed experiment results. Use a new, unique run directory for each experiment.
- Do not launch training, evaluation, or other experiments without explicit authorization.
- Do not fabricate missing information or experimental results. Mark unknowns and unvalidated claims explicitly.
- Preserve existing work unless deletion is explicitly approved.

GC10-DET is the primary dataset and AGE-YOLO is the reference methodology.
Determine proposed improvements only after baseline validation. Keep the
implementation proportional to limited development time and compute.

Keep the AGE-YOLO submodule pinned. Record any approved change to its pin or
deviation from the reference implementation. Establish environment requirements
before selecting dependency versions. Do not commit or push without authorization.

## Experiment execution

- The user starts and resumes full training runs manually from their local
  terminal with no active AI session. Codex must never launch a full training run.
- Training scripts must show live epochs, batches, losses, GPU memory usage,
  and validation metrics. Preserve terminal output in a persistent log while
  keeping that progress visible in the terminal.
- Support resuming an interrupted run from its last checkpoint using
  Ultralytics' native resume functionality. Preserve existing logs and results.
- For each prepared experiment, provide exact commands to activate the
  environment, check CUDA availability, start training, monitor GPU usage,
  resume training, and review the completed results.
- Use standard Python and Ultralytics functionality and standard terminal tools.
  Do not build custom dashboards or monitoring systems. Defer training script
  implementation until the training milestone is authorized.
- Verify the GPU, VRAM, CUDA runtime, and mixed precision. Explicitly select
  CUDA for the model and tensors used in training. Assess data loader workers
  against available CPU cores, RAM, shared memory, and GPU utilization.
- Preserve the official AGE-YOLO training configuration. Document the hardware
  constraint and evidence for any required change.
- Before full training, obtain separate authorization for a short training
  benchmark. Record GPU utilization, peak VRAM, images per second, batch time,
  and estimated epoch duration. Provide `nvidia-smi -l 1` for live monitoring.
- Use the standard PyTorch and Ultralytics execution paths. Avoid experimental
  acceleration frameworks, custom CUDA kernels, and unnecessary optimizations.

Follow [the reproducibility workflow](docs/reproducibility.md) as later stages
are authorized. Keep local datasets, additional weights, generated outputs,
and credentials out of FlawTrace's Git history. Version reproducibility metadata.
