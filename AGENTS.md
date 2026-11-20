# FlawTrace project rules

- Use simple, readable Python and standard coding conventions.
- Prefer official upstream implementations.
- Avoid unnecessary abstractions and dependencies.
- Keep changes narrowly scoped to the assigned task.
- Record dataset provenance, upstream commits, and configurations.
- Require CUDA for model training. Fail clearly if CUDA is unavailable; never silently fall back to CPU during training.
- Never overwrite completed experiment results. Use a new, unique run directory for each experiment.
- Do not launch training without explicit authorization.
- Do not fabricate missing information or experimental results. Mark unknowns and unvalidated claims explicitly.
- Preserve existing work unless deletion is explicitly approved.

GC10-DET is the primary dataset and AGE-YOLO is the reference methodology.
Determine proposed improvements only after baseline validation. Keep the
implementation proportional to limited development time and compute.

The current scope is repository foundation only. Do not implement training or
evaluation, download datasets, weights, or upstream repositories, or introduce
CI/CD, Docker, or experiment tracking services. Establish environment requirements
before selecting dependency versions. Do not commit or push without authorization.

Follow [the reproducibility workflow](docs/reproducibility.md) as later stages
are authorized. Keep dataset payloads, weights, generated outputs, and credentials
out of Git; retain reproducibility metadata in version control.
