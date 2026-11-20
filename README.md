# FlawTrace

FlawTrace is a research project on industrial surface defect detection using
**GC10-DET**, with **AGE-YOLO** as the reference methodology. Proposed improvements
will be determined after baseline validation, within limited development time
and computational resources.

## Current status

Repository foundation only. No dataset, upstream implementation, environment,
training configuration, or experimental results have been established in this
repository. Dependency versions and baseline settings remain undecided.

## Planned reproducibility workflow

1. Verify dataset provenance, terms, integrity, class mapping, and split protocol.
2. Identify the official upstream implementation and record its exact commit.
3. Establish and record a compatible environment, including CUDA requirements.
4. Version the baseline configuration and validation protocol before execution.
5. Train only with explicit authorization and available CUDA, preserving each run
   in a unique directory with its configuration, logs, and provenance.
6. Validate the baseline before selecting improvements; record actual outcomes,
   limitations, and deviations.

See [project rules](AGENTS.md), [data policy](data/README.md), and the
[reproducibility checklist](docs/reproducibility.md).

## Layout

- `data/`: data policy; local dataset payloads are ignored by Git.
- `docs/`: reproducibility requirements and future documented decisions.
- `scripts/` and `configs/`: deferred until authorized work produces actual
  scripts and configuration files; no placeholders are included.

Datasets, weights, generated experiment outputs, caches, virtual environments,
and credentials must remain outside version control.
