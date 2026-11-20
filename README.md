# FlawTrace

FlawTrace is a research project on industrial surface defect detection using
**GC10-DET**, with **AGE-YOLO** as the reference methodology. Proposed improvements
will be determined after baseline validation, within limited development time
and computational resources.

## Current status

The official AGE-YOLO implementation is pinned as a Git submodule at
`upstream/AGE-YOLO` (commit `99002665ff2f7faa5013c081b1b8f547ae4566ec`).
The authors' GC10-DET release has been acquired and audited with its original
splits. The FlawTrace baseline configuration and results are pending.

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
- `scripts/`: environment and dataset setup and checks.
- `configs/`: GC10-DET dataset configuration.

Keep local datasets, additional weights, generated experiment outputs, caches,
virtual environments, and credentials out of FlawTrace's Git history.
