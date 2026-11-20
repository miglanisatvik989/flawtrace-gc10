# FlawTrace

FlawTrace is a research project on industrial surface defect detection using
**GC10-DET**, with **AGE-YOLO** as the reference methodology. Proposed improvements
will be determined after baseline validation, within limited development time
and computational resources.

## Current status

The official AGE-YOLO implementation is pinned as a Git submodule at
`upstream/AGE-YOLO` (commit `99002665ff2f7faa5013c081b1b8f547ae4566ec`).
The GC10-DET dataset protocol, CUDA environment, and FlawTrace baseline
configuration are pending. No FlawTrace baseline results are available.

Initialize the submodule from the repository root:

```sh
git submodule update --init --checkout -- upstream/AGE-YOLO
git submodule status -- upstream/AGE-YOLO
```

The first command checks out the commit recorded by FlawTrace. The submodule
includes the checkpoint files and demo images tracked by upstream. See the
[AGE-YOLO reference notes](docs/age_yolo_reference.md) for their provenance,
licensing questions, and upstream script discrepancies.

## Planned reproducibility workflow

1. Verify dataset provenance, terms, integrity, class mapping, and split protocol.
2. Initialize the pinned upstream implementation and review its reference notes.
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
- `docs/`: reproducibility requirements and the AGE-YOLO source review.
- `upstream/AGE-YOLO`: pinned official reference implementation.
- `scripts/` and `configs/`: deferred until authorized work produces actual
  scripts and configuration files; no placeholders are included.

Keep local datasets, additional weights, generated experiment outputs, caches,
virtual environments, and credentials out of FlawTrace's Git history.
