# Reproducibility and review

The [README](../README.md) contains executable steps. The [registry](experiments.md)
assigns IDs and distinguishes released checkpoints, independent training and
held-out evaluation. Review the variant, data, seed and validation selection
criterion before each experiment.

On the management node, run `python scripts/record_source.py` after editing the
configuration and before submitting or resuming. Keep the deployment and its
submodule fixed while queued or running. This captures HEAD, working-tree
status, the tracked diff, untracked source text, deployed file digests and the
actual submodule revision in an ignored local record. Review any untracked
files first. For a named training YAML, pass `--config` with the same path to source
capture and training. Training checks the selected YAML, model and data
configuration, and required runner files. Other recorded files remain
available for provenance. Git is needed only on the management node.

Each attempt records the command, PBS ID if present, start time, source and
dataset identity, augmentation metadata, runtime versions, GPU details and
initial free memory. On a normal return or handled exception it records
elapsed time and final status. A hard kill can leave an attempt marked
started. Use PBS history to reconcile that case. Training stores resolved
native arguments, results CSV and checkpoint digests at completion.

Preserve the entire per-run directory, including source records, plots,
logs, results and weights. Keep a checksum-verified archive outside the
checkout and arrange storage in a second failure domain. Publish only reviewed
summaries and identifiers in Git. The B1 archive and its limitations are
documented in the registry.

Select settings and checkpoints with validation data only. A separate user
decision authorizes any final test evaluation. Review and approve baseline
results before creating the v0.1-baseline milestone.
