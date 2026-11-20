# FlawTrace experiment registry

B0 and B1 are permanent identifiers. P1 and A1 onward are reserved for
proposed methods and ablations. Use validation alone for model selection.

## B0 — Released checkpoint verification

Completed. Runs `age-yolo-gc10-val-01` and `age-yolo-gc10-test-01` used the
bundled weights with SHA-256
`2daca14beedf36d6f7f74bca018292239a8eeb398fbe5b1c57516e410a61dc49`,
`configs/gc10.yaml`, and pinned upstream `99002665ff2f7faa5013c081b1b8f547ae4566ec`.
HEAD was `28c84eabec80631311509fb94e6e4b8ccb205e62` plus the working-tree
evaluator identified by digest in the [B0 report](checkpoint_evaluation.md).
That report records both split manifest hashes, hardware (RTX 5050 Laptop),
environment, metrics, timings, per-class AP, warnings and limitations. The
historical run directories under `experiments/` were recorded on the original
evaluation host. Their payloads were unavailable in this HPC checkout.

| Split | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| Validation | .71109 | .75881 | .76279 | .42071 |
| Approved historical test | .73293 | .66721 | .72765 | .40935 |

## B1 — Independent baseline reproduction

**Scientifically accepted for milestone `v0.1-baseline`.** B1 is the
hardware-adapted, single-seed AGE-YOLO baseline. Standalone validation of
the selected `best.pt` in `b1-val-02` yielded **69.5288% mAP50** and
**34.1224% mAP50–95**. Run
`experiments/hpc-full-06/` used the [saved run configuration](../configs/baseline/age_yolo_b1.yaml).
The actual args and effective settings are archived with the results. The
release dataset revision is `93d2bc4420b024d3cbd825610ad7aad7b2544311`.
The AUG42 train manifest SHA-256 is
`c68282dc8c3ac5ed68c46f3a3253eaa6782b0acb921b37e5cbbf24a5587264a7`.
Validation uses manifest `c48bbe089cd3813bde9737d7e26672bf1f0ba0d6c9ea5b6839ef8eeb0d443472`.
The run recorded the same pinned upstream commit as B0. FlawTrace HEAD was
observed at `1b87ad8` with local changes during training. There was no
launch-time source snapshot, so the exact executed revision is unrecoverable.

PBS job 47708.mgmt01 finished on gpu02 with exit code 0 and walltime
1:09:50. The trainer reported 1.152 hours, H100 80GB HBM3, logical cuda:5,
22.31 GiB initially free of 79.18 GiB, PyTorch 2.7.1+cu128, CUDA 12.8 and
bundled Ultralytics 8.3.9. The run did not record the physical GPU UUID.
Batch 64 (official 16), validation batch 128, accumulation 1 and decayed
group weight decay .0005 were recorded. This is one seed (42) with the
project’s reconstruction of the authors’ offline augmentation.

Early stopping selected epoch 84 and stopped after epoch 134 of 300.
First-to-last training box/class/DFL losses fell from 3.774/5.321/4.001
to .835/.588/.984. Validation fitness stopped improving for 50 epochs.
The log contains deprecation warnings and the native default scale n
message. It records successful final validation and normal completion.

| Validation measurement | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| Epoch 84 CSV | .66158 | .69717 | .69587 | .34216 |
| Final native validation of best.pt, rounded log | .644 | .715 | .696 | .342 |
| Epoch 134 CSV | .67870 | .68251 | .66860 | .32401 |
| Standalone best.pt, b1-val-02 | .64328 | .71107 | .69529 | .34122 |

PBS job 47713.mgmt01 (gpu01, H100, cuda:0) exited successfully. The evaluator
verified 230 validation images and 345 objects. Full-precision metrics,
per-class AP, plots and timing are in `experiments/b1-val-02/`. The preceding
failed CUDA startup is preserved in `experiments/b1-val-01/`. Independent B1
test evaluation remains outstanding. Single-seed variation and the batch change preclude a claim
of reproducing the authors’ multi-run mean.

| Preserved artifact | SHA-256 |
|---|---|
| weights/best.pt | `7e61fc3da770652acfd58264c06b31480e04b6ac6440e79e3eae90f648c92275` |
| weights/last.pt | `3c6b4cd00ade7d1412b73cd7ba0176602cb4655243e8e599e088d6a384e0a2a0` |
| results.csv | `25b39b409283a20cb9b3ea24eb10c797a24ab995cf3b85d6f72cf7aabb7b9d18` |
| Full HPC archive | `2973d818413fb3c7fbbc1300880e575c0826145e881e039241741b71eb76a37e` |

The archive at `artifacts/B1/hpc-full-06.tar` contains the whole
run including plots, effective arguments, model and data configuration,
environment records, training log, recovery weights and the launcher log.
Verify with `(cd artifacts/B1 && sha256sum -c SHA256SUMS)`.
The companion archive `b1-validation.tar` preserves both evaluation attempts
and launcher logs. Verify both archives from the repository root using
the command above. These copies reside on the HPC home filesystem. Off-filesystem backup and
institutional retention policy have yet to be confirmed.

The [file-level SHA-256 inventory](b1-artifacts.sha256) covers B1 weights,
plots, configuration, environment, CSV and terminal log. Verify it from
the original run directory with `sha256sum -c /path/to/docs/b1-artifacts.sha256`.
