# E2 matched-resolution training

Completed and scientifically reviewed, with mixed results. Run
`e2-train-960-b16-02` preserved its historical ID and used physical batch
32, accumulation 2, and 960-pixel training. It selected epoch 78 and
completed normally after 128 epochs. Batch 64 was reduced after prior
CUDA OOMs. Standalone evaluation `e2-resolution-eval-01` used the same
manifest-verified 230 validation images and 345 objects for every
condition. A and B reused the E1 predictions.
[Training evidence and protocol](../experiments/e2_training_resolution.md)

| Condition | Train size | Inference size | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| A · B1 | 640 | 640 | 69.53% | 34.12% |
| B · B1 | 640 | 960 | 41.10% | 17.24% |
| C · E2 | 960 | 640 | 56.53% | 24.56% |
| D · E2 | 960 | 960 | 67.99% | 34.23% |

In the primary comparison D–A, mAP50 changed by **−1.54 points** and
mAP50–95 by **+0.10 points**, using unrounded values. The paired 95%
interval for mAP50–95 was **−2.57 to +2.49 points**. E2's point
estimate fell short of the proposed two-point gain.

| Fixed-point diagnostic | A | D |
|---|---:|---:|
| Precision | 67.72% | 72.67% |
| Recall | 68.12% | 67.83% |
| False positives | 112 | 88 |
| Localization errors | 30 | 26 |
| Q1 short-side recall | 58.24% | 59.34% |

| FP/image budget | A recall | D recall |
|---|---:|---:|
| 0.10 | 45.80% | 57.97% |
| 0.25 | 62.61% | 66.09% |
| 0.50 | 68.99% | 70.72% |
| 1.00 | 75.94% | 73.62% |

All four conditions missed **77 objects** at the fixed operating point.
These included 26 silk spots, 19 inclusions, 13 creases, 10 oil spots,
and nine from other classes.

Both checkpoints performed substantially better at their training
resolution. E2 recovered the severe 960 performance degradation seen
in E1, improved precision and recall at stricter FP budgets, and
reached roughly the same mAP50–95 as B1 at 640. Physical training batch
changed from 64 to 32 and accumulation changed from 1 to 2. The
across-checkpoint comparison cannot isolate training resolution. All four
bootstrap comparisons completed 5,000 paired validation-image draws.
Those intervals quantify image-level uncertainty for these checkpoints.
Training-seed variation remains unmeasured.

The original training and evaluation runs remain under `experiments/`.
Both complete archives and their verified SHA-256 records are under
`artifacts/E2/`. The next approved step is E3 fixed tiled inference.
[Research roadmap](../research_plan.md)
