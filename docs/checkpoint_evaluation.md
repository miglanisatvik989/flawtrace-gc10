# Official AGE-YOLO checkpoint evaluation

Both manual evaluations completed all 15 batches. The validation run logged
230 images and 345 objects, and the test run logged 230 images and 337 objects.
Both runs saved complete metric tables, timing, precision-recall curves,
and raw and normalized confusion matrices.

## Method and provenance

- The bundled Ultralytics 8.3.9 validator evaluated the authors' GC10
  checkpoint from AGE-YOLO commit `99002665ff2f7faa5013c081b1b8f547ae4566ec`.
  Both recorded checkpoint hashes match the bundled file and the
  [upstream reference record](age_yolo_reference.md).
  SHA-256 `2daca14beedf36d6f7f74bca018292239a8eeb398fbe5b1c57516e410a61dc49`.
- Both runs used the [authors' GC10-DET release](dataset.md) at revision
  `93d2bc4420b024d3cbd825610ad7aad7b2544311`. The saved configuration,
  provenance, audit, and split manifests match their recorded SHA-256
  values and the repository copies. The [validation manifest](../data/manifests/gc10-val.csv)
  hashes to `c48bbe089cd3813bde9737d7e26672bf1f0ba0d6c9ea5b6839ef8eeb0d443472`.
  The [test manifest](../data/manifests/gc10-test.csv) hashes to
  `84a45cc1516015f2170b8f49147f0e9823f1b85d5329ca50946755d9b95bd890`.
- Each run evaluated manifest-verified copies of its split using
  [`configs/gc10.yaml`](../configs/gc10.yaml), leaving the original data
  intact. Configuration and environment records agree across the runs
  apart from split and run paths. The validator used CUDA device 0, FP32,
  nominal image size 640, rectangular batching, batch size 16, eight workers,
  confidence threshold 0.001, class-aware NMS IoU 0.7, at most 300 detections
  per image, and no test-time augmentation. The shared configuration's
  `amp: true` field is a training default. Standalone evaluation used
  `half: false`.
- The run records identify FlawTrace HEAD as
  `28c84eabec80631311509fb94e6e4b8ccb205e62`. The evaluator was still a
  working-tree file. Its recorded SHA-256 matches the version included in
  this milestone,
  `f155079e2bb962e9ce0bf326a092b8e228eff4e6453dc1822aa2078b7fe12589`.

## Results

Detection metrics from `metrics.json`, as proportions:

| Split | Images | Objects | Precision | Recall | mAP50 | mAP50–95 | Fitness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 230 | 345 | 0.7110912001051468 | 0.7588069290084012 | 0.7627876396046485 | 0.42070577742096765 | 0.45491396363933573 |
| Test | 230 | 337 | 0.7329266447589372 | 0.6672132261166135 | 0.7276456319923141 | 0.40934787060716166 | 0.4411776467456769 |

Per-class AP from `per_class.csv`, as percentages rounded to two decimals:

| ID | Class | Val AP50 | Val AP50–95 | Test AP50 | Test AP50–95 |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | `punching_hole` | 98.89 | 52.77 | 95.76 | 61.91 |
| 1 | `welding_line` | 91.52 | 40.25 | 89.80 | 40.62 |
| 2 | `crescent_gap` | 89.24 | 59.09 | 93.82 | 62.74 |
| 3 | `water_spot` | 95.88 | 57.22 | 91.22 | 52.84 |
| 4 | `oil_spot` | 81.80 | 42.67 | 77.20 | 36.90 |
| 5 | `silk_spot` | 66.75 | 33.81 | 63.35 | 27.69 |
| 6 | `inclusion` | 45.84 | 20.22 | 47.42 | 21.10 |
| 7 | `rolled_pit` | 63.49 | 47.13 | 28.00 | 13.54 |
| 8 | `crease` | 37.84 | 18.86 | 43.92 | 29.99 |
| 9 | `waist_folding` | 91.53 | 48.69 | 97.16 | 62.02 |

The [AGE-YOLO paper](https://doi.org/10.1016/j.engappai.2026.114690) reports
72.8% mAP50 on GC10-DET. The [authors' comparison table at the pinned commit](https://github.com/gnmtdt/AGE-YOLO/blob/99002665ff2f7faa5013c081b1b8f547ae4566ec/README.md#comparison-with-sota-models-gc10-det)
gives 72.8 ± 0.23% mAP50 across five runs, with 72.1% precision and 68.4%
recall. The checkpoint test result of 72.7646% mAP50 is 0.0354 percentage
points below that mean. Its precision was 73.29% and recall was 66.72%.
The cited table provides no mAP50–95 value. The validation score describes a
different partition and is reported separately.

## Hardware and timing

Both runs used an NVIDIA GeForce RTX 5050 Laptop GPU with 8,151 MiB VRAM,
driver 610.74, CUDA 12.8, Python 3.12.14, PyTorch 2.7.1+cu128, and
torchvision 0.22.1+cu128 on Linux under WSL2.

| Split | Preprocess | Inference | Postprocess | Sum of those stages |
| --- | ---: | ---: | ---: | ---: |
| Validation | 0.768437 | 3.919751 | 2.330289 | 7.018476 |
| Test | 0.643170 | 3.406251 | 1.558375 | 5.607797 |

Times are mean milliseconds per image from the validator. They exclude full
wall-clock overhead such as loading, warmup, and plotting. The authors report
189 FPS including preprocessing, forward pass, and NMS on an RTX 4090D with
batch size 1 at 640 × 640. The different GPU, batch size, and rectangular
inputs make these timings unsuitable for a direct speed comparison.
[Authors' timing setup](https://github.com/gnmtdt/AGE-YOLO/blob/99002665ff2f7faa5013c081b1b8f547ae4566ec/README.md#map-vs-parameters)

## Warnings, limitations, and test policy

- Both terminal logs contain an upstream invalid escape sequence warning and
  deprecation warnings for older timm and PyTorch AMP APIs. Neither log shows
  a traceback, skipped corrupt samples, duplicate-label removal, or an NMS
  timeout.
- This milestone verifies inference and measured metrics for one released
  checkpoint. Independent training reproduction would require fresh training
  and repeated runs under a documented protocol. The checkpoint's original
  seed, training membership, and relation to the five reported runs remain
  unverified. Close mAP50 agreement cannot establish training reproducibility
  or independent test exposure. Some class estimates have small support,
  including four validation and six test objects for `rolled_pit`.
- Treat these test results and artifacts as a fixed checkpoint verification
  record. Use validation for subsequent hyperparameter tuning, threshold
  selection, and model or checkpoint selection. Exclude test data, metrics,
  predictions, and plots from those decisions. Any later final test evaluation
  requires explicit authorization after freezing the model and settings.

The original artifacts remain in the ignored local directories
`experiments/age-yolo-gc10-val-01/` and `experiments/age-yolo-gc10-test-01/`.
Each contains the terminal log, identities, configuration, environment,
full-precision metrics, per-class results, timing, and plots.
