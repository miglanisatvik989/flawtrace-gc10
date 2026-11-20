# Independent AGE-YOLO baseline

The baseline initializes the pinned AGE-YOLO detection architecture from its
official YAML with random weights. The [released-checkpoint results](checkpoint_evaluation.md)
remain a separate verification record. This baseline trains on the augmented
training partition and selects checkpoints only on the authors' validation
partition. The training YAML exposes no test partition.

## Deterministic offline augmentation

The [pinned augmentation script](https://github.com/gnmtdt/AGE-YOLO/blob/99002665ff2f7faa5013c081b1b8f547ae4566ec/tools/augment_train.py)
uses whole-image resizing by a sampled factor from 0.8 to 1.2, horizontal
flip with probability 0.5, independent HSV saturation and value multipliers
from 0.8 to 1.2, and JPEG encoding. Resize and color jitter leave normalized
YOLO box coordinates unchanged. A flip maps the normalized x center to
`1 - x`. Class ID, y center, width, and height are preserved. Generated
coordinates use the upstream six-decimal format.

FlawTrace keeps the upstream multiplier 4, counting one original and three
variants per training image. The upstream augmentation script does not seed
its random generator. FlawTrace uses seed 42 from the authors' training entry
point as a reproduction assumption for offline augmentation and draws in
sorted source filename order. The authors' actual offline seed, order, and
realized variants are unknown. The pinned upstream documents differ on
whether the offline step is mandatory. The baseline follows the training
entry point's assumption that it is enabled.

| Partition | Images | Label files | Objects |
| --- | ---: | ---: | ---: |
| Train | 7,336 | 7,330 | 11,524 |
| Validation | 230 | 230 | 345 |
| Test, excluded from training configuration | 230 | 230 | 337 |

The two original empty training labels are preserved. As upstream does,
augmentation writes no label file for their six variants. Ultralytics
treats these eight images as backgrounds. Whether the originals are intended
negatives remains unresolved. Validation and test are byte-identical copies
with unchanged membership. The preparation verifies the original manifests
and checks generated images against held-out file and decoded RGB hashes.
No exact cross-split duplicates were found. This does not establish
independence of near duplicates or related production samples.

The derived files and manifests live under the ignored
`data/gc10-det-aug42-x4/`. The training manifest SHA-256 is
`c68282dc8c3ac5ed68c46f3a3253eaa6782b0acb921b37e5cbbf24a5587264a7`.
Regeneration on the pinned environment reproduced every image and label byte
and the manifest. Direct comparisons against the pinned transform covered
flipped and unflipped images. A separate check verified all 8,643 generated
boxes against their source coordinates and class IDs. The
[original dataset](dataset.md) remains intact.

## Training configuration

The [explicit baseline YAML](../configs/baseline/age_yolo_reference.yaml) matches
[upstream train.py](https://github.com/gnmtdt/AGE-YOLO/blob/99002665ff2f7faa5013c081b1b8f547ae4566ec/train.py).
It sets 300 maximum epochs, batch size 16, 640 px training input, SGD,
`lr0=0.01`, patience 50, `pretrained=false`, mosaic 1.0, horizontal and
vertical flip 0, scale 0.1, HSV gains 0.01/0.2/0.2, seed 42, and GPU 0.

| Area | Inherited settings |
| --- | --- |
| Optimization and schedule | Momentum 0.937, weight decay 0.0005, nominal batch 64, accumulation 4 after warmup at batch 16, linear schedule with final LR factor 0.01, warmup 3 epochs, warmup momentum 0.8, warmup bias LR 0.1. |
| Loss and repeatability | Box 7.5, classification 0.5, DFL 1.5, deterministic mode enabled, all training samples used, no freezing beyond the trainer's built-in DFL freeze. |
| Augmentation and loading | Translation 0.1, mixup and copy-paste disabled, mosaic closes for the final 10 epochs, multiscale and image caching disabled. Eight training workers and up to 16 validation workers. |
| Validation and saving | Validation each epoch, rectangular validation batch up to 32, NMS IoU 0.7, confidence 0.001, maximum 300 detections per image. Native `best.pt` follows validation fitness, weighted 0.1 mAP50 and 0.9 mAP50–95. `last.pt` is saved each epoch, with no additional periodic checkpoints by default. |
| Precision and output | Native AMP enabled, verbose live progress and native plots enabled. The displayed `GPU_mem` is PyTorch reserved memory. |

B1 uses batch 64, native validation batch 128 and accumulation 1.
The original and augmented dataset configurations are separate. B1 selects
weights using validation fitness only. Native resume restores the saved
epoch and optimizer state. The pinned fork omits scaler and RNG states, so
a resumed trajectory can differ from an uninterrupted run.

See the [experiment registry](experiments.md) for the completed training
review, loss convergence, artifacts and limitations. Earlier performance
tests on other hardware are retained in the [diagnostic history](benchmark_diagnostics.md).
