# AGE-YOLO with P2

A1 adds a stride-4 route to the [original AGE-YOLO graph](age_yolo.md).
Layer 23 upsamples layer 16 (refined P3) by 2. Layer 24 fuses it with
layer 2 (refined backbone P2) using ACF. Both inputs have 64 channels, so
ACF uses its identity alignment path and returns 128 channels. Layer 25
uses the existing 1×1 Conv with batch normalization and SiLU to learn a
compressed P2 representation. A nominal width of 64 gives 16 channels
at scale `n`. This matches the smallest existing backbone width and
limits cost on the dense grid. Each SIFHead projection still outputs
64 channels where GroupNorm's 16-group requirement applies.
[Complete A1 model YAML](../../configs/models/age_yolo_p2.yaml)

SIFHead reads layers `[25, 16, 19, 22]` with widths `[16, 64, 128, 256]`.
The original backbone and P3–P5 connections carry the same tensors they
do in A0. P2 contributes gradients through the shared head, backbone and
P3 neck. Task-aligned assignment can select candidates from the extra
level, and the shared score prior applies at more grid locations.
The target rules, DFL bins, loss gains and box decoding are common.

## Complexity at 640 × 640

| Measurement | A0 | A1 | A1 change |
|---|---:|---:|---:|
| Unfused parameters | 3,418,721 | 3,446,915 | +28,194 (+0.82%) |
| Fused parameters | 3,408,649 | 3,436,827 | +28,178 (+0.83%) |
| GFLOPs, unfused | 7.874 | 9.690 | +1.815 (+23.05%) |
| Candidate locations | 8,400 | 34,000 | +25,600 (4.05× total) |

Both GFLOP estimates use the pinned `get_flops` THOP routine at
640, which reports twice the MAC count and scales its stride-sized
probe to the image size. It counts supported operations and can omit
custom tensor operations. Fused counts come from the fork's native
Conv–BN fusion. The [layer table](age_yolo.md) reports widths and
dimensions of all feature maps.

## Synthetic verification

The [architecture checker](../../scripts/check_architecture.py) builds
both models from YAML with no checkpoint, using the pinned fork in
offline mode and temporary settings. The FP32 CPU passes produced finite
raw maps and decoded boxes and scores. For square and rectangular inputs
the SIFHead grids were:

| Input H × W | P2 in A1 | P3 | P4 | P5 | Decoded A0 | Decoded A1 |
|---|---|---|---|---|---|---|
| 640 × 640 | 160 × 160 | 80 × 80 | 40 × 40 | 20 × 20 | 1 × 14 × 8,400 | 1 × 14 × 34,000 |
| 352 × 672 | 88 × 168 | 44 × 84 | 22 × 42 | 11 × 21 | 1 × 14 × 4,851 | 1 × 14 × 19,635 |

Each raw map has 74 channels. Anchor layout, strides, feature
projections, GroupNorm, regression scales and predictor biases passed.
Three synthetic boxes gave finite native box/classification/DFL
components of 3.6460/5.6392/4.2410 in A0 and 3.4107/10.1853/4.2448
in A1. Native assignment selected 24 foreground locations in A0 and
30 in A1, including 20 on P2. All trainable parameters received finite
gradients and the added P2 refinement received a nonzero gradient.
These values exercise the loss path on fresh initialization.

## GPU memory feasibility

The table estimates individual tensor sizes at square 640 and two bytes
per element. Actual dtype varies under AMP. It excludes saved
activations, temporary tensors, parameters, gradients, optimizer state,
CUDA workspaces and allocator overhead.

| Tensor | A0, train batch 32 | A1, train batch 32 | A0, native val batch 64 | A1, native val batch 64 |
|---|---:|---:|---:|---:|
| Raw head logits across levels | 37.94 MiB | 153.56 MiB | 75.88 MiB | 307.13 MiB |
| Extra P2 ACF output | — | 200 MiB | — | 400 MiB |
| Extra P2 head projection | — | 100 MiB | — | 200 MiB |
| Extra P2 refinement output | — | 25 MiB | — | 50 MiB |

The P2 grid has four times as many locations as P3 alone. Assignment
uses several tensors of shape batch × maximum ground truths per image
in the batch × locations, so their location-dependent memory grows
4.05×. For example, one FP32 assignment tensor at batch 32 and 32
ground truths per image as the batch maximum is about 33 MiB in A0
and 133 MiB in A1. Training retains intermediate activations for
backward. Native validation runs without gradients, but at twice the
physical batch and with loss computation inside training. Rectangular
padding can reduce spatial cost relative to this square example.

These tensor estimates do not determine peak CUDA memory. Both A0 and
A1 completed AMP training at physical batch 32 with native validation
batch 64. Their displayed peak reserved training memory was 7.94 and
14.00 GiB, respectively. Matched FP32 inference timings and separate
inference memory peaks are recorded in the [reviewed A1 results](../results/a1_p2_detection.md).
