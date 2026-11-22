# Pinned AGE-YOLO architecture

A0 uses the [released model YAML](../../upstream/AGE-YOLO/ultralytics/cfg/models/age_yolo.yaml)
at submodule commit `99002665ff2f7faa5013c081b1b8f547ae4566ec`.
The scale is `n` with depth 0.50, width 0.25 and maximum nominal
channels 1024. Each YAML repeat of 2 resolves to one internal repeat.
The table shows actual channels and effective stride, with H × W
measured for a 640 × 640 input. Layers 0–22 apply to both models. Source
indices refer to the global layer list.

| Layer | Stage | Module | Source layers | Channels in → out | H × W | Stride | Applies to |
|---:|---|---|---|---|---|---:|---|
| 0 | Backbone | Conv 3×3 | RGB input | 3 → 16 | 320 × 320 | 2 | Both |
| 1 | Backbone | Conv 3×3 | 0 | 16 → 32 | 160 × 160 | 4 | Both |
| 2 | Backbone | C3k2 | 1 | 32 → 64 | 160 × 160 | 4 | Both |
| 3 | Backbone | Conv 3×3 | 2 | 64 → 64 | 80 × 80 | 8 | Both |
| 4 | Backbone | C3k2 | 3 | 64 → 128 | 80 × 80 | 8 | Both |
| 5 | Backbone | Conv 3×3 | 4 | 128 → 128 | 40 × 40 | 16 | Both |
| 6 | Backbone | FC-MANet | 5 | 128 → 128 | 40 × 40 | 16 | Both |
| 7 | Backbone | Conv 3×3 | 6 | 128 → 256 | 20 × 20 | 32 | Both |
| 8 | Backbone | FC-MANet | 7 | 256 → 256 | 20 × 20 | 32 | Both |
| 9 | Backbone | SPPF | 8 | 256 → 256 | 20 × 20 | 32 | Both |
| 10 | Backbone | C2PSA | 9 | 256 → 256 | 20 × 20 | 32 | Both |
| 11 | Neck | Nearest upsample ×2 | 10 | 256 → 256 | 40 × 40 | 16 | Both |
| 12 | Neck | ACF | 11, 6 | 256 + 128 → 256 | 40 × 40 | 16 | Both |
| 13 | Neck | FC-MANet | 12 | 256 → 128 | 40 × 40 | 16 | Both |
| 14 | Neck | Nearest upsample ×2 | 13 | 128 → 128 | 80 × 80 | 8 | Both |
| 15 | Neck | ACF | 14, 4 | 128 + 128 → 256 | 80 × 80 | 8 | Both |
| 16 | Neck | FC-MANet (P3) | 15 | 256 → 64 | 80 × 80 | 8 | Both |
| 17 | Neck | Conv 3×3 | 16 | 64 → 64 | 40 × 40 | 16 | Both |
| 18 | Neck | ACF | 17, 13 | 64 + 128 → 256 | 40 × 40 | 16 | Both |
| 19 | Neck | FC-MANet (P4) | 18 | 256 → 128 | 40 × 40 | 16 | Both |
| 20 | Neck | Conv 3×3 | 19 | 128 → 128 | 20 × 20 | 32 | Both |
| 21 | Neck | ACF | 20, 10 | 128 + 256 → 512 | 20 × 20 | 32 | Both |
| 22 | Neck | FC-MANet (P5) | 21 | 512 → 256 | 20 × 20 | 32 | Both |
| 23 | Head | SIFHead | 16, 19, 22 | 64 / 128 / 256 → 74 each | 80² / 40² / 20² | 8 / 16 / 32 | A0 |
| 23 | Neck | Nearest upsample ×2 | 16 | 64 → 64 | 160 × 160 | 4 | A1 |
| 24 | Neck | ACF | 23, 2 | 64 + 64 → 128 | 160 × 160 | 4 | A1 |
| 25 | Neck | Conv 1×1 (P2) | 24 | 128 → 16 | 160 × 160 | 4 | A1 |
| 26 | Head | SIFHead | 25, 16, 19, 22 | 16 / 64 / 128 / 256 → 74 each | 160² / 80² / 40² / 20² | 4 / 8 / 16 / 32 | A1 |

ACF aligns its first input to the second input's channel count using a
1×1 Conv when widths differ. It concatenates the aligned features,
applies channel and spatial attention and 1×1 fusion, then cross-weights
the inputs with residual additions. The two updated tensors are
concatenated on output. FC-MANet refines backbone P4/P5 and the neck
features at layers 13, 16, 19 and 22.
[Pinned module definitions](../../upstream/AGE-YOLO/ultralytics/nn/modules/age_yolo_modules.py)

## Detection head and training loss

- The parser passes the full list of input widths into SIFHead and
  scales its nominal hidden width of 256 to 64. Each level gets a
  1×1 projection and a learnable regression multiplier initialized to
  1. All levels reuse a depthwise 3×3 and pointwise 1×1 stack, SimAM,
  and separate shared 1×1 box and class predictors. GroupNorm uses 16
  groups of four channels. Each level emits 64 box logits (four sides
  × 16 bins) and 10 class logits.
- The model constructor forwards a square zero tensor at size 640.
  It divides 640 by each raw output height to initialize the strides
  and then initializes the shared box bias to 1 and the shared class
  bias to `log(5 / nc / (640 / 16)^2)`. The class prior is shared
  across levels, including the added P2 level.
  [Pinned model parser and constructor](../../upstream/AGE-YOLO/ultralytics/nn/tasks.py)
- `make_anchors` enumerates grid-cell centers at half-cell offsets and
  returns one stride per point. Inference flattens and concatenates all
  levels. DFL computes four expected distances using the 16-bin
  softmax distributions. The decoder converts those distances about
  each point to center-width-height boxes and multiplies by stride.
  The 10 independent sigmoid class scores accompany each decoded box.
- The loss flattens training logits across the same ordered levels.
  Task-aligned assignment uses detached sigmoid scores and decoded
  boxes in pixels. It considers centers inside each ground-truth box
  and selects up to 10 candidates using class score to the power 0.5
  times nonnegative CIoU to the power 6. It resolves conflicts by
  highest overlap and uses normalized alignment quality as soft targets.
  [Pinned assignment and anchor functions](../../upstream/AGE-YOLO/ultralytics/utils/tal.py)
- Classification uses BCE with logits across foreground and
  background. Localization uses foreground CIoU loss and DFL against
  neighboring distance bins. The latter two are weighted by target
  quality. Each term uses total target quality with a minimum
  denominator of 1. The default box, classification and DFL gains
  are 7.5, 0.5 and 1.5.
  [Pinned native loss](../../upstream/AGE-YOLO/ultralytics/utils/loss.py)
