# A2 regression context in SIFHead

A2 retains the [A1 backbone and P2–P5 neck](age_yolo_p2.md) and adds
one residual block before the existing box predictor at layer 26.
The same block weights operate independently on each detection level.
[Complete A2 model YAML](../../configs/models/age_yolo_p2_context.yaml)

## Original SIFHead

The pinned parser resolves the model YAML by module name. For members
of `DETECT_CLASS`, it appends the feature-channel list. The built-in
SIFHead also receives width scaling, so its nominal hidden width of 256
becomes 64 at scale `n`. The model constructor derives strides with a
640-pixel zero-input forward and then initializes the head biases. The
fork’s model device conversion moves the custom head’s stride and
cached anchor tensors through `DETECT_CLASS` dispatch.
[Pinned parser and model constructor](../../upstream/AGE-YOLO/ultralytics/nn/tasks.py)

SIFHead projects each level independently using a 1×1 convolution,
16-group normalization and SiLU. All levels then use the same 3×3
depthwise and 1×1 pointwise convolutions, each with GroupNorm and SiLU.
SimAM reweights the resulting channels and positions using spatial
statistics. Classification and regression consume this same tensor.
Separate 1×1 predictors emit 10 class logits and 64 box logits at each
location. The box logits represent four sides and 16 distance bins.
A scalar per detection level multiplies the box logits. Box bias starts
at 1, the class bias uses the shared prior
`log(5 / nc / (640 / 16)^2)`, and regression scales start at 1.
[Pinned SIFHead](../../upstream/AGE-YOLO/ultralytics/nn/modules/age_yolo_modules.py)

| Sequence | A1 | A2 | Output channels per level |
|---|---|---|---:|
| Backbone layers 0–10 | Original AGE-YOLO | Same | See layer audit |
| Neck layers 11–22 | Original FC-MANet and ACF P3–P5 | Same | P3 64, P4 128, P5 256 |
| Neck layers 23–25 | P3 upsample, ACF with backbone P2, 1×1 Conv | Same | P2 16 |
| 26a | Per-level Conv–GroupNorm–SiLU projection | Same | 64 |
| 26b | Shared depthwise 3×3 and pointwise 1×1 with GN/SiLU, then SimAM | Same | 64 |
| 26c, class | Shared 1×1 classifier | Same input and predictor | 10 |
| 26c, box | Shared 1×1 predictor | Shared residual context followed by the original predictor | 64 |
| 26d | Per-level regression scalar, concatenate box and class logits | Same | 74 |
| 26e, inference | Per-side DFL expectation, center/size decode and class sigmoid | Same | 14 |

The [full backbone and neck layer table](age_yolo.md) includes widths,
feature sizes, fusion connections and strides. The A2 YAML is
identical to A1 through layer 25.

In training, the head returns a list of raw `[B, 74, H, W]` tensors.
The native loss flattens and joins them in level order. Anchors are
cell centers at offset 0.5. Task-aligned assignment uses the original
class and box predictions and the original top-10 rule across levels.
BCE classification, foreground CIoU and DFL remain native. DFL supervises
distance distributions in adjacent bins. In inference the bins are
softmaxed, converted to expected distances and decoded around each
center. Strides convert coordinates to input pixels. Inference returns
decoded `[B, 14, N]` predictions with the raw maps, or only the decoded
tensor in export mode. Native NMS follows decoding.
[Pinned loss](../../upstream/AGE-YOLO/ultralytics/utils/loss.py) ·
[Anchor, assignment and decode functions](../../upstream/AGE-YOLO/ultralytics/utils/tal.py)

## Regression-only change

For shared features `x`, A2 feeds the existing box predictor with
`x + pointwise(depthwise_gn_silu(x))`. The depthwise convolution uses
a 3×3 kernel, 64 groups, dilation 2 and padding 2. GroupNorm has 16
groups of four channels. The bias-free pointwise convolution preserves
width. Its weights start at zero, so the new branch starts as an exact
identity. The pointwise convolution can learn on the first step. The
depthwise convolution and GroupNorm receive gradients once the
pointwise weights move away from zero. The classifier still reads
`x` directly. Regression gradients and changed assignment can affect
the shared trunk during training.

Dilation samples offsets −2, 0 and +2 along each axis, giving a
theoretical 5×5 stencil. With the preceding shared 3×3 convolution,
the convolutional path spans up to 7×7 grid locations. The added
two-cell radius corresponds to 8, 16, 32 and 64 input pixels at
P2 through P5. GroupNorm and SimAM already use spatial statistics.
These spans describe convolutional support. The effective receptive
field after training and the effect on defect coverage remain
unmeasured. Dilation can also mix adjacent structures or background.

The [project extension](../../scripts/context_sifhead.py) subclasses
SIFHead and wraps `cv2`. The inherited forward, bias initialization
and decoder still run. Recording the existing box predictor’s bias
through the wrapper preserves native initialization. The trainer
registers the new class by name and adds it to `DETECT_CLASS` in
memory. The pinned scale rule applies only to specific built-in
classes, so A2 passes the resolved width of 64 explicitly. Native YAML
construction and resume rebuilds use the registration. Native loading
of A2 checkpoints uses the project module on the scripts import path.
The checkpoint’s model graph and the trainer’s optimizer grouping
remain supported by the pinned fork.

## Complexity and synthetic verification

| Model | Strides | Locations at 640 | Unfused parameters | Fused parameters | THOP GFLOPs at 640 |
|---|---|---:|---:|---:|---:|
| A0 | 8, 16, 32 | 8,400 | 3,418,721 | 3,408,649 | 7.874 |
| A1 | 4, 8, 16, 32 | 34,000 | 3,446,915 | 3,436,827 | 9.690 |
| A2 | 4, 8, 16, 32 | 34,000 | 3,451,715 | 3,441,627 | 10.007 |

The shared block adds 576 depthwise weights, 128 GroupNorm parameters
and 4,096 pointwise weights. The total of 4,800 increases A1 parameters
by 0.139%. The pinned `get_flops` routine uses THOP on a square 640 input
and reports twice its counted MACs. A2 adds 0.318 GFLOPs (3.28%).
The estimate does not capture every activation, residual addition,
normalization or memory operation. Latency remains unmeasured.

The [architecture checker](../../scripts/check_architecture.py) passed
on CPU for all three graphs. At 640 × 640, A2 emitted raw spatial grids
160², 80², 40² and 20², and decoded shape `[1, 14, 34000]`.
At 352 × 672, the grids were 88 × 168, 44 × 84, 22 × 42 and 11 × 21,
and decoded shape was `[1, 14, 19635]`. Output tensors, class
probabilities, anchor order and strides passed. The synthetic A2
native box, classification and DFL losses were finite (3.4054,
9.0629 and 4.1821). Assignment selected 20 P2 and 10 P3 foreground
locations. The pointwise gradient was nonzero on the first
backward pass. After one native SGD step, all context parameters
had nonzero finite gradients. Native save, recovery copy, checkpoint
reload, model rebuild, optimizer restore and inference backend passed
with a temporary checkpoint.

## Memory feasibility

At square 640, one 64-channel tensor per level at two bytes per
element occupies 100, 25, 6.25 and 1.56 MiB at batch 32, from
P2 through P5. The four tensors together occupy 132.81 MiB.
At batch 64 those figures double. The context branch adds intermediate
tensors for depthwise convolution, normalization, activation,
pointwise convolution and residual addition. Backward retains tensors
and adds gradients and workspaces. As an illustration, four extra
full-width tensor equivalents would occupy about 531 MiB at batch 32
and two bytes per element. Native validation runs without gradients,
processes levels sequentially and typically sees smaller rectangular
grids. Dtype, tensor lifetimes and workspace size govern actual peaks.

Completed AMP training at batch 32 and native validation batch 64
showed a displayed reserved training peak of 14.9 GiB for A2 and
14.00 GiB for A1. Timing and measured inference allocation are in the
[reviewed A2 results](../results/a2_context_head.md).
