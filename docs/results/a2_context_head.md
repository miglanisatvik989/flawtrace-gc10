# A2 — Context-aware SIFHead ablation

**Completed and independently reviewed. Negative overall result.** A2
added shared, dilated depthwise residual context to the box branch of
A1’s SIFHead. A1 remains the provisional development architecture. A0
remains the matched original AGE-YOLO performance control.

A2 used 640-pixel training, physical batch 32, accumulation 2, native
validation batch 64, and seed 42. It completed 300 epochs in 3.685 h
with a displayed reserved memory peak of 14.9 GiB. Epoch 273 was
selected by native fitness, with training CSV mAP50/mAP50–95 of
65.791%/34.447%. PBS training job `47981.mgmt01` and standalone job
`47983.mgmt01` completed successfully. The selected checkpoint SHA-256
is `d89418563912c8f240d069ec4a666451efd987177ccf69ba2cd1cdedd992db0d`.
The archived YAML, launch-time source, and AUG42 data identities
verified. [Architecture](../architecture/context_sifhead.md) · [Protocol](../experiments/a2_context_head.md)

## Standalone results

Native FP32 validation at 640 used the same 230 GC10-DET validation
images and 345 D1 objects, rectangular batch 16 and the archived A0/A1
methodology. AP exports reproduced native overall and per-class AP.
Fixed-point metrics use confidence ≥0.25 and a correct class with
IoU ≥0.50.

| Metric | A0 | A1 | A2 |
|---|---|---|---|
| mAP50 | 70.01% | 71.00% | 65.74% |
| mAP50–95 | 35.09% | 35.21% | 34.36% |
| Nine-class mAP50 | 74.29% | 70.57% | 72.29% |
| Nine-class mAP50–95 | 37.71% | 35.97% | 37.80% |
| Native precision | 78.59% | 71.31% | 63.31% |
| Native recall | 64.30% | 68.38% | 65.22% |
| Fixed precision | 72.73% | 68.50% | 68.24% |
| Fixed recall | 69.57% | 64.93% | 62.90% |
| Short side ≤16 px recall | 45.71% | 37.14% | 34.29% |
| TP | 240 | 224 | 217 |
| FP | 90 | 103 | 101 |
| FN | 105 | 121 | 128 |

## Paired bootstrap

| Comparison | Metric | Δ (pp) | 95% interval (pp) | Valid draws |
|---|---|---|---|---|
| A2 − A1 | mAP50 | -5.27 | [-9.59, +3.16] | 4325 |
| A2 − A1 | mAP50-95 | -0.85 | [-2.91, +2.43] | 4325 |
| A2 − A1 | nine_class_AP50 | +1.72 | [-2.13, +4.82] | 5000 |
| A2 − A1 | nine_class_AP50-95 | +1.83 | [-0.24, +3.31] | 5000 |
| A2 − A1 | fixed_recall | -2.03 | [-5.96, +1.79] | 5000 |
| A2 − A1 | q1_recall | -5.49 | [-12.35, +1.37] | 5000 |
| A2 − A1 | short_le16_recall | -2.86 | [-14.29, +7.14] | 5000 |
| A2 − A0 | mAP50 | -4.27 | [-9.73, +0.08] | 4325 |
| A2 − A0 | mAP50-95 | -0.73 | [-3.08, +1.36] | 4325 |
| A2 − A0 | nine_class_AP50 | -2.00 | [-6.13, +1.27] | 5000 |
| A2 − A0 | nine_class_AP50-95 | +0.09 | [-2.08, +2.04] | 5000 |
| A2 − A0 | fixed_recall | -6.67 | [-10.24, -3.13] | 5000 |
| A2 − A0 | q1_recall | -12.09 | [-20.00, -4.48] | 5000 |
| A2 − A0 | short_le16_recall | -11.43 | [-30.00, +0.00] | 5000 |

The 5,000 shared image draws used seed 42. Ten-class AP excluded
675 draws without full class support. The remaining metrics had
5,000 valid draws. A2’s nine-class mAP50–95 gain over A1 was
+1.83 pp [−0.24, +3.31]. The primary AP intervals span zero.
These intervals address validation-image variation for fixed
checkpoints from one training seed.

## Class and localization evidence

| Class | A0 AP50 / AP50–95 | A1 AP50 / AP50–95 | A2 AP50 / AP50–95 |
|---|---|---|---|
| oil_spot | 72.20% / 36.66% | 69.64% / 35.68% | 68.21% / 35.22% |
| silk_spot | 59.98% / 24.71% | 58.26% / 26.45% | 56.66% / 23.60% |
| inclusion | 48.20% / 13.96% | 34.03% / 13.48% | 46.27% / 15.39% |
| rolled_pit | 31.46% / 11.44% | 74.85% / 28.32% | 6.75% / 3.37% |
| crease | 33.26% / 15.90% | 22.94% / 14.01% | 26.99% / 17.72% |

A2 recovered 19 objects and lost 26 versus A1. It recovered five
objects missed by both controls. All three missed 83 objects.
Inclusion AP50 improved from 34.03% to 46.27%, while rolled-pit AP50
fell from 74.85% to 6.75%. Rolled pit has only four objects across
two images. The archived per-object table and examples preserve
all four outcomes. Localization-only errors were 27, 28, and 29
for A0, A1, and A2. The regression context block did not reduce
these errors.

| FP/image budget | A0 recall | A1 recall | A2 recall |
|---|---|---|---|
| 0.1 | 53.33% | 50.14% | 47.54% |
| 0.25 | 65.22% | 58.55% | 60.58% |
| 0.5 | 71.30% | 66.09% | 64.06% |
| 1.0 | 75.07% | 71.30% | 69.57% |

A2 did not improve on A0 at any registered FP budget. The saved
coverage, size, level attribution and error tables provide the
support for follow-up diagnostics.

## Same-GPU efficiency

All timing passes used one NVIDIA H100 (UUID
`GPU-61f54ca7-3cee-1c93-def3-118523e95241`) in A0, A1, A2, A2, A1,
A0 order. Mean batch-1 medians and mean batch-16 throughput come
from the repeated passes. Estimated GFLOPs use the pinned profiler.
Memory is the maximum over the matching batch-16 passes.

| Model | Parameters | Estimated GFLOPs | Batch-1 median (ms) | Batch-16 images/s | Batch-16 peak allocated / reserved (MiB) |
|---|---|---|---|---|---|
| A0 | 3,418,721 | 7.874 | 19.47 | 606.75 | 435.1 / 516.0 |
| A1 | 3,446,915 | 9.690 | 24.59 | 461.32 | 696.0 / 976.0 |
| A2 | 3,451,715 | 10.007 | 22.93 | 548.10 | 673.1 / 882.0 |

## Decision and archives

The class-specific benefits do not outweigh the overall result on
this protocol. The experiment gives no evidence of overall statistical
superiority. It tests one implementation and one training seed,
so it does not establish that contextual regression is generally
ineffective.

The complete directories, checkpoints, source snapshots, exports, and
available PBS/launcher logs are archived under `artifacts/A2/`. Both
archives matched their original files. Verify with
`(cd artifacts/A2 && sha256sum -c SHA256SUMS)`.

| Archive | SHA-256 |
|---|---|
| `a2-age-yolo-p2-context-640-b32-01.tar` | `6dc8e1a50e1a865ce7aa789d5d74fd354a1fd3b574a48db122858c7f22434d1e` |
| `a2-architecture-eval-01.tar` | `873f6e9e32e2af9bdbf7cc3c47adf5a40893a75e175afe11667745da12b9e221` |
