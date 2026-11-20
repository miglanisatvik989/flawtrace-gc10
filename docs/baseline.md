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

The [explicit baseline YAML](../configs/age_yolo_baseline.yaml) matches
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

The bundled
[default.yaml](https://github.com/gnmtdt/AGE-YOLO/blob/99002665ff2f7faa5013c081b1b8f547ae4566ec/ultralytics/cfg/default.yaml)
remains authoritative for other inherited fields. A run saves copies of the
explicit, dataset, model, and upstream default YAMLs, plus the actual
`args.yaml`, effective batch and worker counts, provenance, environment,
and terminal log. Full runs use `experiments/<run-id>/`; benchmarks use
`experiments/benchmarks/<run-id>/`. The wrapper reserves each run directory,
uses an exclusive local lock, and appends logs on resume. Native resume
restores the last saved epoch's optimizer and EMA. The pinned fork does
not save scaler or RNG state, and strips optimizer state at normal completion.
Resume therefore starts at the next saved-epoch boundary and does not promise
a bitwise continuation. An interrupted run without an intact `last.pt`
requires a fresh ID.

## AMP and the bounded benchmark

The pinned trainer's built-in AMP probe attempts to load YOLO11n weights.
The wrapper substitutes a local probe on a disposable AGE-YOLO copy. On the
RTX 5050 Laptop GPU with 8,151 MiB VRAM and PyTorch CUDA 12.8, one synthetic
`640 × 640` image and target produced FP16 features, finite native detection
losses, and finite scaled gradients on `cuda:0`. The smoke test uses an
initial scale of 128 and performs no optimizer update. Native training keeps
the upstream dynamic scaler and AMP policy. [PyTorch AMP](https://docs.pytorch.org/docs/2.7/amp.html)

After separate authorization, run the [manual benchmark](../README.md#independent-age-yolo-training).
It uses the full training loader, retains batch size 16 and the 300-epoch
schedule, performs 10 untimed batches and 50 measured batches, then stops
before validation or checkpoint saving. The timed window is still within
upstream LR warmup. The benchmark records one-second GPU samples, PyTorch
peak allocated and reserved memory, loader wait, synchronized batch time,
training images per second, and a projection for 459 batches per epoch and
300 epochs. Loader wait is host time between batches, including native loop
bookkeeping. The duration projection excludes validation, checkpoint writes,
initial setup, and potential early stopping. [NVIDIA monitoring reference](https://docs.nvidia.com/deploy/nvidia-smi/index.html)

Batch 16 completed the short training benchmarks. The optional native
validation check completed all 230 validation images at batch 32 with
training state resident, peaking at 4,182 MiB of sampled device memory.
Stable throughput remains unresolved because the runs showed intermittent
stalls. Review the [diagnostics](benchmark_diagnostics.md) before full
training. End-to-end epoch duration and resume after a real interruption
remain unmeasured.

## Reproducing the environment on HPC

Use uv 0.12.5 and a Linux x86-64 checkout in storage that can hold the Git
submodule, the Python environment, the data, and run outputs. Allocate one
NVIDIA GPU under your site's scheduler and run the CUDA checks on that node.
Start the commands from the root of the checkout. Record the exact commit
shown by Git and use that commit for later reproductions.

```sh
git clone --branch baseline-training https://github.com/miglanisatvik989/flawtrace-gc10.git
cd flawtrace-gc10
git rev-parse HEAD
git submodule update --init --checkout -- upstream/AGE-YOLO
uv python install 3.12.14
uv venv --python 3.12.14 --no-python-downloads .venv
uv pip sync --python .venv/bin/python \
  --index https://download.pytorch.org/whl/cu128 \
  --default-index https://pypi.org/simple \
  --index-strategy unsafe-first-match requirements.txt
uv pip check --python .venv/bin/python
```

The data preparation requires `bsdtar` on PATH. If compute nodes lack
network access, use the site's approved staging location for package setup and
data acquisition, or place the verified GC10 prefix under
`data/.downloads/gc10-prefix.bin` before extraction. Keep the same checkout
and storage paths visible to the GPU job and preserve the absolute checkout
path across resumes. Allow several GiB for the original and derived trees in
addition to package and experiment storage.

```sh
source .venv/bin/activate
test -d data/gc10-det || python scripts/prepare_data.py
python scripts/check_data.py
test -d data/gc10-det-aug42-x4 || python -u scripts/augment_data.py
python -u scripts/augment_data.py --verify
```

From the allocated GPU node, manually run `python scripts/check_env.py`
and `python scripts/train.py --mode check-amp` before a separately authorized
benchmark. Leave the scheduler's `CUDA_VISIBLE_DEVICES` mask intact. The
wrapper uses logical `cuda:0` within that mask and selects the matching
physical GPU by UUID for `nvidia-smi`. Follow the site's resource and
wall-time rules and the [manual execution commands](../README.md#independent-age-yolo-training).
[Slurm GPU assignment](https://slurm.schedmd.com/gres.html#GPU_Management)
