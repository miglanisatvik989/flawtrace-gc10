# AGE-YOLO upstream reference

FlawTrace pins [gnmtdt/AGE-YOLO](https://github.com/gnmtdt/AGE-YOLO) at
[`99002665ff2f7faa5013c081b1b8f547ae4566ec`](https://github.com/gnmtdt/AGE-YOLO/commit/99002665ff2f7faa5013c081b1b8f547ae4566ec)
via the Git submodule at `upstream/AGE-YOLO`. Paths below are relative to that
checkout. Findings are from source inspection. The associated
[AGE-YOLO paper](https://doi.org/10.1016/j.engappai.2026.114690) describes the
YOLOv11-based architecture and its three custom components.

## License and checkpoints

The root [LICENSE](https://github.com/gnmtdt/AGE-YOLO/blob/99002665ff2f7faa5013c081b1b8f547ae4566ec/LICENSE)
contains the GNU Affero General Public License v3, consistent with the
`pyproject.toml` declaration. Bundled third-party code also has notices,
including Apache-2.0 for Mamba and a BSD-style notice for NVIDIA CUTLASS.
Preserve those notices and review applicable terms before distributing an
adaptation or repackaging assets.

The [pinned weights directory](https://github.com/gnmtdt/AGE-YOLO/tree/99002665ff2f7faa5013c081b1b8f547ae4566ec/weights)
contains three Git-tracked checkpoints. The upstream
[GitHub Releases page](https://github.com/gnmtdt/AGE-YOLO/releases) lists no
release. These hashes were computed from the checked-out files without loading
the checkpoints.

| Upstream file | SHA-256 |
| --- | --- |
| `weights/pretrained_gc10.pt` | `2daca14beedf36d6f7f74bca018292239a8eeb398fbe5b1c57516e410a61dc49` |
| `weights/pretrained_neu.pt` | `e49278ed435b865560b736027595863514a1a12faf8f7060d4d2c20c1c97ae53` |
| `weights/pretrained_pcb.pt` | `fc8b786b5f3541193675ca806ef04f199901d83bf2ef34c9d5f05d06d4aaa492` |

The repository provides the files but does not identify their training run,
seed, exact split, or relationship to individual results in the paper. Separate
checkpoint terms and permissions for the bundled sample images were not found
in the pinned documentation. Confirm provenance and applicable terms before
using the weights as reproduction evidence or redistributing the assets.
Checkpoint loading and runtime compatibility remain unverified.

## Model and entry points

The byte-identical `cfg/models/age_yolo.yaml` and
`ultralytics/cfg/models/age_yolo.yaml` define ten classes and one scale, `n`,
with depth multiplier 0.50, width multiplier 0.25, and maximum channels 1024.
The detection head takes three feature maps at strides 8, 16, and 32.

The custom layers live in `ultralytics/nn/modules/age_yolo_modules.py`,
are exported from `ultralytics.nn.modules`, and have custom handling in
`ultralytics/nn/tasks.py`. `FC_MANet` uses partial convolution and gated
blocks in the backbone and neck. `ACF_Module` combines features using channel
and spatial attention with learnable cross-feature weights. `SIFHead` shares
convolutions across feature scales and uses group normalization.

| Entry point | Behavior at the pin |
| --- | --- |
| `train.py` | Builds the model from `ultralytics/cfg/models/age_yolo.yaml`, uses `dataset/gc10_data.yaml`, and trains from scratch on device 0. Defaults are 300 epochs, batch 16, 640 px, SGD, initial LR 0.01, patience 50, seed 42, and fixed output project/name under `runs/AGE-YOLO-GC10`. Online mosaic is enabled, online flips are disabled, and scale/HSV jitter are set. The README lists five reported seeds, while the entry point hardcodes just 42. |
| `test.py` | Opens `weights/pretrained_gc10.pt` and calls `model.val` with `split='test'`, 640 px, and device 0. **The YAML path is wrong** (`dataset/gc10.yaml` instead of the tracked `dataset/gc10_data.yaml`). **The advertised `--split` flag is ignored** because the script never parses arguments. Once the path is resolved, the script still targets the test partition. |
| `detect.py` | Demonstrates prediction on `assets/samples`. It references the absent `weights/age_yolo_gc10.pt` and configures `assets/results` with `exist_ok=True`. |

## Environment

`requirements.txt` declares PyTorch ≥1.8, torchvision ≥0.9, NumPy ≥1.23,
OpenCV ≥4.6, Matplotlib ≥3.3, PyYAML ≥5.3.1, SciPy ≥1.4.1, tqdm ≥4.64,
Pandas ≥1.1.4, Seaborn ≥0.11, Ultralytics ≥8.3, timm ≥0.9, plus psutil and
einops without version bounds. `pyproject.toml` declares Python ≥3.8 and
packages the bundled `ultralytics` as the `age_yolo` distribution. Its
dependency list differs from `requirements.txt` and neither file is a lockfile.
The bundled source reports Ultralytics 8.3.9 and includes unconditional imports
such as `dill` and `requests` that neither manifest directly declares.

Both distributions can supply the `ultralytics` import. The installed public
package can shadow the fork depending on the launch path or package install
order, which can break custom YAML parsing or checkpoint loading. Use an
isolated CUDA environment for the reference, resolve dependencies around the
bundled fork, and verify the import path and custom parser before loading any
checkpoint. The compatible Python, PyTorch, torchvision, driver/CUDA and
extra dependency versions remain to be established. No environment was installed
or exercised in this review.

## Dataset preparation and offline augmentation

The [upstream dataset guide](https://github.com/gnmtdt/AGE-YOLO/blob/99002665ff2f7faa5013c081b1b8f547ae4566ec/dataset/README.md)
offers either [author-provided YOLO splits](https://huggingface.co/datasets/YYYHHHH993/AGE-YOLO)
or a raw workflow using the linked Kaggle GC10-DET source. The processed data
is described as split and converted, without offline augmentation. For raw
data, `tools/xml2yolo.py` converts VOC XML boxes using the GC10 class map and
`tools/split_data.py` performs a basename-grouped 80/10/10 split with seed 42.
The converter expects `dataset/raw/GC10-DET/xmls` rather than the guide's
`dataset/raw/GC10/annotations`. The splitter writes a test list and never reads
the supplied `dataset/testlist_gc10.txt`, despite the guide's optional test-list
instructions. Its filesystem walk is unsorted, so seed 42 alone does not fix
membership across different directory orders.

The actual GC10 YAML is `dataset/gc10_data.yaml`. It defines ten named classes
and the root `dataset/processed/GC10`, with `images/{train,val,test}` and
corresponding YOLO labels under `labels/{train,val,test}`. Bundled Ultralytics
resolves a relative root against its configured datasets directory.
`tools/augment_train.py` interprets that root relative to its working directory.
Resolve these paths explicitly when establishing the FlawTrace configuration.

The root README calls offline augmentation mandatory and `train.py` assumes
it has been applied, whereas the dataset guide calls it optional for training
from the processed splits. The script acts only on training JPG and PNG
images and adds three variants per original at the default `--mult 4`. It
resizes by a factor from 0.8 to 1.2, flips horizontally with probability 0.5,
and jitters HSV saturation and value by up to 20%. It writes outputs alongside
the original training data. Its default `--yaml dataset/data.yaml` is absent.
It uses Python's random generator without a seed or a seed argument, iterates
unsorted filenames, and overwrites same-named variants on repeat runs. The
training seed does not control this step. Exact offline augmentation is
therefore not reproducible as shipped. Before baseline validation, decide
whether to use augmentation and record input membership, options, seed, stable
order, and output hashes if used. Preserve the validation and test splits.
