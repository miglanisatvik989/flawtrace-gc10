# AGE-YOLO CUDA environment

## Host

| Component | Observed version or value |
| --- | --- |
| OS | Ubuntu 26.04.1 LTS, Linux x86-64 under WSL2 |
| Kernel | `6.18.33.1-microsoft-standard-WSL2` |
| Python available | CPython 3.12.14 and system CPython 3.14.4 |
| GPU | NVIDIA GeForce RTX 5050 Laptop GPU |
| GPU memory and compute capability | 8,151 MiB and 12.0 (`sm_120`) |
| NVIDIA driver | 610.74, as reported by `nvidia-smi` |
| Driver CUDA support | CUDA UMD 13.3 as reported by `nvidia-smi`. No `nvcc` was present on PATH. |
| CPU visible to WSL2 | Intel Core i7-13700HX, 24 logical CPUs. `lscpu` reports 12 cores in the guest topology. |
| Host memory visible to WSL2 | 7.6 GiB RAM total, 5.7 GiB available at inspection, and 3.9 GiB `/dev/shm` capacity. |

## Version selection

CPython 3.12.14 is available on this host and supported by torchvision 0.22.
PyTorch 2.7.1 and torchvision 0.22.1 are an official CUDA 12.8 wheel pairing.
PyTorch 2.7 introduced Blackwell support and CUDA 12.8 supports `sm_120`.
The newer driver can run the CUDA 12.8 binaries through driver backward
compatibility. The wheels supply the CUDA runtime libraries, so a local CUDA
compiler is unnecessary for this check.

- [PyTorch wheel pairing](https://pytorch.org/get-started/previous-versions/#v271)
- [torchvision and Python compatibility](https://github.com/pytorch/vision/tree/v0.22.1#installation)
- [PyTorch Blackwell support](https://pytorch.org/blog/pytorch-2-7/)
- [NVIDIA CUDA 12.8 release notes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/)
- [NVIDIA driver compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html)

The dependency list takes the union of upstream's two manifests except for the
public Ultralytics distribution, then adds imports used by the bundled fork.
Those include `dill`, `requests`, `efficientnet-pytorch`, PyWavelets, and
`ultralytics-thop` (supplies the `thop` module). NumPy 1.26.4 retains the
`np.trapz` API used in the fork's metrics code, and OpenCV 4.11 is compatible
with that NumPy version. These choices require no modifications to the pinned
upstream tree.

See the [installation commands](../README.md#environment-installation) and
[65 exact installed package versions](../requirements.txt), including transitive
dependencies. The install uses the official PyTorch CUDA wheel index and PyPI.
The index strategy lets uv find an exact pin on PyPI when the CUDA index lists
the package but lacks that version.

On UPES HPC, run `bash scripts/setup_hpc.sh` from the repository root on
`mgmt01`. The script uses `opencv-python-headless==4.11.0.86` so importing
`cv2` on compute nodes does not require the X11 library `libXext.so.6`.
It takes the version from the shared OpenCV pin, removes conflicting OpenCV
wheels from `flawtrace-gc10` during migration, and checks the import after
installation. All other packages use the shared pins.

## Smoke test on November 20, 2026

Ran `.venv/bin/python scripts/check_env.py` using the pinned AGE-YOLO commit
`99002665ff2f7faa5013c081b1b8f547ae4566ec`.

| Check | Result |
| --- | --- |
| PyTorch and torchvision imports | Passed with PyTorch `2.7.1+cu128` and torchvision `0.22.1+cu128` under CPython 3.12.14. |
| CUDA availability | `torch.cuda.is_available()` returned `True`. PyTorch reported CUDA 12.8. |
| CUDA tensor operations | A 2 × 2 matrix multiplication on `cuda:0` returned the expected result. |
| Ultralytics import | Bundled version 8.3.9 resolved to `/home/kaizen/code/flawtrace-gc10/upstream/AGE-YOLO/ultralytics/__init__.py`. |
| Official model YAML | `ultralytics/cfg/models/age_yolo.yaml` loaded and built 3,418,721 parameters. |
| CUDA forward pass | A synthetic FP32 input of shape `(1, 3, 640, 640)` returned four tensors on `cuda:0`. |
| CUDA mixed precision | The same model and input passed with FP16 CUDA autocast. All four outputs remained finite and on `cuda:0`, with both FP16 and FP32 output tensors. |
| Output finiteness | All entries in all four tensors were finite. Output shapes were `(1, 14, 8400)`, `(1, 74, 80, 80)`, `(1, 74, 40, 40)`, and `(1, 74, 20, 20)`. |
| CUDA unavailable | With `CUDA_VISIBLE_DEVICES=""`, the checker exited with an explicit CUDA error before constructing the model. |
| Fresh environment replay | The documented sync command reproduced all 65 package versions. `uv pip check` and the CUDA smoke test passed again. |

The upstream fork emitted deprecation warnings and a syntax warning in an
unused module. The model parser selected its only configured scale, `n`.
The mixed-precision result establishes synthetic forward compatibility.
Training, gradient scaling, and training memory usage remain untested.
The checker uses native PyTorch autocast. [PyTorch AMP documentation](https://docs.pytorch.org/docs/2.7/amp.html)

## HPC training

The current entry point probes AMP with a disposable copy of the configured
detector. B1 used eight workers and the headless OpenCV wheel. See the
[experiment registry](experiments.md) for run-specific evidence. For a
portable Linux environment, create a Python 3.12.14 virtual environment and
install the pinned wheel set with the CUDA index used by scripts/setup_hpc.sh.
