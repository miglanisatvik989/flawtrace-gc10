"""Verify the bundled AGE-YOLO model with synthetic CUDA forward passes."""

import importlib.util
import os
import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream" / "AGE-YOLO"
PACKAGE = UPSTREAM / "ultralytics" / "__init__.py"
MODEL_CONFIG = UPSTREAM / "ultralytics" / "cfg" / "models" / "age_yolo.yaml"


def output_tensors(output, tensor_type):
    """Yield detection predictions and feature tensors from the model output."""
    if isinstance(output, tensor_type):
        yield output
    elif isinstance(output, (tuple, list)):
        for item in output:
            yield from output_tensors(item, tensor_type)
    else:
        raise TypeError(f"Unexpected model output type: {type(output).__name__}")


def main():
    import torch
    import torchvision

    print(
        f"PASS: Python {platform.python_version()}, PyTorch {torch.__version__}, "
        f"torchvision {torchvision.__version__}"
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. torch.cuda.is_available() returned False.")

    device = torch.device("cuda:0")
    gpu = torch.cuda.get_device_properties(device)
    print(
        f"PASS: CUDA {torch.version.cuda}, {gpu.name}, "
        f"{gpu.total_memory / 2**20:.0f} MiB, compute capability {gpu.major}.{gpu.minor}"
    )
    values = torch.ones((2, 2), device=device)
    if not torch.equal(values @ values, torch.full_like(values, 2)):
        raise RuntimeError("CUDA matrix multiplication returned an unexpected result.")
    print("PASS: CUDA tensor operations")

    if not PACKAGE.is_file() or not MODEL_CONFIG.is_file():
        raise RuntimeError("Initialize the AGE-YOLO submodule before running this check.")

    sys.path.insert(0, str(UPSTREAM))
    sys.dont_write_bytecode = True
    spec = importlib.util.find_spec("ultralytics")
    if spec is None or spec.origin is None or Path(spec.origin).resolve() != PACKAGE:
        raise RuntimeError("Ultralytics must resolve to the bundled AGE-YOLO implementation.")

    config_dir = ROOT / ".cache" / "ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    import ultralytics

    print(f"PASS: Ultralytics {ultralytics.__version__} from {ultralytics.__file__}")
    torch.manual_seed(42)
    model = ultralytics.YOLO(str(MODEL_CONFIG), task="detect", verbose=False).model
    model.to(device).eval()
    if any(parameter.device != device for parameter in model.parameters()):
        raise RuntimeError("Expected all model parameters on CUDA device 0.")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"PASS: Official YAML loaded, {parameter_count:,} parameters")

    sample = torch.rand((1, 3, 640, 640), device=device)
    with torch.inference_mode():
        for precision, amp in (("FP32", False), ("FP16 autocast", True)):
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                outputs = list(output_tensors(model(sample), torch.Tensor))
            if not outputs or any(tensor.device != device for tensor in outputs):
                raise RuntimeError(f"{precision}: expected output tensors on CUDA device 0.")
            if amp and not any(tensor.dtype == torch.float16 for tensor in outputs):
                raise RuntimeError("FP16 autocast produced no FP16 output tensors.")
            if not all(torch.isfinite(tensor).all().item() for tensor in outputs):
                raise RuntimeError(f"{precision}: model output contains non-finite values.")
            dtypes = ", ".join(sorted({str(tensor.dtype) for tensor in outputs}))
            shapes = [tuple(tensor.shape) for tensor in outputs]
            print(f"PASS: {precision} CUDA forward, finite {dtypes} outputs {shapes}")


if __name__ == "__main__":
    main()
