"""Run native AGE-YOLO detection evaluation or an offline checkpoint smoke check."""

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from check_env import output_tensors


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream" / "AGE-YOLO"
DATASET = ROOT / "data" / "gc10-det"
MANIFESTS = ROOT / "data" / "manifests"
CONFIG = ROOT / "configs" / "gc10.yaml"
CHECKPOINT_SHA256 = "2daca14beedf36d6f7f74bca018292239a8eeb398fbe5b1c57516e410a61dc49"
UPSTREAM_COMMIT = "99002665ff2f7faa5013c081b1b8f547ae4566ec"
IMAGE_SIZE = 640


class Tee:
    def __init__(self, terminal, log):
        self.terminal = terminal
        self.log = log

    def write(self, text):
        self.log.write(text)
        return self.terminal.write(text)

    def flush(self):
        self.log.flush()
        self.terminal.flush()

    def __getattr__(self, name):
        return getattr(self.terminal, name)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_inputs(checkpoint):
    if checkpoint.suffix != ".pt" or not checkpoint.is_file():
        raise RuntimeError("Supply a local .pt checkpoint.")
    if sha256(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("Checkpoint SHA-256 does not match the official GC10 checkpoint.")


def load_model(checkpoint):
    import torch
    import yaml

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for AGE-YOLO evaluation.")
    device = torch.device("cuda:0")
    sys.path.insert(0, str(UPSTREAM))
    sys.dont_write_bytecode = True
    expected = UPSTREAM / "ultralytics" / "__init__.py"
    spec = importlib.util.find_spec("ultralytics")
    if spec is None or spec.origin is None or Path(spec.origin).resolve() != expected:
        raise RuntimeError("Ultralytics must resolve to the bundled AGE-YOLO source.")

    os.environ.update(
        YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false", YOLO_VERBOSE="true", MPLBACKEND="Agg"
    )
    import ultralytics

    model = ultralytics.YOLO(str(checkpoint)).to(device)
    names = dict(enumerate(yaml.safe_load(CONFIG.read_text())["names"]))
    if model.task != "detect" or model.names != names:
        raise RuntimeError("Checkpoint task or class mapping disagrees with configs/gc10.yaml.")
    if any(tensor.device != device for tensor in model.model.parameters()):
        raise RuntimeError("Checkpoint parameters must all be on cuda:0.")
    return model, torch, ultralytics, device


def smoke(model, torch, device):
    from ultralytics.nn.autobackend import AutoBackend

    parameters = sum(p.numel() for p in model.model.parameters())
    with torch.inference_mode():
        backend = AutoBackend(weights=model.model, device=device).eval()
        sample = torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), device=device)
        outputs = list(output_tensors(backend(sample), torch.Tensor))
        if backend.device != device or not outputs or any(
            tensor.device != device or not torch.isfinite(tensor).all().item() for tensor in outputs
        ):
            raise RuntimeError("Expected finite synthetic outputs and backend on cuda:0.")
    print(f"PASS: {parameters:,} checkpoint parameters, bundled detection backend on {device}")
    print(f"PASS: Finite synthetic FP32 outputs {[tuple(t.shape) for t in outputs]}")


def prepare_split(split, manifest, destination):
    with manifest.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"Empty dataset manifest: {manifest.name}")
    for field, category in (("image", "images"), ("label", "labels")):
        folder = Path(category) / split
        paths = [Path(row[field]) for row in rows]
        if any(path.parent != folder or path.name in {"", ".", ".."} for path in paths):
            raise RuntimeError(f"Unexpected {field} path in {manifest.name}.")
        if len(paths) != len(set(paths)) or set(paths) != {
            entry.relative_to(DATASET) for entry in (DATASET / folder).iterdir()
        }:
            raise RuntimeError(f"{category}/{split} membership differs from the audited manifest.")
        (destination / folder).mkdir(parents=True)
        for row, relative in zip(rows, paths):
            copy = destination / relative
            shutil.copyfile(DATASET / relative, copy)
            if sha256(copy) != row[f"{field}_sha256"]:
                raise RuntimeError(f"Dataset checksum differs: {relative}")
    return len(rows), sum(int(row["objects"]) for row in rows)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n")


def evaluate(model, torch, ultralytics, device, checkpoint, split, run):
    from matplotlib import get_data_path

    manifest = MANIFESTS / f"gc10-{split}.csv"
    provenance = MANIFESTS / "gc10-provenance.json"
    audit = MANIFESTS / "gc10-audit.json"
    for source in (CONFIG, manifest, provenance, audit):
        shutil.copyfile(source, run / source.name)

    identity = {
        "status": "started",
        "split": split,
        "evaluator_sha256": sha256(Path(__file__)),
        "pinned_upstream_commit": UPSTREAM_COMMIT,
        "checkpoint": {"path": str(checkpoint), "sha256": CHECKPOINT_SHA256},
        "dataset": {
            "revision": json.loads(provenance.read_text())["revision"],
            "config_sha256": sha256(CONFIG),
            "split_manifest_sha256": sha256(manifest),
            "provenance_sha256": sha256(provenance),
            "audit_sha256": sha256(audit),
        },
    }
    write_json(run / "run.json", identity)
    write_json(run / "environment.json", {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "gpu_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        "nvidia_driver": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).splitlines()[0],
        "ultralytics": {"version": ultralytics.__version__, "path": ultralytics.__file__},
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
    })
    # The fork looks for Arial by filename and otherwise tries to fetch it.
    font = Path(os.environ["YOLO_CONFIG_DIR"]) / "Arial.ttf"
    shutil.copyfile(Path(get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf", font)

    with TemporaryDirectory(prefix="inputs-", dir=run) as temporary:
        inputs = Path(temporary)
        images, objects = prepare_split(split, manifest, inputs / "data" / "gc10-det")
        # check_det_dataset always checks the validation directory, even for test.
        (inputs / "data" / "gc10-det" / "images" / "val").mkdir(parents=True, exist_ok=True)
        staged_config = inputs / "configs" / CONFIG.name
        staged_config.parent.mkdir()
        shutil.copyfile(run / CONFIG.name, staged_config)

        def record_settings(validator):
            if validator.device != device or len(validator.dataloader.dataset) != images:
                raise RuntimeError("Validator device or loaded image count differs from the run inputs.")
            write_json(run / "effective_config.json", vars(validator.args))

        def verify_totals(validator):
            if validator.seen != images or int(validator.nt_per_class.sum()) != objects:
                raise RuntimeError("Validator image or object count differs from the audited split.")

        model.add_callback("on_val_start", record_settings)
        model.add_callback("on_val_end", verify_totals)
        print(f"Evaluating {split}: {images} manifest-verified images, {objects} objects.", flush=True)
        metrics = model.val(
            data=str(staged_config), split=split, imgsz=IMAGE_SIZE, device=device,
            project=str(run), name="plots", plots=True, exist_ok=False,
        )

    write_json(run / "metrics.json", {key: float(value) for key, value in metrics.results_dict.items()})
    write_json(run / "timing_ms_per_image.json", {key: float(value) for key, value in metrics.speed.items()})
    per_class = {
        int(index): (float(ap50), float(ap))
        for index, ap50, ap in zip(metrics.ap_class_index, metrics.box.ap50, metrics.box.ap)
    }
    with (run / "per_class.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("class_id", "class_name", "AP50", "AP50-95"))
        for index, name in sorted(model.names.items()):
            writer.writerow((index, name, *per_class.get(index, (None, None))))
    if not list((run / "plots").glob("*PR_curve.png")):
        print("No native precision-recall plot was generated. Check metrics and the terminal log.")
    identity.update(status="completed", images=images, objects=objects)
    write_json(run / "run.json", identity)
    print(f"Evaluation artifacts: {run}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--split", choices=("val", "test"))
    mode.add_argument("--smoke-test", action="store_true", help="Forward one synthetic input only.")
    parser.add_argument("--run-id", help="Unique experiment directory name for dataset evaluation.")
    args = parser.parse_args()
    if args.split and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_id or ""):
        parser.error("--split requires --run-id with letters, numbers, hyphens, or underscores.")
    if args.smoke_test and args.run_id:
        parser.error("--run-id is only used with --split.")
    checkpoint = args.checkpoint.resolve()
    verify_inputs(checkpoint)
    if args.smoke_test:
        with TemporaryDirectory(prefix="flawtrace-smoke-") as config:
            os.environ["YOLO_CONFIG_DIR"] = config
            model, torch, _, device = load_model(checkpoint)
            smoke(model, torch, device)
        return

    run = ROOT / "experiments" / args.run_id
    try:
        run.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.exit(1, f"Run directory already exists: {run}\n")
    with (run / "terminal.log").open("x", buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                with TemporaryDirectory(prefix="settings-", dir=run) as config:
                    os.environ["YOLO_CONFIG_DIR"] = config
                    model, torch, ultralytics, device = load_model(checkpoint)
                    evaluate(model, torch, ultralytics, device, checkpoint, args.split, run)
            except Exception:
                traceback.print_exc()
                raise SystemExit(1) from None
            except KeyboardInterrupt:
                print("Evaluation interrupted.", file=sys.stderr)
                raise SystemExit(130) from None


if __name__ == "__main__":
    main()
