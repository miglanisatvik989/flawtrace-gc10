"""Train or resume the bundled AGE-YOLO detector using configs/hpc.yaml."""

import argparse
import csv
import fcntl
import json
import os
import pickle
import re
import shutil
import sys
import traceback
import time
import platform
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from evaluate import ROOT, UPSTREAM, UPSTREAM_COMMIT, Tee, sha256, write_json


TRAIN_KEYS = {
    "SEED": "seed", "BATCH_SIZE": "batch", "EPOCHS": "epochs", "IMAGE_SIZE": "imgsz",
    "PATIENCE": "patience", "PRETRAINED": "pretrained", "AMP": "amp",
    "DETERMINISTIC": "deterministic", "NUM_WORKERS": "workers", "OPTIMIZER": "optimizer",
    "LR": "lr0", "MOMENTUM": "momentum", "WEIGHT_DECAY": "weight_decay", "MOSAIC": "mosaic",
    "FLIP_LR": "fliplr", "FLIP_UD": "flipud", "SCALE": "scale",
    "HSV_H": "hsv_h", "HSV_S": "hsv_s", "HSV_V": "hsv_v",
}


def load_config(path):
    text = path.read_text()
    # PyYAML normally keeps only the last occurrence of a duplicate key.
    node = yaml.compose(text, Loader=yaml.SafeLoader)
    if not isinstance(node, yaml.MappingNode):
        raise ValueError("Expected a flat YAML mapping.")
    keys = [key.value for key, _ in node.value if isinstance(key, yaml.ScalarNode)]
    expected = set(TRAIN_KEYS) | {"DEVICE", "RUN_ID", "RESUME", "DATA_CONFIG", "MODEL_CONFIG", "OUTPUT_DIR"}
    if len(set(keys)) != len(node.value):
        raise ValueError("Duplicate or non-scalar configuration key.")
    if set(keys) != expected:
        raise ValueError(f"Missing keys: {sorted(expected - set(keys))}; unsupported keys: {sorted(set(keys) - expected)}")
    config = yaml.safe_load(text)
    if not isinstance(config["DEVICE"], str) or not re.fullmatch(r"cuda:(0|[1-9][0-9]*)", config["DEVICE"]):
        raise ValueError("DEVICE must be an indexed CUDA device, for example cuda:0.")
    for key in ("AMP", "RESUME", "DETERMINISTIC", "PRETRAINED"):
        if type(config[key]) is not bool:
            raise ValueError(f"{key} must be a YAML boolean.")
    options = {native: config[key] for key, native in TRAIN_KEYS.items()}
    if config["PRETRAINED"] or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", config["RUN_ID"]):
        raise ValueError("Use random initialization and a simple run ID.")
    if type(options["batch"]) is not int or options["batch"] < 1:
        raise ValueError("BATCH_SIZE must be a positive integer.")
    return config, options


def initialize_cuda(name):
    device = torch.device(name)
    if not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
        raise RuntimeError(f"Requested {name} is unavailable.")
    torch.cuda.set_device(device)
    free, total = torch.cuda.mem_get_info(device)
    print(f"{device} | {torch.cuda.get_device_name(device)} | {free / 2**30:.2f} of {total / 2**30:.2f} GiB free")
    return device


def check_amp(model, device):
    """Check the native loss and scaled gradients on a disposable copy."""
    if next(model.parameters()).device != device:
        raise RuntimeError(f"AMP check requires the AGE-YOLO model on {device}.")
    probe = deepcopy(model).train()
    generator = torch.Generator(device=device).manual_seed(42)
    batch = {
        "img": torch.rand((1, 3, 640, 640), generator=generator, device=device),
        "batch_idx": torch.zeros(1, device=device),
        "cls": torch.zeros((1, 1), device=device),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]], device=device),
    }
    with torch.autocast("cuda", dtype=torch.float16):
        predictions = probe(batch["img"])
        if not any(t.dtype == torch.float16 for t in predictions):
            raise RuntimeError("AGE-YOLO autocast did not produce FP16 features.")
        loss, components = probe.loss(batch, preds=predictions)
    if not torch.isfinite(loss).all().item() or not torch.isfinite(components).all().item():
        raise RuntimeError("AGE-YOLO autocast produced a non-finite detection loss.")
    torch.amp.GradScaler("cuda", init_scale=128).scale(loss).backward()
    grads = [p.grad for p in probe.parameters() if p.requires_grad and p.grad is not None]
    if not grads or not torch.stack([g.isfinite().all() for g in grads]).all().item():
        raise RuntimeError("AGE-YOLO autocast produced missing or non-finite gradients.")
    print(f"PASS: AGE-YOLO FP16 autocast, native detection loss and scaled backward on {device}.")
    return True


def restore_results(run, record, checkpoint):
    results = run / "results.csv"
    saved = checkpoint["train_results"]
    rows = [list(saved)] + [[format(value, ".6g") for value in row] for row in zip(*saved.values())]
    if results.is_file():
        with results.open(newline="") as stream:
            if list(csv.reader(stream)) == rows:
                return
        shutil.copyfile(results, record.with_suffix(".results-before-resume.csv"))
    temporary = results.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)
    temporary.replace(results)


def backup_checkpoint(native):
    temporary = native.wdir / ".last.recovery.tmp"
    shutil.copyfile(native.last, temporary)
    temporary.replace(native.wdir / "last.recovery.pt")


def train(config, options, run, model_yaml, data_yaml, device):
    settings = run / ".settings"
    settings.mkdir(exist_ok=True)
    os.environ.update(YOLO_CONFIG_DIR=str(settings), YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false")
    sys.path.insert(0, str(UPSTREAM))
    from ultralytics import YOLO, __version__
    from ultralytics.engine import trainer
    from matplotlib import get_data_path

    shutil.copyfile(Path(get_data_path()) / "fonts/ttf/DejaVuSans.ttf", settings / "Arial.ttf")
    snapshot = run / "hpc.yaml"
    record = run / "run.json"
    if config["RESUME"]:
        if json.loads(record.read_text())["status"] != "started":
            raise ValueError("Completed experiments cannot resume.")
        previous = yaml.safe_load(snapshot.read_text())
        if config != {**previous, "RESUME": True}:
            raise ValueError("Resume requires the same experiment configuration.")
        for filename in ("last.pt", "last.recovery.pt"):
            checkpoint = run / "weights" / filename
            try:
                model = YOLO(str(checkpoint.resolve(strict=True)), task="detect")
            except (OSError, EOFError, RuntimeError, pickle.UnpicklingError) as error:
                print(f"Cannot read {filename}: {error}")
                continue
            if model.ckpt["optimizer"] is None or model.ckpt["epoch"] < 0:
                raise ValueError("Checkpoint is finalized and cannot resume.")
            saved = model.ckpt["train_args"]
            if Path(saved["project"]) / saved["name"] != run:
                raise ValueError("Checkpoint belongs to a different experiment.")
            if (any(saved.get(key) != value for key, value in options.items())
                    or Path(saved["data"]).resolve() != data_yaml or saved["split"] != "val"):
                raise ValueError("Checkpoint settings differ from this experiment.")
            restore_results(run, run / f"resume-{datetime.now():%Y%m%dT%H%M%S%f}.json", model.ckpt)
            break
        else:
            raise FileNotFoundError(f"No readable checkpoint in {run / 'weights'}")
    else:
        snapshot.write_text(yaml.safe_dump(config, sort_keys=False))
        for source in (data_yaml, model_yaml):
            shutil.copyfile(source, run / source.name)
        write_json(run / "environment.json", {
            "torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device),
            "gpu": torch.cuda.get_device_name(device), "ultralytics": __version__,
            "pinned_upstream_commit": UPSTREAM_COMMIT,
        })
        write_json(record, {"status": "started"})
        model = YOLO(str(model_yaml), task="detect")

    def record_settings(native):
        # The fork serializes its device to text. Pass the torch.device back to
        # final validation so its selector preserves the original CUDA visibility.
        native.validator.args.device = device
        write_json(run / "effective_config.json", {
            "args": vars(native.args), "amp": native.amp, "train_batch": native.train_loader.batch_size,
            "val_batch": native.test_loader.batch_size, "gradient_accumulation": native.accumulate,
            "optimizer_weight_decay": [g["weight_decay"] for g in native.optimizer.param_groups],
        })

    model.add_callback("on_pretrain_routine_end", record_settings)
    model.add_callback("on_model_save", backup_checkpoint)
    # The pinned fork's default AMP probe loads a separate detector from the network.
    with patch.object(trainer, "check_amp", partial(check_amp, device=device)):
        model.train(**options, device=device, data=str(data_yaml), val=True, split="val",
                    project=str(run.parent), name=run.name, exist_ok=True, resume=config["RESUME"])
    write_json(record, {"status": "completed"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/hpc.yaml")
    parser.add_argument("--preflight", action="store_true", help="Resolve settings without using a GPU or creating a run.")
    args = parser.parse_args()
    config, options = load_config(args.config)
    data_yaml = (ROOT / config["DATA_CONFIG"]).resolve(strict=True)
    model_yaml = (ROOT / config["MODEL_CONFIG"]).resolve(strict=True)
    if model_yaml.suffix not in {".yaml", ".yml"}:
        parser.error("Random initialization requires a model YAML.")
    run = (ROOT / config["OUTPUT_DIR"]).resolve() / config["RUN_ID"]
    if args.preflight:
        print(json.dumps({**config, "data": str(data_yaml), "model": str(model_yaml), "run": str(run)}, indent=2))
        return

    deployment = ROOT / ".cache/source.json"
    if not deployment.is_file():
        parser.error("Run python scripts/record_source.py on the deployment host before training.")
    source = json.loads(deployment.read_text())
    if any(not (ROOT / name).is_file() or sha256(ROOT / name) != digest
           for name, digest in source["sha256"].items()):
        raise ValueError("Deployment changed. Run scripts/record_source.py before submission.")
    device = initialize_cuda(config["DEVICE"])
    if config["RESUME"]:
        if run.is_symlink() or not (run / "hpc.yaml").is_file():
            raise ValueError(f"Missing original run configuration in {run}")
    else:
        run.mkdir(parents=True, exist_ok=False)
    with (run / ".lock").open("a") as lock, (run / "terminal.log").open("a", buffering=1) as log:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            started = datetime.now(timezone.utc)
            stamp = started.strftime("%Y%m%dT%H%M%S%fZ")
            write_json(run / f"source-{stamp}.json", source)
            attempt = run / f"attempt-{stamp}.json"
            dataset = json.loads((ROOT / "data/manifests/gc10-provenance.json").read_text())
            train_images = (data_yaml.parent / yaml.safe_load(data_yaml.read_text())["train"]).resolve()
            augmented = train_images.parents[1]
            info = {
                "start_utc": started.isoformat(), "status": "started", "command": [sys.executable, *sys.argv],
                "cwd": str(Path.cwd()), "python": platform.python_version(), "torch": torch.__version__,
                "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device),
                "gpu_uuid": str(torch.cuda.get_device_properties(device).uuid),
                "free_vram_bytes": torch.cuda.mem_get_info(device)[0], "pbs_job_id": os.getenv("PBS_JOBID"),
                "dataset_revision": dataset["revision"], "train_manifest_sha256": sha256(augmented / "train.csv"),
                "augmentation": json.loads((augmented / "augmentation.json").read_text()),
            }
            write_json(attempt, info)
            start_time = time.monotonic()
            try:
                train(config, options, run, model_yaml, data_yaml, device)
                info["checkpoint_sha256"] = {
                    name: sha256(run / "weights" / name) for name in ("best.pt", "last.pt")
                }
                info["status"] = "completed"
            except (Exception, KeyboardInterrupt) as error:
                info["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
                traceback.print_exc()
                raise SystemExit(1) from None
            finally:
                info["finish_utc"] = datetime.now(timezone.utc).isoformat()
                info["duration_seconds"] = time.monotonic() - start_time
                write_json(attempt, info)


if __name__ == "__main__":
    main()
