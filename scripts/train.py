"""Manually run or resume the pinned AGE-YOLO baseline or a bounded benchmark."""

import argparse
import csv
import fcntl
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import yaml

from augment_data import DATA_CONFIG, DERIVED, MANIFESTS, ROOT, class_names, digest
from evaluate import Tee, UPSTREAM, UPSTREAM_COMMIT, git, write_json


CONFIG = ROOT / "configs" / "age_yolo_baseline.yaml"
MODEL = UPSTREAM / "ultralytics" / "cfg" / "models" / "age_yolo.yaml"
DEVICE = torch.device("cuda:0")
WARMUP = 10
MEASURED = 50
GPU_FIELDS = {
    "timestamp": "timestamp", "utilization.gpu": "utilization_pct",
    "memory.used": "memory_used_mib", "memory.total": "memory_total_mib",
    "power.draw": "power_w", "power.limit": "power_limit_w", "temperature.gpu": "temperature_c",
    "clocks.current.graphics": "graphics_mhz", "clocks.current.sm": "sm_mhz",
    "clocks.current.memory": "memory_mhz", "pstate": "pstate",
    "clocks_event_reasons.sw_power_cap": "sw_power_cap",
    "clocks_event_reasons.sw_thermal_slowdown": "sw_thermal_slowdown",
    "clocks_event_reasons.hw_thermal_slowdown": "hw_thermal_slowdown",
    "clocks_event_reasons.hw_power_brake_slowdown": "hw_power_brake_slowdown",
}


class BenchmarkFinished(Exception):
    """Stop the benchmark before the next training batch or validation."""


def select_visible_gpu(device, *_args, **_kwargs):
    """Select logical CUDA 0 without rewriting the scheduler's visibility mask."""
    if str(device) not in {"0", "cuda:0"} or not torch.cuda.is_available():
        raise RuntimeError("Training requires logical cuda:0 in the allocated GPU set.")
    return DEVICE


def nvidia_gpu_uuid():
    """Resolve the CUDA device to the same physical GPU in nvidia-smi."""
    return f"GPU-{torch.cuda.get_device_properties(DEVICE).uuid}"


def check_amp(model):
    """Check the native loss and scaled gradients on a disposable copy."""
    if next(model.parameters()).device != DEVICE:
        raise RuntimeError("AMP check requires the AGE-YOLO model on cuda:0.")
    probe = deepcopy(model).train()
    generator = torch.Generator(device=DEVICE).manual_seed(42)
    batch = {
        "img": torch.rand((1, 3, 640, 640), generator=generator, device=DEVICE),
        "batch_idx": torch.zeros(1, device=DEVICE),
        "cls": torch.zeros((1, 1), device=DEVICE),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]], device=DEVICE),
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
    print("PASS: AGE-YOLO FP16 autocast, native detection loss and scaled backward on cuda:0.")
    return True


def load_upstream(settings):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for AGE-YOLO training.")
    if git(UPSTREAM, "rev-parse", "HEAD") != UPSTREAM_COMMIT or git(UPSTREAM, "status", "--porcelain"):
        raise RuntimeError("Expected the clean, pinned AGE-YOLO checkout.")
    sys.path.insert(0, str(UPSTREAM))
    sys.dont_write_bytecode = True
    spec = importlib.util.find_spec("ultralytics")
    if spec is None or spec.origin is None or Path(spec.origin).resolve() != UPSTREAM / "ultralytics/__init__.py":
        raise RuntimeError("Expected the bundled AGE-YOLO Ultralytics implementation.")
    os.environ.update(
        YOLO_CONFIG_DIR=str(settings), YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false",
        YOLO_VERBOSE="true", MPLBACKEND="Agg",
    )
    import ultralytics
    from matplotlib import get_data_path

    shutil.copyfile(Path(get_data_path()) / "fonts/ttf/DejaVuSans.ttf", settings / "Arial.ttf")
    ultralytics.settings.update({
        key: False for key in (
            "sync", "hub", "clearml", "comet", "dvc", "mlflow", "neptune", "raytune",
            "tensorboard", "wandb",
        )
    })
    return ultralytics


def dataset_identity():
    config = yaml.safe_load(DATA_CONFIG.read_text())
    if set(config) != {"train", "val", "nc", "names"} or class_names(DATA_CONFIG) != class_names(ROOT / "configs/gc10.yaml"):
        raise RuntimeError("Training requires the original class mapping and a train/val-only YAML.")
    for split in ("train", "val"):
        expected = DERIVED / "images" / split
        if (DATA_CONFIG.parent / config[split]).resolve() != expected or not expected.is_dir():
            raise RuntimeError(f"Invalid {split} path. Prepare augmented GC10 first.")
    info = json.loads((DERIVED / "augmentation.json").read_text())
    if (
        info["augmentation_script_sha256"] != digest(ROOT / "scripts/augment_data.py")
        or info["train_manifest_sha256"] != digest(DERIVED / "train.csv")
        or any(digest(MANIFESTS / f"gc10-{split}.csv") != info["source_manifest_sha256"][split]
               for split in ("train", "val"))
    ):
        raise RuntimeError("Augmentation metadata differs. Run the augmentation verifier.")
    return info


def guard_trainer(native, run, info):
    if native.device != DEVICE or native.save_dir.resolve() != run:
        raise RuntimeError("Trainer must use cuda:0 and the reserved run directory.")
    if any(parameter.device != DEVICE for parameter in native.model.parameters()) or not native.amp:
        raise RuntimeError("Expected AGE-YOLO weights on cuda:0 and native AMP enabled.")
    if native.args.split != "val" or not native.args.val:
        raise RuntimeError("Checkpoint selection requires the validation split.")
    if (len(native.train_loader.dataset), len(native.test_loader.dataset)) != (info["train_images"], info["val_images"]):
        raise RuntimeError("Native train or validation image count differs from the audited inputs.")
    if Path(native.data["val"]).resolve() != DERIVED / "images/val":
        raise RuntimeError("Validator resolved an unexpected split.")
    write_json(run / "effective_config.json", {
        "args": vars(native.args), "amp": native.amp, "resume_start_epoch_index": native.start_epoch,
        "train_batch": native.train_loader.batch_size, "val_batch": native.test_loader.batch_size,
        "train_workers": native.train_loader.num_workers, "val_workers": native.test_loader.num_workers,
        "runtime": {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cublas_workspace_config": os.getenv("CUBLAS_WORKSPACE_CONFIG"),
            "train_pin_memory": native.train_loader.pin_memory,
            "train_prefetch_factor": native.train_loader.prefetch_factor,
        },
    })


def validation_memory(native, run):
    """Measure the native training validator while training state remains on CUDA."""
    loader = native.test_loader
    if native.args.split != "val" or loader.batch_size != 32 or not native.amp:
        raise RuntimeError("Expected the native FP16 validation loader with batch size 32.")
    torch.cuda.synchronize(DEVICE)
    free, total = torch.cuda.mem_get_info(DEVICE)
    summary = {
        "status": "failed", "split": "val", "batch": loader.batch_size,
        "workers": loader.num_workers, "precision": "FP16", "start_unix": time.time(),
        "start_device_free_bytes": free, "device_total_bytes": total,
        "start_torch_allocated_bytes": torch.cuda.memory_allocated(DEVICE),
        "start_torch_reserved_bytes": torch.cuda.memory_reserved(DEVICE), "input_shapes": [],
    }

    def record_shape(_, inputs):
        summary["input_shapes"].append(list(inputs[0].shape))

    torch.cuda.reset_peak_memory_stats(DEVICE)
    hook = native.ema.ema.register_forward_pre_hook(record_shape)
    try:
        native.validator(trainer=native)
        torch.cuda.synchronize(DEVICE)
        if sum(shape[0] for shape in summary["input_shapes"]) != len(loader.dataset):
            raise RuntimeError("Native validation did not consume the full validation split.")
        summary["status"] = "completed"
    except torch.cuda.OutOfMemoryError:
        summary["status"] = "cuda-out-of-memory"
        traceback.print_exc()
    finally:
        hook.remove()
        summary.update(
            end_unix=time.time(), peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(DEVICE),
            peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(DEVICE),
        )
        write_json(run / "validation_memory.json", summary)
    return summary["status"]


def benchmark(model, run, options, sync_each_batch=False, check_validation=False):
    """Time the native loop and optionally check validation before any checkpoint writes."""
    timings, finite_losses = [], []
    index = 0
    previous_end = start = loader_wait = 0.0
    batch_start_unix = thread_cpu_start = 0.0
    training_start = window_start = window_end = 0.0
    start_clock = end_clock = 0.0
    validation_status = None
    peaks = {}

    def ready(native):
        nonlocal previous_end, training_start
        if len(native.train_loader) <= WARMUP + MEASURED or native.train_loader.batch_size != 16:
            raise RuntimeError("Benchmark requires at least 61 training batches of size 16.")
        torch.cuda.reset_peak_memory_stats(DEVICE)
        previous_end = time.perf_counter()
        training_start = time.time()

    def batch_start(native):
        nonlocal start, window_start, start_clock, loader_wait, batch_start_unix, thread_cpu_start
        loader_wait = time.perf_counter() - previous_end
        if index == WARMUP:
            torch.cuda.synchronize(DEVICE)
            window_start = time.time()
            start_clock = time.perf_counter()
            loader_wait = 0.0  # The measurement starts after this batch is delivered.
        start = time.perf_counter()
        batch_start_unix = time.time()
        thread_cpu_start = time.thread_time()

    def batch_end(native):
        nonlocal index, previous_end, window_end, end_clock, validation_status
        finite = native.loss_items.isfinite().all()
        if sync_each_batch:
            if not finite.item():
                raise RuntimeError("Benchmark encountered a non-finite training loss.")
            torch.cuda.synchronize(DEVICE)
        else:
            finite_losses.append(finite)
        if index + 1 == WARMUP + MEASURED:
            torch.cuda.synchronize(DEVICE)
            window_end = time.time()
            end_clock = time.perf_counter()
        previous_end = time.perf_counter()
        if index >= WARMUP:
            timings.append((
                index + 1, time.time(), loader_wait, previous_end - start, native.accumulate,
                batch_start_unix, time.thread_time() - thread_cpu_start,
            ))
        index += 1
        if index == WARMUP + MEASURED:
            if finite_losses and not torch.stack(finite_losses).all().item():
                raise RuntimeError("Benchmark encountered a non-finite training loss.")
            peaks.update(
                peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(DEVICE),
                peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(DEVICE),
            )
            if check_validation:
                validation_status = validation_memory(native, run)
            raise BenchmarkFinished

    model.add_callback("on_train_start", ready)
    model.add_callback("on_train_batch_start", batch_start)
    model.add_callback("on_train_batch_end", batch_end)
    with (run / "gpu.csv").open("x") as gpu:
        gpu.write(",".join(GPU_FIELDS.values()) + "\n")
        gpu.flush()
        monitor = subprocess.Popen([
            "nvidia-smi", "-i", nvidia_gpu_uuid(), "--query-gpu=" + ",".join(GPU_FIELDS),
            "--format=csv,noheader,nounits", "-l", "1",
        ], stdout=gpu)
        try:
            model.train(**options)
        except BenchmarkFinished:
            pass
        else:
            raise RuntimeError("Benchmark exited before its measured batch limit.")
        finally:
            monitor.terminate()
            monitor.wait()
    training_samples, timing_samples = [], []
    validation_samples = []
    validation_path = run / "validation_memory.json"
    validation = json.loads(validation_path.read_text()) if check_validation else None
    with (run / "gpu.csv").open(newline="") as stream:
        for row in csv.DictReader(stream, skipinitialspace=True):
            sample_time = datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f").timestamp()
            if training_start <= sample_time <= window_end:
                training_samples.append(row)
            if window_start <= sample_time <= window_end:
                timing_samples.append(row)
            if validation and validation["start_unix"] <= sample_time <= validation["end_unix"]:
                validation_samples.append(row)

    def readings(samples, field):
        return [float(row[field]) for row in samples if row[field] not in {"N/A", "[N/A]", "Not Supported", "[Not Supported]"}]

    utilization = readings(timing_samples, "utilization_pct")
    memory = readings(training_samples, "memory_used_mib")
    if validation:
        validation["sampled_device_peak_vram_mib"] = max(readings(validation_samples, "memory_used_mib"), default=None)
        validation["gpu_samples"] = len(validation_samples)
        write_json(validation_path, validation)
    with (run / "batch_timing.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "batch", "end_unix", "loader_wait_seconds", "batch_body_wall_seconds", "accumulate",
            "start_unix", "main_thread_cpu_seconds",
        ))
        writer.writerows(timings)
    wait_seconds = statistics.mean(row[2] for row in timings)
    body_seconds = statistics.mean(row[3] for row in timings)
    batch_seconds = (end_clock - start_clock) / MEASURED
    epoch_seconds = batch_seconds * len(model.trainer.train_loader)
    write_json(run / "benchmark.json", {
        "batch": 16, "warmup_batches": WARMUP, "measured_batches": MEASURED,
        "sync_each_batch": sync_each_batch, "window_start_unix": window_start, "window_end_unix": window_end,
        "accumulation_values": sorted({row[4] for row in timings}),
        "sampled_gpu_utilization_mean_pct": statistics.mean(utilization) if utilization else None,
        "sampled_device_peak_vram_mib": max(memory, default=None), "measured_window_gpu_samples": len(timing_samples),
        **peaks,
        "mean_loader_wait_seconds": wait_seconds, "mean_batch_body_wall_seconds": body_seconds,
        "mean_batch_total_seconds": batch_seconds, "training_images_per_second": 16 / batch_seconds,
        "estimated_training_seconds_per_epoch": epoch_seconds,
        "estimated_300_epoch_training_hours": epoch_seconds * 300 / 3600,
    })
    if validation_status == "cuda-out-of-memory":
        raise RuntimeError(f"Native validation ran out of CUDA memory. See {validation_path}.")


def run_training(args, run, ultralytics, options):
    from ultralytics.engine import trainer, validator

    info = dataset_identity()
    gpu_uuid = nvidia_gpu_uuid()
    identity = {
        "mode": args.mode, "repository_commit": git(ROOT, "rev-parse", "HEAD"),
        "upstream_commit": UPSTREAM_COMMIT, "script_sha256": digest(Path(__file__)),
        "model_yaml_sha256": digest(MODEL), "baseline_config_sha256": digest(CONFIG),
        "data_config_sha256": digest(DATA_CONFIG), "requirements_sha256": digest(ROOT / "requirements.txt"),
        "dataset": info,
        "dataset_provenance_sha256": digest(MANIFESTS / "gc10-provenance.json"),
    }
    if args.mode == "benchmark":
        identity["diagnostics"] = {
            "sync_each_batch": args.sync_each_batch, "validation_memory": args.validation_memory,
        }
    record = run / "run.json"
    last = run / "weights" / "last.pt"
    if args.resume:
        original = json.loads(record.read_text())
        resume_commit = identity["repository_commit"]
        identity["repository_commit"] = original["repository_commit"]
        if original != {**identity, "status": "started"} or not last.is_file() or last.is_symlink():
            raise RuntimeError("Resume requires a matching unfinished baseline and its local weights/last.pt.")
        model = ultralytics.YOLO(str(last), task="detect")
        saved = model.ckpt["train_args"]
        if (model.ckpt["epoch"] < 0 or model.ckpt["optimizer"] is None
                or Path(saved["data"]).resolve() != DATA_CONFIG
                or Path(saved["project"]).resolve() / saved["name"] != run
                or saved["split"] != "val" or not saved["exist_ok"]
                or any(saved[key] != value for key, value in options.items())):
            raise RuntimeError("last.pt is finalized or its training arguments differ from this run.")
        print(f"Resuming {last.name} at FlawTrace commit {resume_commit}.")
    else:
        for file in (CONFIG, DATA_CONFIG, MODEL, UPSTREAM / "ultralytics/cfg/default.yaml"):
            shutil.copyfile(file, run / file.name)
        write_json(record, {**identity, "status": "started"})
        write_json(run / "environment.json", {
            "python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(DEVICE),
            "gpu_uuid": gpu_uuid, "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            "gpu_memory_bytes": torch.cuda.get_device_properties(DEVICE).total_memory,
            "nvidia_driver": subprocess.check_output(
                ["nvidia-smi", "-i", gpu_uuid, "--query-gpu=driver_version", "--format=csv,noheader"], text=True
            ).strip(),
            "ultralytics": ultralytics.__version__,
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        })
        model = ultralytics.YOLO(str(MODEL), task="detect")
    model.add_callback("on_pretrain_routine_end", lambda native: guard_trainer(native, run, info))
    options = {
        **options, "data": str(DATA_CONFIG), "split": "val", "val": True,
        "project": str(run.parent), "name": run.name, "exist_ok": True,
    }
    print(f"Using {DEVICE} ({gpu_uuid}); CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')!r}.")
    # Keep the local AMP check and preserve GPU visibility in training and final validation.
    with (
        patch.object(trainer, "check_amp", check_amp),
        patch.object(trainer, "select_device", select_visible_gpu),
        patch.object(validator, "select_device", select_visible_gpu),
    ):
        if args.mode == "benchmark":
            benchmark(model, run, options, args.sync_each_batch, args.validation_memory)
        else:
            model.train(**options, resume=args.resume)
            if not all(path.is_file() for path in (last, run / "weights/best.pt", run / "results.csv")):
                raise RuntimeError("Training returned without its expected checkpoints and results.")
    write_json(record, {**identity, "status": "completed"})
    print(f"Artifacts: {run}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("check-amp", "benchmark", "full"))
    parser.add_argument("--run-id", help="A new run ID, or the interrupted full run ID with --resume.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sync-each-batch", action="store_true", help="Reproduce the old benchmark synchronization.")
    parser.add_argument("--validation-memory", action="store_true", help="After benchmark timing, measure native validation.")
    args = parser.parse_args()
    if (args.sync_each_batch or args.validation_memory) and args.mode != "benchmark":
        parser.error("Diagnostic flags require --mode benchmark.")
    options = yaml.safe_load(CONFIG.read_text())
    if options.get("device") != 0 or options.get("pretrained") is not False:
        parser.error("The baseline must select CUDA 0 and initialize from YAML.")
    if args.mode == "check-amp":
        if args.run_id or args.resume:
            parser.error("The AMP check does not accept --run-id or --resume.")
        with TemporaryDirectory(prefix="age-yolo-amp-") as config:
            upstream = load_upstream(Path(config))
            from ultralytics.cfg import get_cfg

            torch.manual_seed(options["seed"])
            model = upstream.YOLO(str(MODEL), task="detect").model.to(DEVICE)
            model.args = get_cfg(overrides=options)
            check_amp(model)
        return
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_id or ""):
        parser.error("Supply a run ID using letters, numbers, hyphens, or underscores.")
    if args.resume and args.mode != "full":
        parser.error("Only full training can resume.")
    parent = ROOT / "experiments"
    if args.mode == "benchmark":
        parent /= "benchmarks"
    run = parent / args.run_id
    if args.resume:
        if run.is_symlink() or not (run / "run.json").is_file():
            parser.error("The requested run does not exist.")
    else:
        try:
            run.mkdir(parents=True)
        except FileExistsError:
            parser.error(f"Run already exists: {run}")
    with (run / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("The run is already active.")
        if args.resume and json.loads((run / "run.json").read_text())["status"] != "started":
            parser.error("Completed runs cannot resume.")
        with (run / "terminal.log").open("a", buffering=1) as log:
            with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                try:
                    with TemporaryDirectory(prefix="settings-", dir=run) as config:
                        run_training(args, run, load_upstream(Path(config)), options)
                except KeyboardInterrupt:
                    print("Training interrupted. Keep this run and resume from the last saved epoch.", file=sys.stderr)
                    raise SystemExit(130) from None
                except Exception:
                    traceback.print_exc()
                    raise SystemExit(1) from None


if __name__ == "__main__":
    main()
