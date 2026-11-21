"""Evaluate frozen B1 on two fixed, overlapping horizontal crops per image."""

import argparse
import math
import os
import platform
import random
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import yaml

from analyze_errors import box, match, ratio, read_csv, save_csv
from evaluate import DATASET, MANIFESTS, ROOT, UPSTREAM_COMMIT, Tee, load_model, sha256, write_json
from evaluate_e2 import diagnostics, figures, paired_bootstrap, verify_inputs
from evaluate_fusion import combined, fused_records
from evaluate_resolution import MATCH_SETTINGS, native_ap, read_json

WIDTH, HEIGHT = 2048, 1000
CONDITIONS = {"A": (640, 640), "B": (640, 960), "D": (960, 960), "T": (640, 640)}
COMPARISONS = (("T", "A"), ("T", "B"), ("T", "D"))
E2_RUN = ROOT / "experiments/e2-resolution-eval-01"
E3_HASH = "7e61fc3da770652acfd58264c06b31480e04b6ac6440e79e3eae90f648c92275"
BUDGETS = (.10, .25, .50, 1.00)
REVIEWED_E2_HASHES = {
    "summary.json": "3045ac08181feb00ccd64fc30030847a66a70934fd84e430425e2a4139bad305",
    "per_object_comparison.csv": "7f14ade55c2b3441ff87cb4370945688bc9480c59c918634f08766e76231f008",
    "predictions/960.json": "3f2393c49c3ae5bfb4171a3b5a0a58993a6a6c6d364cf2440a81a9092da3cae0",
    "predictions/960-ap.npz": "4b7329362daccfd9504ec209fae669fbc00091701fdea2d0895010dc876902b6",
}


def rectangles(conf, tiled):
    if not tiled:
        return (("full", 0, 0, WIDTH, HEIGHT),)
    width, height = conf["CROP_WIDTH"], conf["CROP_HEIGHT"]
    return (("left", 0, 0, width, height), ("right", width-conf["CROP_OVERLAP"], 0, WIDTH, height))


def boundary(obj, crops):
    x1, _, x2, _ = box(obj)
    left_end, right_start = crops[0][3], crops[1][1]
    left = x2 <= left_end
    right = x1 >= right_start
    fraction = max(min(x2, left_end)-x1, x2-max(x1, right_start)) / (x2-x1)
    return {"contained_left": left, "contained_right": right, "contained_either": left or right,
            "partial_both": not (left or right), "intersects_overlap": x1 < left_end and x2 > right_start,
            "within_overlap": left and right,
            "crosses_boundary": x1 < left_end < x2 or x1 < right_start < x2,
            "max_visible_area_fraction": fraction}


def crop_input(image, crop, size, stride):
    _, x1, y1, x2, y2 = crop
    width, height = x2-x1, y2-y1
    gain = size / max(width, height)
    resized_w, resized_h = min(math.ceil(width*gain), size), min(math.ceil(height*gain), size)
    shape_h, shape_w = (math.ceil(dim*gain/stride + .5)*stride for dim in (height, width))
    resized = cv2.resize(image[y1:y2, x1:x2], (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    left = round((shape_w-resized_w)/2-.1)
    top = round((shape_h-resized_h)/2-.1)
    padded = cv2.copyMakeBorder(resized, top, shape_h-resized_h-top, left, shape_w-resized_w-left,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return np.ascontiguousarray(padded.transpose(2, 0, 1)[::-1]), {
        "crop": crop[0], "crop_x1": x1, "crop_y1": y1, "crop_x2": x2, "crop_y2": y2,
        "resized_width": resized_w, "resized_height": resized_h, "gain_x": resized_w/width,
        "gain_y": resized_h/height, "pad_left": left, "pad_top": top,
        "pad_right": shape_w-resized_w-left, "pad_bottom": shape_h-resized_h-top,
        "tensor_chw": str([3, shape_h, shape_w]),
    }


def remap(detections, geometry):
    """Remove padding and per-axis resize, clip within crop, then offset."""
    rows = np.asarray(detections, dtype=float).reshape(-1, 6).copy()
    for axis, key in ((0, "x"), (1, "y")):
        origin = geometry[f"crop_{key}1"]
        extent = geometry[f"crop_{key}2"] - origin
        pad = geometry["pad_left" if key == "x" else "pad_top"]
        rows[:, [axis, axis+2]] = np.clip((rows[:, [axis, axis+2]]-pad)/geometry[f"gain_{key}"], 0, extent) + origin
    return [dict(zip(("x1", "y1", "x2", "y2", "score"), row[:5]), class_id=int(row[5]), crop=geometry["crop"])
            for row in rows.tolist() if row[2] > row[0] and row[3] > row[1]]


def merge(rows, threshold, maximum):
    """Score-ordered class-aware NMS in original-image coordinates."""
    import torch
    from torchvision.ops import batched_nms

    if not rows:
        return []
    boxes = torch.tensor([box(row) for row in rows], dtype=torch.float64, device="cpu")
    scores = torch.tensor([row["score"] for row in rows], dtype=torch.float64, device="cpu")
    classes = torch.tensor([row["class_id"] for row in rows], device="cpu")
    indices = batched_nms(boxes, scores, classes, threshold)[:maximum].tolist()
    return [rows[index] for index in indices]


def infer_group(names, tiled, batch_size, conf, backend, torch, device, pool=None):
    from ultralytics.utils.ops import non_max_suppression

    start = time.perf_counter()
    crops = rectangles(conf, tiled)

    def read(name):
        image = cv2.imread(str(DATASET / name))
        if image is None or image.shape[:2] != (HEIGHT, WIDTH):
            raise ValueError(f"Missing or unexpected validation image size: {name}")
        return [crop_input(image, crop, conf["IMAGE_SIZE"], backend.stride) for crop in crops]

    entries = [(name, image, geometry)
               for name, inputs in zip(names, pool.map(read, names) if pool else map(read, names))
               for image, geometry in inputs]
    prepared = time.perf_counter()
    stages = {"decode_crop_resize_ms": 1000*(prepared-start), "transfer_ms": 0., "inference_ms": 0., "nms_remap_ms": 0.}
    predictions = {name: [] for name in names}
    shapes = set()
    for start_index in range(0, len(entries), batch_size):
        group = entries[start_index:start_index+batch_size]
        began = time.perf_counter()
        images = torch.from_numpy(np.stack([image for _, image, _ in group])).to(device).float().div_(255)
        torch.cuda.synchronize(device)
        transferred = time.perf_counter()
        raw = backend(images, augment=False)
        torch.cuda.synchronize(device)
        inferred = time.perf_counter()
        outputs = non_max_suppression(raw, conf["NMS_CONFIDENCE_FLOOR"], conf["NMS_IOU"],
                                      multi_label=True, agnostic=False, max_det=conf["MAX_DETECTIONS"])
        for (name, _, geometry), pred in zip(group, outputs, strict=True):
            predictions[name].extend(remap(pred.cpu().numpy(), geometry))
        torch.cuda.synchronize(device)
        stages["transfer_ms"] += 1000*(transferred-began)
        stages["inference_ms"] += 1000*(inferred-transferred)
        stages["nms_remap_ms"] += 1000*(time.perf_counter()-inferred)
        shapes.add(tuple(images.shape))
    began = time.perf_counter()
    merged = {name: merge(rows, conf["NMS_IOU"], conf["MAX_DETECTIONS"]) if tiled else rows
              for name, rows in predictions.items()}
    stages["merge_ms"] = 1000*(time.perf_counter()-began)
    stages["end_to_end_ms"] = 1000*(time.perf_counter()-start)
    return predictions, merged, [dict(geometry, image=name) for name, _, geometry in entries], stages, shapes


def measure(names, conf, backend, torch, device, pool, tiled, batch_size, pass_id):
    warmup, samples = 5, 40
    sample_names = random.Random(conf["SEED"]).sample(names, warmup + samples)
    per_group = batch_size // len(rectangles(conf, tiled)) if batch_size > 1 else 1
    warmup_names = sample_names[:warmup] if batch_size == 1 else names[:warmup*per_group]
    for start in range(0, len(warmup_names), per_group):
        infer_group(warmup_names[start:start+per_group], tiled, batch_size, conf, backend, torch, device,
                    pool if batch_size > 1 else None)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    durations, shapes = [], set()
    selected = sample_names[warmup:] if batch_size == 1 else names
    for start in range(0, len(selected), per_group):
        *_, timing, batch_shapes = infer_group(selected[start:start+per_group], tiled, batch_size, conf,
                                              backend, torch, device, pool if batch_size > 1 else None)
        durations.append(timing)
        shapes.update(batch_shapes)
    totals = {key: sum(row[key] for row in durations) for key in durations[0]}
    seconds = totals["end_to_end_ms"] / 1000
    return {"condition": "T" if tiled else "A", "pass": pass_id, "view_batch": batch_size,
            "original_images": len(selected), "crop_images": 2*len(selected) if tiled else 0,
            **{f"{key}_per_original": value / len(selected) for key, value in totals.items()},
            "original_images_per_second": len(selected)/seconds,
            "crop_images_per_second": 2*len(selected)/seconds if tiled else None,
            "median_batch1_ms": float(np.median([row["end_to_end_ms"] for row in durations])) if batch_size == 1 else None,
            "p95_batch1_ms": float(np.quantile([row["end_to_end_ms"] for row in durations], .95)) if batch_size == 1 else None,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "input_shapes_nchw": str(sorted(shapes))}


def load_inputs(conf):
    e2_conf = yaml.safe_load((ROOT / "configs/evaluation/age_yolo_e2.yaml").read_text())
    prior, truth, _, b1, guesses, native, hashes = verify_inputs(e2_conf)
    checkpoint = (ROOT / conf["CHECKPOINT"]).resolve(strict=True)
    if sha256(checkpoint) != E3_HASH or b1["checkpoint_sha256"] != E3_HASH:
        raise ValueError("Expected the frozen B1 best.pt.")
    for name, digest in REVIEWED_E2_HASHES.items():
        if sha256(E2_RUN / name) != digest:
            raise ValueError(f"E2 artifact differs from reviewed archive: {name}")
    e2 = read_json(E2_RUN / "summary.json")
    if (e2["status"] != "completed" or e2["B1_checkpoint_sha256"] != E3_HASH
            or e2["dataset_manifest_sha256"] != prior["dataset_manifest_sha256"]
            or e2["d1_objects_sha256"] != sha256(ROOT / "experiments/d1-baseline-01/per_object.csv")
            or e2["upstream_commit"] != UPSTREAM_COMMIT):
        raise ValueError("E2 results differ from frozen B1 or D1 inputs.")
    from evaluate_fusion import read_inputs

    e2_guesses, e2_native = read_inputs(E2_RUN, truth)
    d_run = read_json(E2_RUN / "960/run.json")
    if (d_run["status"] != "completed" or d_run["split"] != "val"
            or d_run["checkpoint"]["sha256"] != e2["E2_checkpoint_sha256"]
            or d_run["dataset"]["split_manifest_sha256"] != prior["dataset_manifest_sha256"]["val"]):
        raise ValueError("E2 condition D provenance differs.")
    for size, condition in ((640, "A"), (960, "B")):
        for key in ("mAP50", "mAP50-95"):
            if b1["metrics"][str(size)][key] != e2["metrics"][condition][key]:
                raise ValueError(f"Archived parity differs for {condition}.")
    for name in ("summary.json", "per_object_comparison.csv", "960/run.json", "predictions/960.json", "predictions/960-ap.npz"):
        path = E2_RUN / name
        hashes[str(path.relative_to(ROOT))] = sha256(path)
    earlier = {row["object_id"]: row["missed_all"] == "True" for row in read_csv(E2_RUN / "per_object_comparison.csv")}
    if earlier.keys() != {o["object_id"] for rows in truth.values() for o in rows} or sum(earlier.values()) != 77:
        raise ValueError("Unexpected E2 missed-object identities.")
    manifest = read_csv(MANIFESTS / "gc10-val.csv")
    for row in manifest:
        path = DATASET / row["image"]
        if (int(row["width"]), int(row["height"])) != (WIDTH, HEIGHT) or sha256(path) != row["image_sha256"]:
            raise ValueError(f"Validation image differs: {path}")
    return (checkpoint, prior, truth, {"A": guesses[640], "B": guesses[960], "D": e2_guesses[960]},
            {"A": native[640], "B": native[960], "D": e2_native[960]},
            {c: e2["metrics"][c] for c in ("A", "B", "D")}, hashes, earlier)


def analyze_boundaries(run, conf, objects):
    crops = rectangles(conf, True)
    for obj in objects:
        obj.update(boundary(obj, crops))
    sizes = ("all", "Q4")
    groups = ("contained_either", "partial_both", "crosses_boundary", "intersects_overlap", "within_overlap")
    rows = []
    for group in groups:
        for size in sizes:
            subset = [obj for obj in objects if obj[group] and (size == "all" or obj["area_640_group"] == size)]
            for c in CONDITIONS:
                hits = sum(obj[f"{c}_category"] == "correct" for obj in subset)
                rows.append({"group": group, "area": size, "condition": c, "objects": len(subset),
                             "images": len({obj["image"] for obj in subset}), "recall": ratio(hits, len(subset)),
                             "localization": sum(obj[f"{c}_category"] == "localization" for obj in subset)})
    save_csv(run / "crop_boundary_analysis.csv", rows)
    return rows


def duplicate_counts(run, truth, before, after):
    examples = []
    counts = {"premerge_predictions_at_floor": sum(map(len, before.values())),
              "postmerge_predictions_at_floor": sum(map(len, after.values())),
              "premerge_duplicates_at_fixed_point": 0, "postmerge_duplicates_at_fixed_point": 0}
    for label, guesses in (("premerge", before), ("postmerge", after)):
        for image, rows in truth.items():
            _, detections = match(rows, guesses[image], *MATCH_SETTINGS)
            duplicates = [dict(d, image=image) for d in detections if d["category"] == "duplicate"]
            counts[f"{label}_duplicates_at_fixed_point"] += len(duplicates)
            if label == "premerge":
                examples.extend(duplicates)
    write_json(run / "predictions" / "premerge_duplicates.json", examples)
    return counts, examples


def boundary_figures(run, conf, rows, objects, duplicates, before, after):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    groups = [r for r in rows if r["condition"] == "A" and r["area"] == "all"]
    labels = [f"{r['group'].replace('_', ' ')} (n={r['objects']})" for r in groups]
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    x = np.arange(len(labels))
    for i, c in enumerate(CONDITIONS):
        values = [r["recall"] for r in rows if r["condition"] == c and r["area"] == "all"]
        ax.bar(x + .19*(i-1.5), [np.nan if value is None else value for value in values], .19, label=c)
    ax.set(ylim=(0, 1), ylabel="Fixed-point recall")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.legend(frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(run / "plots" / f"crop_boundary_recall.{ext}", dpi=250)
    plt.close(fig)

    rng = random.Random(conf["SEED"])
    clipped = [o for o in objects if o["crosses_boundary"] and o["T_category"] != "correct"]
    for label, candidates in (("clipped", clipped), ("duplicate", duplicates)):
        for i, obj in enumerate(rng.sample(candidates, min(3, len(candidates))), 1):
            image = obj["image"]
            with Image.open(DATASET / image) as source:
                pixels = source.convert("RGB")
            fig, axes = plt.subplots(2, 1, figsize=(9, 7), layout="constrained")
            for ax, records, title in zip(axes, (before, after), ("Before merge", "After merge")):
                ax.imshow(pixels)
                for _, x1, _, x2, _ in rectangles(conf, True):
                    for edge in (x1, x2):
                        if 0 < edge < WIDTH:
                            ax.axvline(edge, ls="--", color="white")
                for pred in records[image]:
                    if pred["score"] >= conf["CONFIDENCE"]:
                        x, y, x2, y2 = box(pred)
                        color = "#ffb34a" if pred["crop"] == "left" else "#42cef5"
                        ax.add_patch(Rectangle((x, y), x2-x, y2-y, fill=False, ec=color))
                x, y, x2, y2 = box(obj)
                ax.add_patch(Rectangle((x, y), x2-x, y2-y, fill=False, ec="#55efbf", lw=2))
                ax.set_title(title)
                ax.set_axis_off()
            fig.suptitle(f"{label.title()} example · {Path(image).name}")
            for ext in ("png", "pdf"):
                fig.savefig(run / "examples" / f"{label}-{i:02}.{ext}", dpi=250)
            plt.close(fig)


def report(run, metrics, intervals, rows, counts, oracle, rescued, timing, classes):
    lines = ["# E3 fixed tiled inference · pending scientific review", "",
             "| Condition | mAP50 | mAP50–95 | Native P / R | Fixed P / R | TP / FP / FN |",
             "|---|---:|---:|---:|---:|---:|"]
    for c, r in metrics.items():
        lines.append(f"| {c} | {r['mAP50']:.2%} | {r['mAP50-95']:.2%} | "
                     f"{r['native_precision']:.2%} / {r['native_recall']:.2%} | "
                     f"{r['fixed_precision']:.2%} / {r['fixed_recall']:.2%} | {r['tp']} / {r['fp']} / {r['fn']} |")
    lines.extend(["", "A is B1 full-image 640, B is B1 full-image 960, D is E2 full-image 960, and T is B1 tiled 640.",
                  "", "| Pair | Metric | Change (points) | Paired 95% interval (points) |",
                  "|---|---|---:|---:|"])
    for r in intervals:
        interval = f"{100*r['ci_2.5']:+.2f} to {100*r['ci_97.5']:+.2f}" if r["ci_2.5"] is not None else "Undefined"
        lines.append(f"| {r['comparison']} | {r['metric']} | {100*r['delta']:+.2f} | {interval} |")
    lines.extend(["", "The intervals describe image sampling with fixed checkpoints. Training seed variability is unmeasured. "
                  "T–A is primary. T–D also differs in checkpoint and physical training batch.", "",
                  f"T recovered **{oracle['T_only']}** objects relative to A and lost **{oracle['A_only']}**. "
                  f"T recovered **{rescued}** of the 77 objects missed by every E2 condition. The retrospective A–T "
                  f"union recall is **{oracle['recall']:.2%}**.", "",
                  f"Fixed-point duplicates changed from {counts['premerge_duplicates_at_fixed_point']} before merge to "
                  f"{counts['postmerge_duplicates_at_fixed_point']} after merge.", "",
                  "| T crop group | Area stratum | Images | Objects | Recall | Localization errors |",
                  "|---|---|---:|---:|---:|---:|"])
    for r in rows:
        if r["condition"] == "T":
            recall = f"{r['recall']:.2%}" if r["recall"] is not None else "Undefined"
            lines.append(f"| {r['group']} | {r['area']} | {r['images']} | {r['objects']} | {recall} | {r['localization']} |")
    lines.extend(["", "Groups overlap. The Q4 rows describe whether large-defect misses coincide with limited crop coverage. "
                  "Compare denominators and per-object coverage before attributing errors to clipping.", "", "## Priority classes", ""])
    for name in ("silk_spot", "inclusion", "crease", "oil_spot", "rolled_pit"):
        subset = [r for r in classes if r["class"] == name]
        lines.append(f"- {name.replace('_', ' ').title()} ({subset[0]['objects']} objects). Fixed recall "
                     + ", ".join(f"{r['condition']} {r['fixed_recall']:.1%}" for r in subset) + ".")
    lines.extend(["", "Rolled pit has four objects in two images. Treat results as descriptive.", "",
                  "| Timing condition | Pass | View batch | Batch-1 median / p95 (ms) | Original images/s | Crop images/s |",
                  "|---|---:|---:|---:|---:|---:|"])
    for r in timing:
        latency = f"{r['median_batch1_ms']:.2f} / {r['p95_batch1_ms']:.2f}" if r["view_batch"] == 1 else "—"
        crops = f"{r['crop_images_per_second']:.1f}" if r["crop_images_per_second"] is not None else "—"
        lines.append(f"| {r['condition']} | {r['pass']} | {r['view_batch']} | {latency} | "
                     f"{r['original_images_per_second']:.1f} | {crops} |")
    (run / "report.md").write_text("\n".join(lines) + "\n")


def analyze(run, conf, prior, truth, predictions, records, expected, earlier, before, timing):
    ap = native_ap(combined(records["T"]))
    expected["T"] = dict(zip(("mAP50", "mAP50-95"), ap))
    metrics, classes, errors, sizes, curves, objects = diagnostics(
        run, conf, truth, predictions, records, expected, conditions=CONDITIONS)
    for c in ("A", "B", "D"):
        if any(metrics[c][key] != expected[c][key] for key in ("tp", "fp", "fn")):
            raise ValueError(f"{c} fixed-point counts differ from the reviewed E2 comparison.")
    boundaries = analyze_boundaries(run, conf, objects)
    for obj in objects:
        obj["missed_by_all_E2_conditions"] = earlier[obj["object_id"]]
    save_csv(run / "per_object_comparison.csv", objects)
    outcomes = {name: sum(obj["T_vs_A"] == name for obj in objects)
                for name in ("retained", "lost", "recovered", "both_missed")}
    oracle = {"shared": outcomes["retained"], "A_only": outcomes["lost"],
              "T_only": outcomes["recovered"], "both_missed": outcomes["both_missed"],
              "objects": len(objects), "recall": 1 - outcomes["both_missed"]/len(objects)}
    write_json(run / "oracle_analysis.json", oracle)
    counts, duplicates = duplicate_counts(run, truth, before, predictions["T"])
    intervals = paired_bootstrap(run, conf, truth, objects, records, metrics,
                                 conditions=CONDITIONS, comparisons=COMPARISONS)
    figures(run, conf, metrics, classes, errors, sizes, curves, objects, predictions,
            conditions=CONDITIONS, example_conditions=("T",))
    boundary_figures(run, conf, boundaries, objects, duplicates, before, predictions["T"])
    rescued = sum(obj["missed_by_all_E2_conditions"] and obj["T_category"] == "correct" for obj in objects)
    report(run, metrics, intervals, boundaries, counts, oracle, rescued, timing, classes)
    partial = {}
    for c in CONDITIONS:
        rows = [o for o in objects if o[f"{c}_gt_coverage"] is not None]
        partial[c] = {"objects": len(rows), "partial": sum((o[f"{c}_containment"] or 0) >= .8
                                                           and o[f"{c}_area_ratio"] < .5 for o in rows)}
        for key in ("gt_coverage", "area_ratio", "containment"):
            values = [o[f"{c}_{key}"] for o in rows if o[f"{c}_{key}"] is not None]
            partial[c][f"median_{key}"] = float(np.median(values)) if values else None
    write_json(run / "partial_detection.json", partial)
    write_json(run / "summary.json", {
        "status": "completed", "split": "val", "images": len(truth), "objects": len(objects),
        "checkpoint_sha256": E3_HASH, "dataset_manifest_sha256": prior["dataset_manifest_sha256"],
        "upstream_commit": UPSTREAM_COMMIT, "pbs_job_id": os.getenv("PBS_JOBID"),
        "metrics": metrics, "reused_AP_parity": True, "bootstrap": intervals, "oracle": oracle,
        "timing": timing, "environment": read_json(run / "environment.json"), "duplicates": counts,
        "partial_detection": partial, "recovered_from_all_prior_misses": rescued,
        "rolled_pit": [o for o in objects if o["class"] == "rolled_pit"],
    })


def execute(run, conf, checkpoint, prior, truth, predictions, records, expected, earlier):
    model, torch, ultralytics, device = load_model(checkpoint, conf["DEVICE"])
    from ultralytics.nn.autobackend import AutoBackend
    import subprocess

    props = torch.cuda.get_device_properties(device)
    uuid = "GPU-" + str(props.uuid).removeprefix("GPU-")
    write_json(run / "environment.json", {
        "device": str(device), "gpu": props.name, "gpu_uuid": uuid, "gpu_bytes": props.total_memory,
        "python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "numpy": np.__version__, "opencv": cv2.__version__,
        "ultralytics": ultralytics.__version__,
        "nvidia_driver": subprocess.check_output(
            ["nvidia-smi", "-i", uuid, "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip(),
    })
    names = list(truth)
    before, after, manifest, timing = {}, {}, [], []
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=conf["NUM_WORKERS"]) as pool, torch.inference_mode():
        backend = AutoBackend(weights=model.model, device=device, fp16=conf["FP16"]).eval()
        group = conf["BATCH_SIZE"] // 2
        for start in range(0, len(names), group):
            raw, merged, crops, _, _ = infer_group(names[start:start+group], True, conf["BATCH_SIZE"],
                                                   conf, backend, torch, device, pool)
            before.update(raw)
            after.update(merged)
            manifest.extend(crops)
            print(f"Tiled validation {min(start+group, len(names))}/{len(names)} originals", flush=True)
        save_csv(run / "crop_manifest.csv", manifest)
        write_json(run / "predictions" / "T-premerge.json", before)
        write_json(run / "predictions" / "T.json", after)
        for pass_id, tiled in enumerate((False, True, True, False), 1):
            for batch in (1, conf["BATCH_SIZE"]):
                print(f"Timing pass {pass_id}, {'T' if tiled else 'A'}, view batch {batch}", flush=True)
                torch.cuda.empty_cache()
                timing.append(measure(names, conf, backend, torch, device, pool, tiled, batch, pass_id))
    save_csv(run / "timing_comparison.csv", timing)
    predictions["T"] = after
    records["T"] = fused_records(after, truth, list(truth))
    analyze(run, conf, prior, truth, predictions, records, expected, earlier, before, timing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/tiling.yaml")
    parser.add_argument("--preflight", action="store_true", help="Verify inputs without CUDA inference.")
    args = parser.parse_args()
    conf = yaml.safe_load(args.config.read_text())
    fixed = {"SPLIT": "val", "IMAGE_SIZE": 640, "CROP_WIDTH": 1365, "CROP_HEIGHT": 1000,
             "CROP_OVERLAP": 682, "BATCH_SIZE": 16, "NUM_WORKERS": 8, "FP16": False,
             "NMS_CONFIDENCE_FLOOR": .001, "NMS_IOU": .7, "MAX_DETECTIONS": 300,
             "CONFIDENCE": .25, "MATCH_IOU": .50, "BACKGROUND_IOU": .10,
             "BOOTSTRAP_REPETITIONS": 5000, "SEED": 42}
    if any(conf.get(key) != value for key, value in fixed.items()):
        parser.error("E3 requires the frozen crop geometry and evaluation protocol.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", conf["RUN_ID"]) or not re.fullmatch(r"cuda:(0|[1-9][0-9]*)", conf["DEVICE"]):
        parser.error("Use an unused run ID and an authorized CUDA index.")
    run = ROOT / "experiments" / conf["RUN_ID"]
    if run.exists() or run.is_symlink():
        parser.error(f"Run directory already exists: {run}")
    conf["FP_PER_IMAGE_BUDGETS"] = list(BUDGETS)
    checkpoint, prior, truth, predictions, records, expected, hashes, earlier = load_inputs(conf)
    if args.preflight:
        print(f"Verified frozen B1 checkpoint and A/B/D records, 230 original images and 345 unique D1 objects. Fresh path: {run}")
        return
    source = read_json(ROOT / ".cache/source.json")
    files = ("scripts/evaluate_tiling.py", "scripts/evaluate_e2.py", "scripts/evaluate_resolution.py",
             "scripts/evaluate.py", "scripts/check_env.py", "scripts/evaluate_fusion.py", "scripts/analyze_errors.py",
             "submit_tiling.sh", "configs/evaluation/age_yolo_e2.yaml", str(args.config.resolve().relative_to(ROOT)))
    if source["upstream_commit"] != UPSTREAM_COMMIT or any(sha256(ROOT / name) != source["sha256"].get(name) for name in files):
        raise ValueError("Deployment source differs. Run scripts/record_source.py before submitting.")
    run.mkdir(exist_ok=False)
    for directory in ("predictions", "plots", "examples"):
        (run / directory).mkdir()
    write_json(run / "source.json", source)
    write_json(run / "reused_inputs_sha256.json", hashes)
    (run / "effective_config.yaml").write_text(yaml.safe_dump({
        **conf, "CROPS": [list(c) for c in rectangles(conf, True)], "TIMING_ORDER": ["A", "T", "T", "A"],
        "TIMING_WARMUP_BATCHES": 5, "TIMING_BATCH1_IMAGES": 40, "AGNOSTIC_NMS": False,
        "TEST_TIME_AUGMENTATION": False, "RECT": True,
    }, sort_keys=False))
    with (run / "terminal.log").open("x", buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            with TemporaryDirectory(prefix="settings-", dir=run) as settings:
                os.environ["YOLO_CONFIG_DIR"] = settings
                try:
                    execute(run, conf, checkpoint, prior, truth, predictions, records, expected, earlier)
                except Exception:
                    traceback.print_exc()
                    raise SystemExit(1) from None
            print(f"E3 artifacts: {run}", flush=True)


if __name__ == "__main__":
    main()
