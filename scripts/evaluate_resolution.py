"""Compare B1 at 640 and 960 on the frozen D1 validation objects."""

import argparse
import json
import os
import random
import re
import sys
import time
import traceback
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import yaml
from PIL import Image

from analyze_errors import CATEGORIES, box, box_geometry, match, ratio, read_csv, save_csv
from evaluate import (
    CONFIG, DATASET, MANIFESTS, ROOT, UPSTREAM_COMMIT, Tee, evaluate, load_model,
    prepare_split, sha256, verify_inputs, write_json,
)

SIZES = (640, 960)
PRIOR = ROOT / "experiments/d1-baseline-01"
CHECKPOINT = ROOT / "experiments/hpc-full-06/weights/best.pt"
FOCUS = ("inclusion", "silk_spot", "oil_spot", "crease", "rolled_pit")
MATCH_SETTINGS = (0.25, 0.50, 0.10)
METRICS = ("mAP50", "mAP50-95", "precision", "recall")
GEOMETRY = ("area_640", "short_side_640", "aspect_ratio")


def read_json(path):
    return json.loads(path.read_text())


def frozen_inputs():
    prior = read_json(PRIOR / "summary.json")
    if not prior["ap_parity_passed"] or prior["upstream_commit"] != UPSTREAM_COMMIT:
        raise ValueError("Expected accepted D1 artifacts from the pinned fork.")
    if verify_inputs(CHECKPOINT) != prior["checkpoint_sha256"]["B1"]:
        raise ValueError("B1 checkpoint differs from D1.")
    for split in ("train", "val"):
        if sha256(MANIFESTS / f"gc10-{split}.csv") != prior["dataset_manifest_sha256"][split]:
            raise ValueError(f"{split} manifest differs from D1.")
    manifest = read_csv(MANIFESTS / "gc10-val.csv")
    truth = {row["image"]: [] for row in manifest}
    bounds = prior["quartiles"]
    if bounds["short_side_640"] != [21.25, 38.75, 86.875]:
        raise ValueError("Unexpected D1 short-side boundaries.")
    ids = set()
    for row in read_csv(PRIOR / "per_object.csv"):
        if row["object_id"] in ids:
            raise ValueError("Repeated D1 object identity.")
        ids.add(row["object_id"])
        obj = {k: row[k] for k in ("image", "object_id", "class")}
        obj["class_id"] = int(row["class_id"])
        for key in (*GEOMETRY, "x1", "y1", "x2", "y2"):
            obj[key] = float(row[key])
        for key in GEOMETRY:
            group = int(np.searchsorted(bounds[key], obj[key], side="left")) + 1
            if row[f"{key}_group"] != f"Q{group}":
                raise ValueError(f"Unexpected D1 group for {obj['object_id']}.")
            obj[f"{key}_group"] = f"Q{group}"
        truth[obj["image"]].append(obj)
    if len(truth) != 230 or len(ids) != 345 or any(len(truth[r["image"]]) != int(r["objects"]) for r in manifest):
        raise ValueError("Unexpected D1 validation membership.")
    for row in manifest:
        label = DATASET / row["label"]
        if sha256(label) != row["label_sha256"]:
            raise ValueError(f"Validation label hash differs: {label}")
        lines = [line for line in label.read_text().splitlines() if line.strip()]
        for index, (line, obj) in enumerate(zip(lines, truth[row["image"]], strict=True)):
            cls, cx, cy, width, height = map(float, line.split())
            expected = box_geometry(cx, cy, width, height, int(row["width"]), int(row["height"]))
            if (obj["object_id"] != f"{row['image']}:{index}" or cls != obj["class_id"]
                    or any(obj[key] != expected[key] for key in (*GEOMETRY, "x1", "y1", "x2", "y2"))):
                raise ValueError(f"D1 object geometry differs from the verified label: {obj['object_id']}")
    return prior, truth


def export_validation(run, conf, size, truth):
    destination = run / str(size)
    destination.mkdir()
    model, torch, ultralytics, device = load_model(CHECKPOINT, conf["DEVICE"])
    from ultralytics.models.yolo.detect import DetectionValidator

    predictions, ap_records, shapes = {}, {}, set()

    class ExportValidator(DetectionValidator):
        def update_metrics(self, preds, batch):
            shapes.add(tuple(batch["img"].shape))
            for i, pred in enumerate(preds):
                image = f"images/val/{Path(batch['im_file'][i]).name}"
                if image in predictions or image not in truth:
                    raise ValueError(f"Unexpected validation image: {image}")
                prepared = self._prepare_batch(i, batch)
                native = self._prepare_pred(pred, prepared)
                target_classes = prepared["cls"]
                if Counter(target_classes.cpu().tolist()) != Counter(o["class_id"] for o in truth[image]):
                    raise ValueError(f"Native labels differ from D1 for {image}.")
                # The fork computes TP flags per image. Resampling can reuse them unchanged.
                tp = (self._process_batch(native, prepared["bbox"], target_classes)
                      if len(native) and len(target_classes)
                      else torch.zeros((len(native), self.niou), dtype=torch.bool, device=self.device))
                rows = native.cpu().numpy()
                predictions[image] = [
                    dict(zip(("x1", "y1", "x2", "y2", "score"), map(float, row[:5])), class_id=int(row[5]))
                    for row in rows
                ]
                ap_records[image] = (tp.cpu().numpy(), rows[:, 4], rows[:, 5], target_classes.cpu().numpy())
            super().update_metrics(preds, batch)

    print(f"E1 native validation at {size}px", flush=True)
    evaluate(model, torch, ultralytics, device, CHECKPOINT, sha256(CHECKPOINT),
             "val", destination, validator=ExportValidator, image_size=size)
    if predictions.keys() != truth.keys():
        raise ValueError("Native prediction images differ from the validation manifest.")
    write_json(run / "predictions" / f"{size}.json", predictions)
    np.savez_compressed(run / "predictions" / f"{size}-ap.npz", images=np.array(list(ap_records)),
                        prediction_counts=[len(record[1]) for record in ap_records.values()],
                        target_counts=[len(record[3]) for record in ap_records.values()],
                        **dict(zip(("tp", "conf", "pred_cls", "target_cls"),
                                   (np.concatenate(column) for column in zip(*ap_records.values())))))
    write_json(destination / "input_shapes.json", sorted(shapes))
    return predictions, ap_records


def score_curve(detections, total, images):
    # Correct matches at the floor are prefix-invariant under D1's score order.
    rows = sorted(detections, key=lambda row: -row["score"])
    curve = [{"threshold": None, "fp_per_image": 0.0, "recall": 0.0}]
    tp = fp = 0
    for index, row in enumerate(rows):
        correct = row["category"] == "correct"
        tp += correct
        fp += not correct
        if index + 1 == len(rows) or row["score"] != rows[index + 1]["score"]:
            curve.append({"threshold": row["score"], "fp_per_image": fp / images, "recall": tp / total})
    return curve


def diagnostics(run, conf, truth, predictions):
    objects, counts, curves, errors = [], {}, {}, []
    for image, targets in truth.items():
        for target in targets:
            objects.append(dict(target))
    by_id = {o["object_id"]: o for o in objects}
    per_class = {}
    for size in SIZES:
        all_detections, score_detections = [], []
        for image, targets in truth.items():
            matched, detections = match(targets, predictions[size][image], *MATCH_SETTINGS)
            all_detections.extend(dict(d, image=image) for d in detections)
            _, floor = match(targets, predictions[size][image], 0.001, *MATCH_SETTINGS[1:])
            score_detections.extend(floor)
            for obj, result in zip(targets, matched):
                by_id[obj["object_id"]].update({f"{key}_{size}": value for key, value in result.items()})
        frequencies = Counter(d["category"] for d in all_detections)
        frequencies["missed"] = sum(o[f"category_{size}"] == "missed" for o in objects)
        errors.extend({"resolution": size, "category": c, "count": frequencies[c]} for c in CATEGORIES)
        tp = frequencies["correct"]
        counts[size] = {"tp": tp, "fp": len(all_detections) - tp, "fn": len(objects) - tp,
                        "fixed_precision": ratio(tp, len(all_detections)), "fixed_recall": tp / len(objects)}
        curves[size] = score_curve(score_detections, len(objects), len(truth))
        per_class[size] = {int(r["class_id"]): r for r in read_csv(run / str(size) / "per_class.csv")}
        write_json(run / "predictions" / f"{size}-fixed-point.json", all_detections)

    outcomes = {(True, True): "both_detected", (False, True): "recovered",
                (True, False): "lost", (False, False): "both_missed"}
    for o in objects:
        o["outcome"] = outcomes[tuple(o[f"category_{size}"] == "correct" for size in SIZES)]
    groups = []
    for field in GEOMETRY:
        for quartile in ("Q1", "Q2", "Q3", "Q4"):
            subset = [o for o in objects if o[f"{field}_group"] == quartile]
            row = {"measure": field, "group": quartile, "objects": len(subset),
                   "images": len({o["image"] for o in subset})}
            for size in SIZES:
                row[f"recall_{size}"] = ratio(sum(o[f"category_{size}"] == "correct" for o in subset), len(subset))
                row[f"localization_{size}"] = sum(o[f"category_{size}"] == "localization" for o in subset)
            row["delta_recall"] = row["recall_960"] - row["recall_640"] if subset else None
            row["delta_localization"] = row["localization_960"] - row["localization_640"]
            groups.append(row)
    classes = []
    for cls, info in sorted(per_class[640].items()):
        label = info["class_name"]
        subset = [o for o in objects if o["class_id"] == cls]
        row = {"class_id": cls, "class": label, "objects": len(subset), "images": len({o["image"] for o in subset})}
        for size in SIZES:
            row[f"recall_{size}"] = ratio(sum(o[f"category_{size}"] == "correct" for o in subset), len(subset))
            for metric in ("AP50", "AP50-95"):
                value = per_class[size][cls][metric]
                row[f"{metric}_{size}"] = float(value) if value else None
        for metric in ("AP50", "AP50-95", "recall"):
            a, b = row[f"{metric}_640"], row[f"{metric}_960"]
            row[f"delta_{metric}"] = b - a if a is not None and b is not None else None
        classes.append(row)
    budgets = []
    for budget in conf["FP_PER_IMAGE_BUDGETS"]:
        row = {"fp_per_image_budget": budget}
        for size in SIZES:
            best = max((p for p in curves[size] if p["fp_per_image"] <= budget),
                       key=lambda p: (p["recall"], -p["fp_per_image"]))
            row.update({f"{k}_{size}": v for k, v in best.items()})
            row[f"budget_within_sweep_{size}"] = budget <= curves[size][-1]["fp_per_image"]
        row["delta_recall"] = row["recall_960"] - row["recall_640"]
        budgets.append(row)
    for name, rows in (("per_class_comparison", classes), ("per_object_comparison", objects),
                       ("size_analysis", groups), ("error_breakdown", errors), ("recall_fp_comparison", budgets)):
        save_csv(run / f"{name}.csv", rows)
    write_json(run / "predictions" / "recall_fp_curves.json", curves)
    return objects, counts, curves, errors, groups, classes


def native_ap(arrays):
    from ultralytics.utils.metrics import ap_per_class

    ap = ap_per_class(*arrays, plot=False)[5]
    return np.array([ap[:, 0].mean(), ap.mean()])


def bootstrap(run, conf, truth, objects, records, metrics):
    images = list(truth)
    indices = {image: i for i, image in enumerate(images)}
    per_image_counts = np.zeros((len(images), 6), dtype=int)
    for row in objects:
        image = indices[row["image"]]
        q1 = row["short_side_640_group"] == "Q1"
        hit640, hit960 = (row[f"category_{size}"] == "correct" for size in SIZES)
        per_image_counts[image] += [1, q1, hit640, hit960, q1 and hit640, q1 and hit960]

    native_order = {}
    for size in SIZES:
        # Keep each validator's image order, including its ordering of tied scores.
        native_order[size] = [(indices[image], record) for image, record in records[size].items()]
        combined = [np.concatenate(col) for col in zip(*records[size].values())]
        expected = [metrics[size]["mAP50"], metrics[size]["mAP50-95"]]
        if not np.allclose(native_ap(combined), expected, rtol=0, atol=1e-8):
            raise ValueError(f"{size}px exported AP records do not reproduce native AP.")

    rng = np.random.default_rng(conf["SEED"])
    names = ("mAP50", "mAP50-95", "fixed_recall", "q1_fixed_recall")
    repetitions = conf["BOOTSTRAP_REPETITIONS"]
    deltas = np.empty((repetitions, len(names)))
    for repetition in range(repetitions):
        sampled = rng.integers(len(images), size=len(images))
        multiplicity = np.bincount(sampled, minlength=len(images))
        aps = []
        for size in SIZES:
            selected = [record for index, record in native_order[size] for _ in range(multiplicity[index])]
            aps.append(native_ap([np.concatenate(col) for col in zip(*selected)]))
        n, q1, tp640, tp960, q640, q960 = multiplicity @ per_image_counts
        deltas[repetition] = [*(aps[1] - aps[0]), (tp960 - tp640) / n,
                              (q960 - q640) / q1 if q1 else np.nan]
        if (repetition + 1) % 500 == 0:
            print(f"Paired bootstrap {repetition + 1}/{repetitions}", flush=True)
    rows = []
    for name, values in zip(names, deltas.T):
        valid = values[np.isfinite(values)]
        low, high = np.quantile(valid, [0.025, 0.975]) if len(valid) else (None, None)
        if name == "q1_fixed_recall":
            _, total, _, _, q640, q960 = per_image_counts.sum(axis=0)
            observed = (q960 - q640) / total if total else None
        else:
            observed = metrics[960][name] - metrics[640][name]
        rows.append({"metric": name, "delta_960_minus_640": observed, "ci_2.5": low,
                     "ci_97.5": high, "valid_repetitions": len(valid), "seed": conf["SEED"]})
    save_csv(run / "bootstrap_results.csv", rows)
    np.save(run / "predictions" / "bootstrap_deltas.npy", deltas)
    return rows


def measure_timing(run, conf):
    model, torch, _, device = load_model(CHECKPOINT, conf["DEVICE"])
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.nn.autobackend import AutoBackend

    def clock():
        torch.cuda.synchronize(device)
        return time.perf_counter()

    def process(validator, batch):
        begin = clock()
        batch = validator.preprocess(batch)
        preprocessed = clock()
        raw = backend(batch["img"], augment=False)
        inferred = clock()
        validator.postprocess(raw)
        finished = clock()
        return [preprocessed - begin, inferred - preprocessed, finished - inferred], tuple(batch["img"].shape)

    properties = torch.cuda.get_device_properties(device)
    write_json(run / "timing_device.json", {"logical_device": str(device), "gpu": properties.name,
                                             "uuid": str(properties.uuid), "total_bytes": properties.total_memory})
    results = []
    with TemporaryDirectory(prefix="timing-inputs-", dir=run) as temp, torch.inference_mode():
        staging = Path(temp)
        prepare_split("val", MANIFESTS / "gc10-val.csv", staging)
        image_path = str(staging / "images/val")
        data = {**yaml.safe_load(CONFIG.read_text()), "val": image_path, "channels": 3}
        backend = AutoBackend(weights=model.model, device=device, fp16=False).eval()
        torch.cuda.empty_cache()
        sample_names = random.Random(conf["SEED"]).sample(
            sorted(Path(image_path).glob("*")), conf["LATENCY_IMAGES"] + conf["WARMUP_BATCHES"])
        for pass_index, size in enumerate((640, 960, 960, 640), 1):
            for batch_size in (1, 16):
                print(f"Timing pass {pass_index}, {size}px, batch {batch_size}", flush=True)
                validator = DetectionValidator(
                    save_dir=run, args={"imgsz": size, "batch": batch_size, "workers": 8, "half": False,
                                        "rect": True, "conf": 0.001, "iou": 0.7, "max_det": 300,
                                        "agnostic_nms": False, "augment": False, "split": "val", "plots": False})
                validator.device, validator.stride, validator.data = device, backend.stride, data
                if batch_size == 1:
                    dataset = validator.build_dataset(image_path, batch=1)
                    by_name = {Path(path).name: i for i, path in enumerate(dataset.im_files)}
                    samples = [by_name[path.name] for path in sample_names]

                    def sample_batch(index):
                        return dataset.collate_fn([dataset[index]])

                    for index in samples[:conf["WARMUP_BATCHES"]]:
                        process(validator, sample_batch(index))
                    measured = samples[conf["WARMUP_BATCHES"]:]
                    get_batch = sample_batch
                else:
                    loader = validator.get_dataloader(image_path, batch_size)
                    batches = iter(loader)
                    # Finish the epoch to return the infinite loader to its original order.
                    for index, batch in enumerate(batches):
                        if index < conf["WARMUP_BATCHES"]:
                            process(validator, batch)
                    batches = iter(loader)
                    measured = range(len(loader))

                    def get_batch(index):
                        return next(batches)

                clock()
                torch.cuda.reset_peak_memory_stats(device)
                accumulated, elapsed, shapes = [], [], set()
                for index in measured:
                    begin = clock()
                    batch = get_batch(index)
                    phases, shape = process(validator, batch)
                    elapsed.append(clock() - begin)
                    accumulated.append(phases)
                    shapes.add(shape)
                    del batch
                samples_count = len(measured) if batch_size == 1 else len(loader.dataset)
                phases = np.asarray(accumulated)
                results.append({
                    "resolution": size, "pass": pass_index, "batch": batch_size, "images": samples_count,
                    "preprocess_ms_per_image": 1000 * phases[:, 0].sum() / samples_count,
                    "inference_ms_per_image": 1000 * phases[:, 1].sum() / samples_count,
                    "postprocess_ms_per_image": 1000 * phases[:, 2].sum() / samples_count,
                    "end_to_end_ms_per_image": 1000 * sum(elapsed) / samples_count,
                    "images_per_second": samples_count / sum(elapsed),
                    "batch1_median_ms": 1000 * np.median(elapsed) if batch_size == 1 else None,
                    "batch1_p95_ms": 1000 * np.quantile(elapsed, .95) if batch_size == 1 else None,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                    "input_shapes_nchw": json.dumps(sorted(shapes)),
                })
                if batch_size == 16:
                    del batches, loader
                else:
                    del dataset
                torch.cuda.empty_cache()
    save_csv(run / "timing_comparison.csv", results)
    return results


def plot_results(run, conf, metrics, classes, groups, errors, curves, timing, objects, predictions):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {640: "#2166ac", 960: "#b35806"}

    def save(fig, name, directory="plots"):
        for ext in ("png", "pdf"):
            fig.savefig(run / directory / f"{name}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    def bars(name, labels, values, ylabel, ylim=None):
        fig, ax = plt.subplots(figsize=(max(6, len(labels)), 4))
        x = np.arange(len(labels))
        for offset, size in zip((-.18, .18), SIZES):
            ax.bar(x + offset, [np.nan if v is None else v for v in values[size]], .36,
                   color=colors[size], label=f"{size} px")
        ax.set_xticks(x, labels, rotation=28, ha="right")
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(frameon=False)
        save(fig, name)

    bars("overall_accuracy", (*METRICS, "Fixed recall"), {
        s: [metrics[s][key] for key in (*METRICS, "fixed_recall")] for s in SIZES}, "Metric (fraction)", (0, 1))
    for key, filename in (("AP50", "class_ap50"), ("AP50-95", "class_ap50_95")):
        bars(filename, [r["class"].replace("_", " ") for r in classes],
             {s: [r[f"{key}_{s}"] for r in classes] for s in SIZES}, key, (0, 1))
    for field in GEOMETRY:
        rows = [r for r in groups if r["measure"] == field]
        bars(f"recall_{field}", [f"{r['group']} (n={r['objects']})" for r in rows],
             {s: [r[f"recall_{s}"] for r in rows] for s in SIZES}, "Fixed-point recall", (0, 1))
    bars("detection_errors", [c.replace("_", " ") for c in CATEGORIES],
         {s: [r["count"] for r in errors if r["resolution"] == s] for s in SIZES}, "Assignments")

    fig, ax = plt.subplots(figsize=(6, 4))
    for size in SIZES:
        points = curves[size]
        ax.step([p["fp_per_image"] for p in points], [p["recall"] for p in points], where="post",
                color=colors[size], label=f"{size} px")
    ax.set(xlabel="False positives per image", ylabel="Recall", xlim=(0, max(conf["FP_PER_IMAGE_BUDGETS"])), ylim=(0, 1))
    ax.legend(frameon=False)
    save(fig, "recall_fp")

    fig, ax = plt.subplots(figsize=(6, 4))
    for size in SIZES:
        repeats = [r for r in timing if r["resolution"] == size and r["batch"] == 1]
        ax.scatter([r["batch1_median_ms"] for r in repeats], [metrics[size]["mAP50-95"]] * len(repeats),
                   color=colors[size], label=f"{size} px")
    ax.set(xlabel="Median batch-1 end-to-end latency per pass (ms)", ylabel="Native mAP50–95")
    ax.legend(frameon=False)
    save(fig, "accuracy_latency")

    rng = random.Random(conf["SEED"])
    for outcome in ("recovered", "lost"):
        candidates = sorted((obj for obj in objects if obj["outcome"] == outcome), key=lambda obj: obj["object_id"])
        for index, obj in enumerate(rng.sample(candidates, min(len(candidates), 3)), 1):
            image = obj["image"]
            with Image.open(DATASET / image) as original:
                pixels = original.convert("RGB")
            fig, axes = plt.subplots(2, 1, figsize=(10, 8), layout="constrained")
            for ax, size in zip(axes, SIZES):
                ax.imshow(pixels)
                for pred in predictions[size][image]:
                    if pred["score"] >= MATCH_SETTINGS[0]:
                        x1, y1, x2, y2 = box(pred)
                        ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor="#ffb34a"))
                x1, y1, x2, y2 = box(obj)
                ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor="#55efbf", linewidth=2))
                ax.set_title(f"{size} px · {obj[f'category_{size}'].replace('_', ' ')}")
                ax.set_axis_off()
            fig.suptitle(f"{outcome.title()} {obj['class'].replace('_', ' ')} · {Path(image).name}")
            save(fig, f"{outcome}-{index:02}", "examples")


def report(run, metrics, intervals, objects, classes, groups, timing):
    lines = ["# E1 · B1 resolution comparison on validation", "",
             "| Resolution | mAP50 | mAP50–95 | Native precision | Native recall | Fixed-point TP / FP / FN |",
             "|---|---:|---:|---:|---:|---:|"]
    for size, row in metrics.items():
        lines.append(f"| {size} | {row['mAP50']:.2%} | {row['mAP50-95']:.2%} | {row['precision']:.2%} | "
                     f"{row['recall']:.2%} | {row['tp']} / {row['fp']} / {row['fn']} |")
    lines.extend(["", "## Paired accuracy changes", "",
                  "| Metric | 960 minus 640 (percentage points) | 95% interval (points) |",
                  "|---|---:|---:|"])
    for row in intervals:
        interval = f"{100 * row['ci_2.5']:+.2f} to {100 * row['ci_97.5']:+.2f}" if row["ci_2.5"] is not None else "Undefined"
        lines.append(f"| {row['metric']} | {100 * row['delta_960_minus_640']:+.2f} | {interval} |")
    outcomes = Counter(o["outcome"] for o in objects)
    lines.extend(["", f"At the D1 operating point, 960 recovered **{outcomes['recovered']}** objects and lost "
                  f"**{outcomes['lost']}**. Both resolutions missed **{outcomes['both_missed']}**.", "",
                  "| Short-side group | Objects | Recall at 640 | Recall at 960 | Localization error change |",
                  "|---|---:|---:|---:|---:|"])
    for row in groups:
        if row["measure"] == "short_side_640":
            lines.append(f"| {row['group']} | {row['objects']} | {row['recall_640']:.2%} | {row['recall_960']:.2%} | "
                         f"{row['delta_localization']:+d} |")
    lines.extend(["", "## Priority classes", ""])
    for row in classes:
        if row["class"] in FOCUS:
            lines.append(f"- {row['class'].replace('_', ' ').title()} · {row['objects']} objects across {row['images']} "
                         f"images. Fixed recall {row['recall_640']:.2%} → {row['recall_960']:.2%}.")
    lines.extend(["", "Rolled pit has four objects in two images. Interpret its comparison descriptively.", "",
                  "## Performance", "",
                  "| Resolution | Pass | Batch-1 median (ms) | Batch-1 p95 (ms) | Batch-16 images/s |",
                  "|---|---:|---:|---:|---:|"])
    for single, batch in zip(timing[::2], timing[1::2]):
        lines.append(f"| {single['resolution']} | {single['pass']} | {single['batch1_median_ms']:.2f} | "
                     f"{single['batch1_p95_ms']:.2f} | {batch['images_per_second']:.2f} |")
    lines.extend(["", "The CSV records per-stage latency, peak CUDA memory and actual input shapes. "
                  "The recall–FP table includes realized FP/image at each shared budget. "
                  "Results apply to this checkpoint and validation split. The intervals resample images and do not "
                  "measure variation across training seeds."])
    (run / "report.md").write_text("\n".join(lines) + "\n")


def run_experiment(run, conf, prior, truth):
    predictions, records, metrics = {}, {}, {}
    for size in SIZES:
        predictions[size], records[size] = export_validation(run, conf, size, truth)
        native = read_json(run / str(size) / "metrics.json")
        metrics[size] = {key: native[f"metrics/{key}(B)"] for key in METRICS}
        if size == 640:
            prior_metrics = prior["metrics"]["B1"]["native"]
            control_delta = {key: metrics[size][key] - prior_metrics[f"metrics/{key}(B)"] for key in METRICS[:2]}
            if any(abs(delta) > conf["AP_PARITY_TOLERANCE"] for delta in control_delta.values()):
                raise ValueError(f"640px control failed D1 AP parity: {control_delta}")
            print(f"640px control passed D1 AP parity: {control_delta}", flush=True)
    objects, counts, curves, errors, groups, classes = diagnostics(run, conf, truth, predictions)
    for size in SIZES:
        metrics[size].update(counts[size])
    save_csv(run / "metrics_comparison.csv", [{"resolution": size, **metrics[size]} for size in SIZES])
    timing = measure_timing(run, conf)
    intervals = bootstrap(run, conf, truth, objects, records, metrics)
    plot_results(run, conf, metrics, classes, groups, errors, curves, timing, objects, predictions)
    report(run, metrics, intervals, objects, classes, groups, timing)
    write_json(run / "summary.json", {
        "status": "completed", "split": "val", "images": len(truth), "objects": len(objects),
        "checkpoint_sha256": sha256(CHECKPOINT), "dataset_manifest_sha256": prior["dataset_manifest_sha256"],
        "d1_summary_sha256": sha256(PRIOR / "summary.json"), "d1_objects_sha256": sha256(PRIOR / "per_object.csv"),
        "upstream_commit": UPSTREAM_COMMIT, "device": conf["DEVICE"], "pbs_job_id": os.getenv("PBS_JOBID"),
        "quartiles": prior["quartiles"], "control_ap_delta_from_d1": control_delta, "control_ap_parity_passed": True,
        "metrics": metrics, "object_outcomes": dict(Counter(o["outcome"] for o in objects)),
        "bootstrap": intervals, "rolled_pit": next(r for r in classes if r["class"] == "rolled_pit"),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/resolution.yaml")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    conf = yaml.safe_load(args.config.read_text())
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", conf["RUN_ID"]):
        parser.error("RUN_ID must be a single directory name.")
    if not re.fullmatch(r"cuda:(0|[1-9][0-9]*)", conf["DEVICE"]):
        parser.error("DEVICE must be an indexed CUDA device authorized for the job.")
    if conf["BOOTSTRAP_REPETITIONS"] != 5000 or conf["SEED"] != 42:
        parser.error("E1 requires 5000 paired repetitions and seed 42.")
    if not (0 < conf["WARMUP_BATCHES"] < 15 and 0 < conf["LATENCY_IMAGES"] <= 230 - conf["WARMUP_BATCHES"]):
        parser.error("Timing counts exceed the available validation batches or images.")
    if not conf["FP_PER_IMAGE_BUDGETS"] or any(not np.isfinite(v) or v <= 0 for v in conf["FP_PER_IMAGE_BUDGETS"]):
        parser.error("FP/image budgets must be finite and positive.")
    run = ROOT / "experiments" / conf["RUN_ID"]
    if run.exists() or run.is_symlink():
        parser.error(f"Choose an unused run ID. Already exists: {run}")
    prior, truth = frozen_inputs()
    recorded = yaml.safe_load((PRIOR / "effective_config.yaml").read_text())
    if (recorded["CONFIDENCE"], recorded["MATCH_IOU"], recorded["BACKGROUND_IOU"]) != MATCH_SETTINGS:
        raise ValueError("D1 matching settings differ from E1.")
    protocol = {"SPLIT": "val", "BATCH": 16, "WORKERS": 8, "RECT": True, "FP16": False,
                "NMS_CONFIDENCE_FLOOR": 0.001, "NMS_IOU": 0.7, "MAX_DETECTIONS": 300,
                "AGNOSTIC_NMS": False, "TEST_TIME_AUGMENTATION": False}
    if any(recorded[key] != value for key, value in protocol.items()):
        raise ValueError("D1 evaluator settings differ from the frozen E1 protocol.")
    conf.update(protocol, AP_PARITY_TOLERANCE=recorded["AP_PARITY_TOLERANCE"],
                CONFIDENCE=MATCH_SETTINGS[0], MATCH_IOU=MATCH_SETTINGS[1], BACKGROUND_IOU=MATCH_SETTINGS[2])
    conf.update(RESOLUTIONS=list(SIZES), CHECKPOINT=str(CHECKPOINT.relative_to(ROOT)), TIMING_ORDER=[640, 960, 960, 640])
    if args.preflight:
        print(f"Verified frozen B1 checkpoint, D1 manifests and 345 objects on 230 images. Fresh run path: {run}")
        return
    source = read_json(ROOT / ".cache/source.json")
    source_files = ("scripts/evaluate_resolution.py", "scripts/evaluate.py", "scripts/analyze_errors.py",
                    "submit_resolution.sh", str(args.config.resolve().relative_to(ROOT)))
    if source["upstream_commit"] != UPSTREAM_COMMIT or any(
        sha256(ROOT / name) != source["sha256"].get(name) for name in source_files
    ):
        raise ValueError("Deployment changed. Run scripts/record_source.py before submitting.")
    run.mkdir(exist_ok=False)
    for directory in ("plots", "examples", "predictions"):
        (run / directory).mkdir()
    write_json(run / "source.json", source)
    (run / "effective_config.yaml").write_text(yaml.safe_dump(conf, sort_keys=False))
    with (run / "terminal.log").open("x", buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            with TemporaryDirectory(prefix="settings-", dir=run) as settings:
                os.environ["YOLO_CONFIG_DIR"] = settings
                try:
                    run_experiment(run, conf, prior, truth)
                except Exception:
                    traceback.print_exc()
                    raise SystemExit(1) from None
            print(f"E1 artifacts: {run}")


if __name__ == "__main__":
    main()
