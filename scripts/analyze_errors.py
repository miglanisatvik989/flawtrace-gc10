"""Analyze B0/B1 detections on the manifest-verified GC10-DET validation split."""

import argparse
import csv
import json
import os
import random
import re
import sys
import traceback
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import yaml
from PIL import Image

from evaluate import (
    CONFIG, DATASET, IMAGE_SIZE, MANIFESTS, ROOT, UPSTREAM_COMMIT, Tee,
    evaluate, load_model, sha256, verify_inputs, write_json,
)

MODELS = ("B0", "B1")
CATEGORIES = (
    "correct", "classification", "localization", "classification_and_localization",
    "duplicate", "background", "overlap_with_accounted_object", "missed",
)
GEOMETRY = {
    "area_640": "Bounding-box area at 640 (px²)",
    "short_side_640": "Bounding-box short side at 640 (px)",
    "aspect_ratio": "Bounding-box aspect ratio (long / short)",
    "contrast": "Local grayscale contrast (0–255)",
}


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def box_geometry(cx, cy, width, height, image_width, image_height):
    width *= image_width
    height *= image_height
    cx *= image_width
    cy *= image_height
    scale = IMAGE_SIZE / max(image_width, image_height)
    return {
        "x1": cx - width / 2, "y1": cy - height / 2,
        "x2": cx + width / 2, "y2": cy + height / 2,
        "width": width, "height": height, "width_640": width * scale,
        "height_640": height * scale, "area": width * height,
        "area_640": width * height * scale**2, "relative_area": width * height / (image_width * image_height),
        "short_side": min(width, height), "short_side_640": min(width, height) * scale,
        "aspect_ratio": max(width, height) / min(width, height),
    }


def box(row):
    return [row[key] for key in ("x1", "y1", "x2", "y2")]


def local_contrast(gray, coordinates):
    x1, y1, x2, y2 = coordinates
    height, width = gray.shape
    left, top = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
    right, bottom = min(width, int(np.ceil(x2))), min(height, int(np.ceil(y2)))
    pad = max(2, round(min(right - left, bottom - top) / 4))
    outer_left, outer_top = max(0, left - pad), max(0, top - pad)
    ring = gray[outer_top:bottom + pad, outer_left:right + pad].astype(float)
    ring[top - outer_top:bottom - outer_top, left - outer_left:right - outer_left] = np.nan
    inside = gray[top:bottom, left:right]
    if not inside.size or not np.isfinite(ring).any():
        return None
    return abs(float(np.median(inside)) - float(np.nanmedian(ring)))


def read_objects(split):
    objects = {}
    for record in read_csv(MANIFESTS / f"gc10-{split}.csv"):
        image, label = record["image"], DATASET / record["label"]
        # The native evaluator independently verifies validation image bytes on staging.
        if sha256(label) != record["label_sha256"]:
            raise ValueError(f"Annotation checksum differs: {label}")
        rows = []
        if split == "val":
            with Image.open(DATASET / image) as original:
                if original.size != (int(record["width"]), int(record["height"])):
                    raise ValueError(f"Image dimensions differ: {image}")
                gray = np.asarray(original.convert("L"))
        for index, line in enumerate(filter(str.strip, label.read_text().splitlines())):
            cls, cx, cy, width, height = map(float, line.split())
            geometry = box_geometry(cx, cy, width, height, int(record["width"]), int(record["height"]))
            row = {"image": image, "object_id": f"{image}:{index}", "class_id": int(cls), **geometry}
            if split == "val":
                row["contrast"] = local_contrast(gray, box(row))
            rows.append(row)
        if len(rows) != int(record["objects"]):
            raise ValueError(f"Object count differs: {label}")
        objects[image] = rows
    return objects


def box_iou(first, second):
    first = np.asarray(first, dtype=float).reshape(-1, 4)
    second = np.asarray(second, dtype=float).reshape(-1, 4)
    sizes = np.minimum(first[:, None, 2:], second[:, 2:]) - np.maximum(first[:, None, :2], second[:, :2])
    intersection = np.maximum(sizes, 0).prod(axis=2)
    union = (first[:, 2:] - first[:, :2]).prod(axis=1)[:, None] + (
        second[:, 2:] - second[:, :2]).prod(axis=1) - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def match(objects, predictions, confidence, foreground, background):
    # The native score sweep supplies AP. These assignments describe one fixed operating point.
    predictions = sorted((p for p in predictions if p["score"] >= confidence),
                         key=lambda p: -p["score"])
    overlaps = box_iou([box(p) for p in predictions], [box(o) for o in objects])
    targets = [{"category": "missed", "iou": None, "score": None} for _ in objects]
    detections = [dict(p, category=None, object_id=None, iou=None) for p in predictions]

    def assign(pred_index, obj_index, category):
        detection = detections[pred_index]
        iou = float(overlaps[pred_index, obj_index])
        detection.update(category=category, object_id=objects[obj_index]["object_id"], iou=iou)
        targets[obj_index] = {"category": category, "iou": iou, "score": detection["score"]}

    for i, p in enumerate(detections):
        eligible = [j for j, o in enumerate(objects) if o["class_id"] == p["class_id"] and overlaps[i, j] >= foreground]
        free = [j for j in eligible if targets[j]["category"] == "missed"]
        if free:
            assign(i, max(free, key=lambda j: overlaps[i, j]), "correct")
        elif eligible:
            j = max(eligible, key=lambda j: overlaps[i, j])
            p.update(category="duplicate", object_id=objects[j]["object_id"], iou=float(overlaps[i, j]))

    for i, p in enumerate(detections):
        if p["category"] is not None:
            continue
        eligible = [j for j in range(len(objects)) if overlaps[i, j] >= background]
        free = [j for j in eligible if targets[j]["category"] == "missed"]
        if not free:
            p["category"] = "overlap_with_accounted_object" if eligible else "background"
            p["iou"] = float(max(overlaps[i], default=0))
            continue
        j = max(free, key=lambda j: overlaps[i, j])
        if overlaps[i, j] >= foreground:
            category = "classification"
        elif p["class_id"] == objects[j]["class_id"]:
            category = "localization"
        else:
            category = "classification_and_localization"
        assign(i, j, category)
    return targets, detections


def export_predictions(name, path, conf, run):
    destination = run / name
    destination.mkdir()
    predictions = {}
    model, torch, ultralytics, device = load_model(path, conf["DEVICE"])
    from ultralytics.models.yolo.detect import DetectionValidator

    class ExportValidator(DetectionValidator):
        """Record the native post-NMS detections in original pixel coordinates."""

        def update_metrics(self, preds, batch):
            for index, pred in enumerate(preds):
                image = f"images/val/{Path(batch['im_file'][index]).name}"
                if image in predictions:
                    raise ValueError(f"Repeated validation image: {image}")
                native = self._prepare_pred(pred, self._prepare_batch(index, batch))
                predictions[image] = [
                    dict(zip(("x1", "y1", "x2", "y2", "score"), row[:5]), class_id=int(row[5]))
                    for row in native.cpu().tolist()
                ]
            super().update_metrics(preds, batch)

    evaluate(model, torch, ultralytics, device, path, verify_inputs(path, name == "B0"),
             "val", destination, validator=ExportValidator)
    write_json(run / "predictions" / f"{name}.json", predictions)
    return predictions


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def grouped(objects, field, bounds, model, training):
    labels = ("Q1", "Q2", "Q3", "Q4")
    rows = []
    for index, label in enumerate(labels):
        group = [o for o in objects if o[field] is not None and np.searchsorted(bounds, o[field], side="left") == index]
        total = len(group)
        tp = sum(o[f"{model}_category"] == "correct" for o in group)
        rows.append({
            "model": model, "measure": field, "group": label,
            "lower_exclusive": bounds[index - 1] if index else None,
            "upper_inclusive": bounds[index] if index < 3 else None,
            "training_objects": sum(np.searchsorted(bounds, o[field], side="left") == index for o in training),
            "images": len({o["image"] for o in group}), "objects": total,
            "tp": tp, "fn": total - tp, "recall": ratio(tp, total),
            "localization_only": sum(o[f"{model}_category"] == "localization" for o in group),
            "median_correct_iou": median([o[f"{model}_iou"] for o in group
                                          if o[f"{model}_category"] == "correct"]),
        })
    return rows


def plot_results(run, objects, class_rows, groups, errors, examples, conf, names, training):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = ("#2166ac", "#b35806")

    def save(fig, filename):
        for extension in ("pdf", "png"):
            fig.savefig(run / "plots" / f"{filename}.{extension}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    def compare(rows, key, labels, ylabel, filename, limit=None):
        fig, ax = plt.subplots(figsize=(9, 4))
        x = np.arange(len(labels))
        for index, model in enumerate(MODELS):
            values = [r[key] if r[key] is not None else np.nan for r in rows if r["model"] == model]
            ax.bar(x + (index - 0.5) * 0.36, values, width=0.36, label=model, color=colors[index])
        ax.set_xticks(x, labels, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        if limit:
            ax.set_ylim(0, limit)
        ax.legend(frameon=False)
        save(fig, filename)

    class_labels = [name.replace("_", " ") for name in names.values()]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for ax, rows, label in zip(axes, (training, objects), ("Original training", "Validation")):
        ax.bar(class_labels, [sum(o["class_id"] == cls for o in rows) for cls in names], color=colors[0])
        ax.tick_params(axis="x", labelrotation=60)
        ax.set_ylabel("Objects")
        ax.set_title(label)
    save(fig, "class_distribution")
    for field in ("area_640", "short_side_640", "aspect_ratio"):
        fig, ax = plt.subplots(figsize=(6, 4))
        values = [o[field] for o in objects]
        edges = np.geomspace(min(values), max(values), 26)
        ax.hist(values, bins=edges, color=colors[0])
        ax.set_xscale("log")
        ax.set(xlabel=GEOMETRY[field], ylabel="Validation objects")
        save(fig, field)
    short_side = [r for r in groups if r["measure"] == "short_side_640"]
    labels = [f"{r['group']} (n={r['objects']})" for r in short_side if r["model"] == "B0"]
    compare(short_side, "recall", labels, "Recall at fixed operating point", "short_side_recall", 1)
    compare(class_rows, "AP50", class_labels, "AP50", "per_class_AP50", 1)
    compare(class_rows, "AP50-95", class_labels, "AP50–95", "per_class_AP50_95", 1)
    compare(errors, "count", [s.replace("_", " ") for s in CATEGORIES], "Assignments", "error_distribution")

    rng = random.Random(conf["EXAMPLE_SEED"])
    for category in CATEGORIES:
        choices = sorted(image for image, (targets, detections) in examples.items()
                         if any(row["category"] == category for row in targets + detections))
        if not choices:
            continue
        image = rng.choice(choices)
        targets, detections = examples[image]
        fig, axes = plt.subplots(2, 1, figsize=(11, 9), layout="constrained")
        with Image.open(DATASET / image) as original:
            pixels = original.convert("RGB")
        for ax, rows, title in zip(axes, (targets, detections), ("Ground truth", "B1 detections")):
            ax.imshow(pixels)
            for row in rows:
                x1, y1, x2, y2 = box(row)
                color = "#48ecb2" if row["category"] == "correct" else "#ffb34a"
                ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=1.5))
                ax.text(x1, y1, names[row["class_id"]].replace("_", " "), color=color, fontsize=8,
                        bbox={"facecolor": "#202020", "edgecolor": "none", "alpha": 0.8, "pad": 1})
            ax.set_title(title)
            ax.set_axis_off()
        fig.suptitle(f"{category.replace('_', ' ').title()} example · {Path(image).name}")
        fig.savefig(run / "examples" / f"{category}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def analyze(run, conf, checkpoints, truth, training, names):
    objects = [o for rows in truth.values() for o in rows]
    bounds = {field: np.quantile([o[field] for o in training], [0.25, 0.5, 0.75]).tolist()
              for field in ("area_640", "short_side_640", "aspect_ratio")}
    train_class = Counter(o["class_id"] for o in training)
    for o in objects:
        o["class"] = names[o["class_id"]]
        o["training_class_objects"] = train_class[o["class_id"]]
        for field, cuts in bounds.items():
            o[f"{field}_group"] = f"Q{1 + np.searchsorted(cuts, o[field], side='left')}"
    image_rows, class_rows, groups, errors, score_deltas = [], [], [], [], {}
    metrics = {}
    examples = {}
    for model in MODELS:
        predictions = export_predictions(model, checkpoints[model], conf, run)
        if predictions.keys() != truth.keys():
            raise ValueError(f"{model} prediction and validation manifest membership differ.")
        detections = []
        for image, targets in truth.items():
            matches, guesses = match(targets, predictions[image], conf["CONFIDENCE"],
                                     conf["MATCH_IOU"], conf["BACKGROUND_IOU"])
            for o, result in zip(targets, matches):
                o.update({f"{model}_{key}": value for key, value in result.items()})
            tp = sum(row["category"] == "correct" for row in matches)
            image_rows.append({"model": model, "image": image, "objects": len(targets),
                               "tp": tp, "fp": len(guesses) - tp, "fn": len(targets) - tp})
            detections.extend(dict(p, image=image) for p in guesses)
            if model == "B1":
                examples[image] = ([dict(o, category=r["category"]) for o, r in zip(targets, matches)], guesses)
        ap = {int(row["class_id"]): row for row in read_csv(run / model / "per_class.csv")}
        for cls, label in names.items():
            targets = [o for o in objects if o["class_id"] == cls]
            tp = sum(o[f"{model}_category"] == "correct" for o in targets)
            fp = sum(p["class_id"] == cls and p["category"] != "correct" for p in detections)
            class_rows.append({"model": model, "class": label, "training_objects": train_class[cls],
                               "images": len({o["image"] for o in targets}), "objects": len(targets),
                               "tp": tp, "fp": fp, "fn": len(targets) - tp, "precision": ratio(tp, tp + fp),
                               "recall": ratio(tp, len(targets)), **{key: float(ap[cls][key]) if ap[cls][key] else None
                                                                  for key in ("AP50", "AP50-95")}})
        counts = Counter(p["category"] for p in detections)
        counts["missed"] = sum(o[f"{model}_category"] == "missed" for o in objects)
        errors.extend({"model": model, "category": category, "count": counts[category]}
                      for category in CATEGORIES)
        groups.extend(row for field, cuts in bounds.items() for row in grouped(objects, field, cuts, model, training))
        output = json.loads((run / model / "metrics.json").read_text())
        baseline = (
            {"metrics/mAP50(B)": 0.7627876396046485, "metrics/mAP50-95(B)": 0.42070577742096765}
            if model == "B0" else json.loads((ROOT / "experiments/b1-val-02/metrics.json").read_text())
        )
        score_deltas[model] = {key: output[key] - baseline[key] for key in
                               ("metrics/mAP50(B)", "metrics/mAP50-95(B)")}
        tp = counts["correct"]
        metrics[model] = {"native": output, "tp": tp, "fp": len(detections) - tp,
                          "fn": len(objects) - tp, "precision": ratio(tp, len(detections)),
                          "recall": ratio(tp, len(objects))}
        write_json(run / "predictions" / f"{model}-fixed-point.json", detections)

    comparison = []
    for b0, b1, name in ((True, True, "both_detected"), (False, False, "both_missed"),
                         (True, False, "B0_only"), (False, True, "B1_only")):
        rows = [o for o in objects if (o["B0_category"] == "correct", o["B1_category"] == "correct") == (b0, b1)]
        for o in rows:
            o["comparison"] = name
        comparison.append({
            "group": name, "images": len({o["image"] for o in rows}), "objects": len(rows),
            **{f"median_{field}": median([o[field] for o in rows if o[field] is not None])
               for field in (*GEOMETRY, "training_class_objects")},
            "contrast_objects": sum(o["contrast"] is not None for o in rows),
        })

    tables = {"per_class": class_rows, "per_object": objects, "per_image": image_rows,
              "size_analysis": [r for r in groups if r["measure"] != "aspect_ratio"],
              "aspect_ratio_analysis": [r for r in groups if r["measure"] == "aspect_ratio"],
              "error_breakdown": errors, "checkpoint_comparison": comparison}
    for filename, rows in tables.items():
        save_csv(run / f"{filename}.csv", rows)
    plot_results(run, objects, class_rows, groups, errors, examples, conf, names, training)
    parity = all(abs(delta) <= conf["AP_PARITY_TOLERANCE"] for values in score_deltas.values() for delta in values.values())
    rolled_pit = [row for row in class_rows if row["class"] == "rolled_pit"]
    write_json(run / "summary.json", {
        "split": "val", "images": len(truth), "objects": len(objects), "classes": len(names),
        "training_images": 1834, "training_objects": len(training), "quartiles": bounds,
        "metrics": metrics, "comparison": comparison, "rolled_pit": rolled_pit,
        "prior_validation_ap_deltas": score_deltas, "ap_parity_passed": parity,
        "upstream_commit": UPSTREAM_COMMIT,
        "dataset_manifest_sha256": {split: sha256(MANIFESTS / f"gc10-{split}.csv") for split in ("train", "val")},
        "checkpoint_sha256": {name: sha256(path) for name, path in checkpoints.items()},
        "analysis_source_sha256": sha256(Path(__file__)), "evaluator_source_sha256": sha256(Path(__file__).with_name("evaluate.py")),
    })
    write_report(run, metrics, groups, comparison, rolled_pit, parity, class_rows, errors)


def median(values):
    return float(np.median(values)) if values else None


def write_report(run, metrics, groups, comparison, rolled_pit, parity, class_rows, errors):
    def percent(value):
        return f"{value:.1%}" if value is not None else "undefined"

    lines = ["# D1 validation error analysis", "",
             f"Native AP parity within configured tolerance for both checkpoints: **{parity}**.", "",
             "| Model | Native AP50 | Native AP50–95 | TP | FP | FN | Fixed precision | Fixed recall |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model, result in metrics.items():
        native = result["native"]
        lines.append(f"| {model} | {native['metrics/mAP50(B)']:.4f} | {native['metrics/mAP50-95(B)']:.4f} | "
                     f"{result['tp']} | {result['fp']} | {result['fn']} | {percent(result['precision'])} | "
                     f"{percent(result['recall'])} |")
    lines.extend(["", "## Short side at 640", "",
                  "| Model | Training-derived quartile | Images | Objects | Recall |",
                  "|---|---|---:|---:|---:|"])
    for row in groups:
        if row["measure"] == "short_side_640":
            lines.append(f"| {row['model']} | {row['group']} | {row['images']} | {row['objects']} | {percent(row['recall'])} |")
    for model in MODELS:
        short_side = [r for r in groups if r["model"] == model and r["measure"] == "short_side_640"]
        if all(r["recall"] is not None for r in (short_side[0], short_side[-1])):
            difference = 100 * (short_side[0]["recall"] - short_side[-1]["recall"])
            lines.append(f"\n{model} Q1 minus Q4 recall: {difference:+.1f} percentage points.")
    lines.extend(["", "## Paired object outcomes", "",
                  "| Outcome | Objects | Images | Median short side (px) | Median area (px²) | Median aspect | Median contrast | Median training class count |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in comparison:
        medians = [row[f"median_{field}"] for field in
                   ("short_side_640", "area_640", "aspect_ratio", "contrast", "training_class_objects")]
        display = " | ".join(f"{v:.2f}" if v is not None else "—" for v in medians)
        lines.append(f"| {row['group']} | {row['objects']} | {row['images']} | {display} |")
    lines.extend(["", "## Rolled-pit subset", "",
                  "Four validation objects across two images. Treat results as descriptive."])
    for row in rolled_pit:
        lines.append(f"- {row['model']} detected {row['tp']} of {row['objects']} at the fixed operating point.")
    lines.extend(["", "## Error mix and class frequency", ""])
    for model in MODELS:
        common = sorted((r for r in errors if r["model"] == model and r["category"] != "correct"),
                        key=lambda r: -r["count"])[:2]
        labels = ", ".join(f"{r['category'].replace('_', ' ')} ({r['count']})" for r in common)
        classes = [r for r in class_rows if r["model"] == model and r["recall"] is not None]
        frequencies = np.asarray([r["training_objects"] for r in classes])
        recalls = np.asarray([r["recall"] for r in classes])
        correlation = np.corrcoef(frequencies, recalls)[0, 1] if np.std(recalls) and np.std(frequencies) else None
        association = f"{correlation:.2f}" if correlation is not None else "undefined"
        lines.append(f"- {model}. Largest error categories: {labels}. Pearson correlation between "
                     f"training class count and validation class recall: {association} across {len(classes)} classes.")
    if not parity:
        lines.extend(["", "**Parity check failed. Review baseline settings and hardware differences before interpreting comparisons.**"])
    lines.extend(["", "## Interpretation and limitations", "",
                  "Compare short-side recall across quartiles and examine the paired group medians alongside class support. "
                  "Class, area, aspect ratio and contrast may covary. Multiple objects can share an image. "
                  "These descriptive associations cannot isolate a resizing effect.",
                  "", "Contrast uses the difference between median grayscale intensity inside each box and in a local ring. "
                  "It can miss textured defects and depends on annotation tightness and the surrounding material. "
                  "Native AP integrates over retained detection scores. Error counts use a single operating point; "
                  "a different confidence threshold can change the counts. This analysis uses validation only."])
    (run / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/error_analysis.yaml")
    parser.add_argument("--preflight", action="store_true", help="Check inputs without inference or writing artifacts.")
    args = parser.parse_args()
    conf = yaml.safe_load(args.config.read_text())
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", conf["RUN_ID"]):
        parser.error("RUN_ID must be a single directory name.")
    if not re.fullmatch(r"cuda:(0|[1-9][0-9]*)", conf["DEVICE"]):
        parser.error("DEVICE must be an indexed CUDA device authorized for the job.")
    if not 0.001 <= conf["CONFIDENCE"] <= 1 or not 0 < conf["BACKGROUND_IOU"] < conf["MATCH_IOU"] <= 1:
        parser.error("Invalid score or IoU thresholds.")
    if conf["AP_PARITY_TOLERANCE"] < 0:
        parser.error("AP_PARITY_TOLERANCE must be nonnegative.")
    checkpoints = {name: (ROOT / conf[f"{name}_CHECKPOINT"]).resolve() for name in MODELS}
    digests = {name: verify_inputs(path, name == "B0") for name, path in checkpoints.items()}
    prior = json.loads((ROOT / "experiments/b1-val-02/run.json").read_text())
    if prior["checkpoint"]["sha256"] != digests["B1"] or prior["dataset"]["split_manifest_sha256"] != sha256(MANIFESTS / "gc10-val.csv"):
        raise ValueError("B1 checkpoint or validation manifest differs from the accepted evaluation.")
    run = ROOT / "experiments" / conf["RUN_ID"]
    if run.exists() or run.is_symlink():
        parser.error(f"Choose an unused run ID. Already exists: {run}")
    names = dict(enumerate(yaml.safe_load(CONFIG.read_text())["names"]))
    truth = read_objects("val")
    training = [o for rows in read_objects("train").values() for o in rows]
    val_objects = [o for rows in truth.values() for o in rows]
    if (len(truth), len(val_objects), len(names)) != (230, 345, 10) or {o["class_id"] for o in val_objects} != set(names):
        raise ValueError("Unexpected validation split counts.")
    if args.preflight:
        print(f"Verified B0 and B1 checkpoints, 230 validation images, 345 objects, "
              f"10 classes and {len(training)} original training objects. Fresh run path: {run}")
        return
    run.mkdir(exist_ok=False)
    for directory in ("plots", "examples", "predictions"):
        (run / directory).mkdir()
    settings = {"SPLIT": "val", "IMAGE_SIZE": IMAGE_SIZE, "BATCH": 16, "WORKERS": 8, "RECT": True,
                "FP16": False, "NMS_CONFIDENCE_FLOOR": 0.001, "NMS_IOU": 0.7, "MAX_DETECTIONS": 300,
                "AGNOSTIC_NMS": False, "TEST_TIME_AUGMENTATION": False}
    (run / "effective_config.yaml").write_text(yaml.safe_dump({**conf, **settings}, sort_keys=False))
    with (run / "terminal.log").open("x", buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            with TemporaryDirectory(prefix="settings-", dir=run) as settings:
                os.environ["YOLO_CONFIG_DIR"] = settings
                try:
                    analyze(run, conf, checkpoints, truth, training, names)
                except Exception:
                    traceback.print_exc()
                    raise SystemExit(1) from None
            print(f"D1 artifacts: {run}")


if __name__ == "__main__":
    main()
