"""Evaluate the completed A0 and A1 checkpoints on the original validation split."""

import argparse
import hashlib
import os
import re
import sys
import traceback
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import yaml

from analyze_errors import CATEGORIES, box, box_iou, match, ratio, read_csv, save_csv
from evaluate import CONFIG, DATASET, MANIFESTS, ROOT, UPSTREAM, UPSTREAM_COMMIT, Tee, sha256, write_json
from evaluate_resolution import (
    FOCUS, GEOMETRY, PRIOR, export_validation, frozen_inputs, measure_timing, read_json, score_curve,
)
from record_source import architecture_eval_files
from train import initialize_cuda

MODELS = {
    "A0": ("a0-age-yolo-640-b32-02", "upstream/AGE-YOLO/ultralytics/cfg/models/age_yolo.yaml",
           "d81967a9ec44d7df0235daf67c1d08fe34b8900714c875bceacd250a7c6aaf88", (8, 16, 32), 3418721, 7.874,
           146, 196, 2.008, 7.94),
    "A1": ("a1-age-yolo-p2-640-b32-01", "configs/models/age_yolo_p2.yaml",
           "72533a364a7169df6ec7c47352416d30ab529d1607fc8ded79fa260c6bc18082", (4, 8, 16, 32), 3446915, 9.690,
           160, 210, 2.666, 14.0),
}
NATIVE_OPTIONS = {
    "SPLIT": ("split", "val"), "IMAGE_SIZE": ("imgsz", 640), "BATCH_SIZE": ("batch", 16),
    "NUM_WORKERS": ("workers", 8), "FP16": ("half", False), "RECT": ("rect", True),
    "NMS_CONFIDENCE_FLOOR": ("conf", .001), "NMS_IOU": ("iou", .7), "MAX_DETECTIONS": ("max_det", 300),
    "AGNOSTIC_NMS": ("agnostic_nms", False), "TEST_TIME_AUGMENTATION": ("augment", False),
}
DESIGN = {"CONFIDENCE": .25, "MATCH_IOU": .50, "BACKGROUND_IOU": .10, "BOOTSTRAP_REPETITIONS": 5000,
          "SEED": 42, "FP_PER_IMAGE_BUDGETS": [.10, .25, .50, 1.], "LATENCY_IMAGES": 40, "WARMUP_BATCHES": 5}
SHORT_BOUNDS = (16, 24, 32, 64)
AP_KEYS = ("mAP50", "mAP50-95", "nine_class_AP50", "nine_class_AP50-95")
BOOT_KEYS = (*AP_KEYS, "fixed_recall", "q1_recall", "short_le16_recall")


def load_config(path):
    conf = yaml.safe_load(path.read_text())
    fixed = {key: value for key, (_, value) in NATIVE_OPTIONS.items()} | DESIGN
    fixed.update({f"{model}_CHECKPOINT": f"experiments/{row[0]}/weights/best.pt" for model, row in MODELS.items()})
    if not isinstance(conf, dict) or set(conf) != set(fixed) | {"RUN_ID", "DEVICE"}:
        raise ValueError("Expected the flat A0/A1 evaluation settings.")
    for key, value in fixed.items():
        if type(conf[key]) is not type(value) or conf[key] != value:
            raise ValueError(f"{key} must equal the registered setting {value!r}.")
    for key, pattern in (("RUN_ID", r"[A-Za-z0-9][A-Za-z0-9_-]*"), ("DEVICE", r"cuda:(0|[1-9][0-9]*)")):
        if not isinstance(conf[key], str) or not re.fullmatch(pattern, conf[key]):
            raise ValueError(f"Invalid {key}.")
    return conf


def verify_source(config, conf):
    path = ROOT / ".cache" / f"source-{conf['RUN_ID']}.json"
    command = f"python scripts/record_source.py --architecture-eval --config {config}"
    if not path.is_file():
        raise ValueError(f"Missing source record. Run {command}")
    source = read_json(path)
    if source["upstream_commit"] != UPSTREAM_COMMIT:
        raise ValueError("Source record has the wrong AGE-YOLO pin.")
    for name in architecture_eval_files(config, conf):
        file = ROOT / name
        if not file.is_file() or sha256(file) != source["sha256"].get(name):
            raise ValueError(f"Source mismatch for {name}. Run {command}")
    return source


def verify_training(conf, prior):
    train_digest = sha256(ROOT / "data/gc10-det-aug42-x4/train.csv")
    provenance = {}
    for name, (run_id, model_file, digest, strides, params, flops, best, epochs, hours, peak) in MODELS.items():
        checkpoint = ROOT / conf[f"{name}_CHECKPOINT"]
        run = checkpoint.parents[1]
        source_path, = run.glob("source-*.json")
        attempt_path, = run.glob("attempt-*.json")
        source, attempt = read_json(source_path), read_json(attempt_path)
        effective = read_json(run / "effective_config.json")
        args = effective["args"]
        saved = yaml.safe_load((run / "hpc.yaml").read_text())
        archived_model = run / Path(model_file).name
        if (read_json(run / "run.json")["status"] != "completed" or attempt["status"] != "completed"
                or source["upstream_commit"] != UPSTREAM_COMMIT
                or sha256(checkpoint) != digest or attempt["checkpoint_sha256"]["best.pt"] != digest
                or sha256(run / "weights/last.pt") != attempt["checkpoint_sha256"]["last.pt"]
                or sha256(archived_model) != source["sha256"].get(model_file)
                or sha256(ROOT / model_file) != sha256(archived_model)):
            raise ValueError(f"{name} training, source or checkpoint identity differs.")
        archived_config = attempt["command"][-1]
        source_yaml = source["untracked_source"][archived_config]
        if (yaml.safe_load(source_yaml) != saved
                or hashlib.sha256(source_yaml.encode()).hexdigest() != source["sha256"][archived_config]
                or saved["MODEL_CONFIG"] != model_file or saved["RUN_ID"] != run_id
                or yaml.safe_load((run / "args.yaml").read_text()) != args
                or str(ROOT / model_file) != args["model"]):
            raise ValueError(f"{name} archived training arguments disagree.")
        if (attempt["train_manifest_sha256"] != train_digest
                or attempt["augmentation"]["source_manifest_sha256"]["val"] != prior["dataset_manifest_sha256"]["val"]
                or sha256(run / "gc10_augmented.yaml") != source["sha256"].get(saved["DATA_CONFIG"])):
            raise ValueError(f"{name} dataset provenance differs.")
        if (args["imgsz"], args["batch"], args["seed"], args["patience"], args["split"], args["val"],
                args["multi_scale"], args["pretrained"], args["resume"], args["deterministic"],
                effective["amp"], effective["gradient_accumulation"], effective["train_batch"],
                effective["val_batch"], effective["optimizer_weight_decay"]) != (
                    640, 32, 42, 50, "val", True, False, False, False, True, True, 2, 32, 64, [0, .0005, 0]):
            raise ValueError(f"{name} native trainer settings differ from the approved design.")
        approved = {"epochs": 300, "optimizer": "SGD", "lr0": .01, "momentum": .937, "weight_decay": .0005,
                    "workers": 8, "mosaic": 1., "fliplr": 0., "flipud": 0., "scale": .1,
                    "hsv_h": .01, "hsv_s": .2, "hsv_v": .2}
        if any(args[key] != value for key, value in approved.items()):
            raise ValueError(f"{name} native optimizer or augmentation settings differ.")
        results = read_csv(run / "results.csv")
        winning = max(results, key=lambda row: .1 * float(row["metrics/mAP50(B)"]) + .9 * float(row["metrics/mAP50-95(B)"]))
        log = (run / "terminal.log").read_text()
        gpu_peak = max(map(float, re.findall(r"\b\d+/300\s+([0-9.]+)G\b", log)))
        if ([int(row["epoch"]) for row in results] != list(range(1, epochs + 1)) or int(winning["epoch"]) != best
                or f"Best results observed at epoch {best}," not in log
                or f"{epochs} epochs completed in {hours:.3f} hours." not in log or gpu_peak != peak):
            raise ValueError(f"{name} training results or early stopping differ.")
        files = [source_path, attempt_path, archived_model, *(run / p for p in (
            "hpc.yaml", "args.yaml", "gc10_augmented.yaml", "effective_config.json", "environment.json",
            "results.csv", "terminal.log", "run.json"))]
        provenance[name] = {
            "run_id": run_id, "checkpoint_sha256": digest, "training_commit": source["commit"],
            "trainer_sha256": source["sha256"]["scripts/train.py"], "effective_config": effective,
            "saved_config": saved, "attempt": attempt, "architecture": str(archived_model),
            "architecture_sha256": sha256(archived_model), "strides": strides, "parameters": params,
            "estimated_gflops": flops, "best_epoch": best, "completed_epochs": epochs, "training_hours": hours,
            "training_peak_reserved_gib_displayed": peak, "best_epoch_csv": winning,
            "environment": read_json(run / "environment.json"),
            "evidence_sha256": {str(path.relative_to(run)): sha256(path) for path in files},
        }
    a, b = provenance.values()
    vary = {"model", "device", "name", "save_dir"}
    differences = {k for k in a["effective_config"]["args"] | b["effective_config"]["args"]
                   if a["effective_config"]["args"].get(k) != b["effective_config"]["args"].get(k)}
    if differences != vary or a["trainer_sha256"] != b["trainer_sha256"] or a["training_commit"] != b["training_commit"]:
        raise ValueError(f"Training provenance or effective methodology differs: {sorted(differences)}")
    return provenance


def verify_architectures(conf, provenance):
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    for name, info in provenance.items():
        model = YOLO(str(ROOT / conf[f"{name}_CHECKPOINT"])).model.float().eval()
        configuration = yaml.safe_load(Path(info["architecture"]).read_text())
        if (any(model.yaml[key] != configuration[key] for key in ("nc", "scales", "backbone", "head"))
                or tuple(model.stride.tolist()) != info["strides"]
                or sum(p.numel() for p in model.parameters()) != info["parameters"]):
            raise ValueError(f"{name} checkpoint architecture differs from its source-verified YAML.")
        gflops = get_flops(model, imgsz=640)
        if round(gflops, 3) != info["estimated_gflops"]:
            raise ValueError(f"{name} estimated GFLOPs differ: {gflops}.")
        with torch.inference_mode():
            decoded, raw = model(torch.zeros(1, 3, 352, 672))
        grids = [(352 // s, 672 // s) for s in info["strides"]]
        if ([tuple(feature.shape[-2:]) for feature in raw] != grids
                or decoded.shape != (1, 14, sum(h*w for h, w in grids))
                or not torch.isfinite(decoded).all()):
            raise ValueError(f"{name} failed the rectangular synthetic forward.")
        info["estimated_gflops"] = gflops
        print(f"{name}: {info['strides']}, {info['parameters']:,} parameters, {gflops:.3f} GFLOPs")


def check_validation_images():
    rows = read_csv(MANIFESTS / "gc10-val.csv")
    for field in ("image", "label"):
        expected = {row[field]: row[f"{field}_sha256"] for row in rows}
        folder = next(iter(expected))
        actual = {path.relative_to(DATASET).as_posix() for path in (DATASET / Path(folder).parent).iterdir()}
        if actual != expected.keys() or any(sha256(DATASET / path) != digest for path, digest in expected.items()):
            raise ValueError(f"Validation {field} bytes or membership differ from the audited manifest.")


def prediction_levels(decoded, grids, kept, prepared, truth):
    # Join native post-NMS input-space boxes to exact decoded boxes and class scores.
    # Reading a clone keeps native NMS, post-NMS order and metric updates untouched.
    from ultralytics.utils.ops import xywh2xyxy

    expected = np.array([box(obj) for obj in truth]).reshape(-1, 4)
    actual = prepared["bbox"].cpu().numpy()
    # Rounded rectangular resizing can introduce a few original pixels of discrepancy.
    if actual.shape != expected.shape or not np.allclose(actual, expected, atol=4, rtol=0):
        raise ValueError("Native validation boxes disagree with original D1 label coordinates.")
    if prepared["cls"].cpu().tolist() != [obj["class_id"] for obj in truth]:
        raise ValueError("Native validation label order differs from D1.")
    height, width = prepared["imgsz"]
    strides = [height // h for h, _ in grids]
    if strides not in ([8, 16, 32], [4, 8, 16, 32]) or any(
            height != s*h or width != s*w for s, (h, w) in zip(strides, grids)):
        raise ValueError("Could not recover the detection grids for native NMS attribution.")
    offsets = np.cumsum([h*w for h, w in grids])
    candidates = xywh2xyxy(decoded[:4].T).cpu().numpy()
    scores = decoded[4:].cpu().numpy()
    if len(candidates) != offsets[-1]:
        raise ValueError("Raw detection counts disagree with the decoded grid.")
    rows = kept.cpu().numpy()
    result = [{"level": "unattributed"} for _ in rows]
    for class_id in np.unique(rows[:, 5]).astype(int):
        order = np.argsort(scores[class_id])
        sorted_scores = scores[class_id, order]
        indices = np.flatnonzero(rows[:, 5] == class_id)
        lower = np.searchsorted(sorted_scores, rows[indices, 4], side="left")
        upper = np.searchsorted(sorted_scores, rows[indices, 4], side="right")
        for index, lo, hi in zip(indices, lower, upper):
            matching = order[lo:hi]
            matching = matching[(candidates[matching] == rows[index, :4]).all(axis=1)]
            levels = np.unique(np.searchsorted(offsets, matching, side="right"))
            if len(levels) == 1:
                result[index]["level"] = f"P{strides[levels[0]].bit_length()-1}"
            elif len(levels) > 1:
                result[index]["level"] = "ambiguous"
    return result


def hits(obj, name):
    return obj[f"{name}_category"] == "correct"


def summarize_group(objects):
    row = {"objects": len(objects), "images": len({o["image"] for o in objects}),
           "recovered": sum(o["outcome"] == "recovered" for o in objects),
           "lost": sum(o["outcome"] == "lost" for o in objects)}
    for name in MODELS:
        ious = [o[f"{name}_iou"] for o in objects if hits(o, name)]
        row.update({f"recall_{name}": ratio(len(ious), len(objects)),
                    f"localization_{name}": sum(o[f"{name}_category"] == "localization" for o in objects),
                    f"median_correct_iou_{name}": float(np.median(ious)) if ious else None})
    for key in ("recall", "localization", "median_correct_iou"):
        a, b = (row[f"{key}_{name}"] for name in MODELS)
        row[f"delta_{key}"] = b - a if a is not None and b is not None else None
    return row


def compare_false_positives(image, fixed, threshold):
    a0 = [row for row in fixed["A0"] if row["category"] != "correct"]
    a1 = [row for row in fixed["A1"] if row["category"] != "correct"]
    reference = [dict(row, object_id=str(i)) for i, row in enumerate(a0)]
    guesses = [dict(row, index=i) for i, row in enumerate(a1)]
    _, paired = match(reference, guesses, 0, threshold, threshold)
    counterparts = {row["index"]: int(row["object_id"]) for row in paired if row["category"] == "correct"}
    return [dict(row, image=image, change="shared" if i in counterparts else "added",
                 A0_counterpart_index=counterparts.get(i)) for i, row in enumerate(a1)] + [
                     dict(row, image=image, change="removed", A0_counterpart_index=i)
                     for i, row in enumerate(a0) if i not in counterparts.values()]


def paired_objects(run, conf, truth, predictions):
    objects, changes = [], []
    detections = {name: [] for name in MODELS}
    curves = {name: [] for name in MODELS}
    for image, targets in truth.items():
        matches, fixed = {}, {}
        for name in MODELS:
            guesses = predictions[name][image]
            matches[name], fixed[name] = match(targets, guesses, conf["CONFIDENCE"], conf["MATCH_IOU"], conf["BACKGROUND_IOU"])
            detections[name].extend(dict(row, image=image) for row in fixed[name])
            curves[name].extend(match(targets, guesses, conf["NMS_CONFIDENCE_FLOOR"], conf["MATCH_IOU"],
                                      conf["BACKGROUND_IOU"])[1])
        changes.extend(compare_false_positives(image, fixed, conf["MATCH_IOU"]))
        for target, a0, a1 in zip(targets, matches["A0"], matches["A1"], strict=True):
            row = dict(target)
            for name, result in (("A0", a0), ("A1", a1)):
                row.update({f"{name}_{key}": value for key, value in result.items()})
            pair = hits(row, "A0"), hits(row, "A1")
            row["outcome"] = {(True, True): "retained", (False, True): "recovered",
                              (True, False): "lost", (False, False): "both_missed"}[pair]
            row["delta_correct_iou"] = a1["iou"] - a0["iou"] if all(pair) else None
            row["poorer_correct_iou"] = all(pair) and row["delta_correct_iou"] < 0
            for name in MODELS:
                diagnosis = next((p for p in fixed[name] if p["object_id"] == target["object_id"]
                                  and p["category"] != "duplicate"), None)
                row[f"{name}_predicted_class"] = diagnosis["class_id"] if diagnosis else None
                row[f"{name}_matched_level"] = diagnosis.get("level") if diagnosis else None
                candidates = predictions[name][image] if target["class"] == "rolled_pit" else []
                overlaps = box_iou([box(target)], [box(p) for p in candidates])[0]
                for kind, eligible in (("any", range(len(candidates))),
                                       ("same_class", (i for i, p in enumerate(candidates)
                                                       if p["class_id"] == target["class_id"]))):
                    index = max(eligible, key=lambda i: (overlaps[i], candidates[i]["score"]), default=None)
                    found = index is not None and overlaps[index] >= conf["BACKGROUND_IOU"]
                    row.update({f"{name}_best_{kind}_class": candidates[index]["class_id"] if found else None,
                                f"{name}_best_{kind}_score": candidates[index]["score"] if found else None,
                                f"{name}_best_{kind}_iou": float(overlaps[index]) if found else None})
            objects.append(row)
    for name in MODELS:
        write_json(run / "predictions" / f"{name}-fixed-point.json", detections[name])
        curves[name] = score_curve(curves[name], len(objects), len(truth))
    write_json(run / "predictions" / "false_positive_changes.json", changes)
    write_json(run / "predictions" / "recall_fp_curves.json", curves)
    return objects, detections, curves, changes


def per_class_ap(records):
    from ultralytics.utils.metrics import ap_per_class

    arrays = [np.concatenate(column) for column in zip(*records)]
    _, _, precision, recall, _, ap, ids, *_ = ap_per_class(*arrays, plot=False)
    return ap, ids, precision, recall


def mean_ap(ap, ids, requested):
    present = np.isin(ids, requested)
    if present.sum() != len(requested):
        return np.full(2, np.nan)
    return np.array([ap[present, 0].mean(), ap[present].mean()])


def analyze(run, conf, objects, detections, curves, records):
    names = yaml.safe_load(CONFIG.read_text())["names"]
    ids = list(range(len(names)))
    nine = [i for i in ids if names[i] != "rolled_pit"]
    metrics, classes, errors, sizes, small, budgets = {}, [], [], [], [], []
    for name in MODELS:
        ap, class_ids, p, r = per_class_ap(records[name].values())
        metrics[name] = dict(zip(AP_KEYS, [*mean_ap(ap, class_ids, ids), *mean_ap(ap, class_ids, nine)]))
        native = read_json(run / name / "metrics.json")
        parity = {"mAP50": metrics[name]["mAP50"], "mAP50-95": metrics[name]["mAP50-95"],
                  "precision": p.mean(), "recall": r.mean()}
        if not all(np.isclose(v, native[f"metrics/{key}(B)"], atol=1e-8, rtol=0) for key, v in parity.items()):
            raise ValueError(f"{name} prediction export failed native metric parity.")
        counts = Counter(d["category"] for d in detections[name])
        counts["missed"] = sum(o[f"{name}_category"] == "missed" for o in objects)
        tp = counts["correct"]
        metrics[name].update(native_precision=float(p.mean()), native_recall=float(r.mean()),
                             fixed_precision=ratio(tp, len(detections[name])), fixed_recall=tp/len(objects),
                             tp=tp, fp=len(detections[name])-tp, fn=len(objects)-tp,
                             q1_recall=np.mean([hits(o, name) for o in objects if o["short_side_640_group"] == "Q1"]),
                             short_le16_recall=np.mean([hits(o, name) for o in objects if o["short_side_640"] <= 16]))
        errors.extend({"condition": name, "category": key, "count": counts[key]} for key in CATEGORIES)
        native_classes = read_csv(run / name / "per_class.csv")
        for i, class_id in enumerate(class_ids):
            native_class = native_classes[class_id]
            if not np.allclose([ap[i, 0], ap[i].mean()],
                               [float(native_class[k]) for k in ("AP50", "AP50-95")], atol=1e-8, rtol=0):
                raise ValueError(f"{name} class {class_id} failed native AP parity.")
            group = [o for o in objects if o["class_id"] == class_id]
            classes.append({"condition": name, "class": names[class_id], "priority": names[class_id] in FOCUS,
                            "AP50": ap[i, 0], "AP50-95": ap[i].mean(), **summarize_group(group)})

    for field in GEOMETRY:
        for quartile in ("Q1", "Q2", "Q3", "Q4"):
            group = [o for o in objects if o[f"{field}_group"] == quartile]
            sizes.append({"measure": field, "group": quartile, **summarize_group(group)})
    for i, label in enumerate(("≤16", "(16, 24]", "(24, 32]", "(32, 64]", ">64")):
        group = [o for o in objects if np.searchsorted(SHORT_BOUNDS, o["short_side_640"], side="left") == i]
        small.append({"short_side_px": label, **summarize_group(group)})
    for budget in conf["FP_PER_IMAGE_BUDGETS"]:
        row = {"fp_per_image_budget": budget}
        for name in MODELS:
            best = max((p for p in curves[name] if p["fp_per_image"] <= budget),
                       key=lambda point: (point["recall"], -point["fp_per_image"]))
            row.update({f"{key}_{name}": value for key, value in best.items()})
            row[f"budget_within_sweep_{name}"] = budget <= curves[name][-1]["fp_per_image"]
        row["delta_recall"] = row["recall_A1"] - row["recall_A0"]
        budgets.append(row)
    paired_classes = []
    for class_name in names:
        a, b = [row for row in classes if row["class"] == class_name]
        row = {k: v for k, v in a.items() if k not in ("condition", "AP50", "AP50-95")}
        for key in ("AP50", "AP50-95"):
            row.update({f"{key}_A0": a[key], f"{key}_A1": b[key], f"delta_{key}": b[key]-a[key]})
        paired_classes.append(row)
    for filename, rows in (("metrics_comparison", [{"condition": name, **values} for name, values in metrics.items()]),
                           ("per_class_comparison", paired_classes), ("size_analysis", sizes),
                           ("small_defect_analysis", small), ("error_breakdown", errors),
                           ("recall_fp_comparison", budgets)):
        save_csv(run / f"{filename}.csv", rows)
    return metrics, paired_classes, sizes, small, errors, budgets


def bootstrap(run, conf, truth, objects, records, metrics):
    images = {image: i for i, image in enumerate(truth)}
    counts = np.zeros((len(images), 3, 3), dtype=int)
    for obj in objects:
        groups = np.array([True, obj["short_side_640_group"] == "Q1", obj["short_side_640"] <= 16])
        counts[images[obj["image"]]] += groups[:, None] * [1, hits(obj, "A0"), hits(obj, "A1")]
    orders = {name: [(images[im], row) for im, row in table.items()] for name, table in records.items()}
    ids = list(range(10))
    rolled = yaml.safe_load(CONFIG.read_text())["names"].index("rolled_pit")
    nine = [i for i in ids if i != rolled]
    rng = np.random.default_rng(conf["SEED"])
    reps = conf["BOOTSTRAP_REPETITIONS"]
    deltas = np.empty((reps, len(BOOT_KEYS)))
    for i in range(reps):
        multiplicity = np.bincount(rng.integers(len(images), size=len(images)), minlength=len(images))
        aps = []
        for order in orders.values():
            # Preserve native image order, including tie ordering, with a shared multiplicity.
            selected = [row for index, row in order for _ in range(multiplicity[index])]
            ap, classes, *_ = per_class_ap(selected)
            aps.append(np.r_[mean_ap(ap, classes, ids), mean_ap(ap, classes, nine)])
        total, first, second = np.einsum("i,ijk->jk", multiplicity, counts).T
        recalls = np.divide(second-first, total, out=np.full(3, np.nan), where=total > 0)
        deltas[i] = np.r_[aps[1]-aps[0], recalls]
        if (i+1) % 500 == 0:
            print(f"Paired bootstrap {i+1}/{reps}", flush=True)
    rows = []
    for key, values in zip(BOOT_KEYS, deltas.T):
        valid = values[np.isfinite(values)]
        low, high = np.quantile(valid, [.025, .975]) if len(valid) else (None, None)
        rows.append({"metric": key, "delta_A1_minus_A0": metrics["A1"][key]-metrics["A0"][key],
                     "ci_2.5": low, "ci_97.5": high, "valid_repetitions": len(valid),
                     "insufficient_support": reps-len(valid), "seed": conf["SEED"]})
    save_csv(run / "bootstrap_results.csv", rows)
    np.save(run / "predictions" / "bootstrap_deltas.npy", deltas)
    return rows


def level_analysis(run, predictions, objects, detections):
    object_sizes = {o["object_id"]: o["short_side_640"] for o in objects}
    rows = []
    for name in MODELS:
        floor = Counter(p["level"] for items in predictions[name].values() for p in items)
        fixed = Counter((p["level"], p["category"]) for p in detections[name])
        small = Counter(p["level"] for p in detections[name] if p["category"] == "correct"
                        and object_sizes[p["object_id"]] <= 16)
        recovered = Counter(o["A1_matched_level"] for o in objects if o["outcome"] == "recovered") if name == "A1" else {}
        levels = [f"P{s.bit_length()-1}" for s in MODELS[name][3]]
        for level in (*levels, "ambiguous", "unattributed"):
            counts = {category: fixed[level, category] for category in CATEGORIES if category != "missed"}
            rows.append({"condition": name, "level": level, "post_nms_at_floor": floor[level],
                         "fixed_predictions": sum(counts.values()), "correct_short_le16": small[level],
                         "A1_recovered": recovered.get(level, 0) if name == "A1" else None, **counts})
    save_csv(run / "detection_level_analysis.csv", rows)
    return rows


def figures(run, conf, metrics, classes, sizes, small, errors, curves, objects, predictions, changes, timing, provenance):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    colors = ("#255F85", "#B24A38")
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, filename):
        for extension in ("pdf", "png"):
            fig.savefig(filename.with_suffix(f".{extension}"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    def bars(ax, labels, values, ylabel, title):
        x = np.arange(len(labels))
        for i, name in enumerate(MODELS):
            ax.bar(x + .36*(i-.5), values[name], .36, label=name, color=colors[i])
        ax.set_xticks(x, labels, rotation=30 if len(labels) > 4 else 0, ha="right" if len(labels) > 4 else "center")
        ax.set(ylabel=ylabel, title=title)
        ax.legend(frameon=False)

    def compare(filename, labels, values, ylabel, title):
        fig, ax = plt.subplots(figsize=(max(6, .75*len(labels)), 4), layout="constrained")
        bars(ax, labels, values, ylabel, title)
        save(fig, run / "plots" / filename)

    compare("overall_accuracy", ["AP50", "AP50–95", "Fixed precision", "Fixed recall"],
            {name: [metrics[name][k] for k in ("mAP50", "mAP50-95", "fixed_precision", "fixed_recall")] for name in MODELS},
            "Proportion", "Matched validation accuracy")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), layout="constrained")
    for ax, key in zip(axes, ("AP50", "AP50-95")):
        bars(ax, [r["class"].replace("_", " ") for r in classes],
             {name: [r[f"{key}_{name}"] for r in classes] for name in MODELS}, "Average precision", key)
    save(fig, run / "plots/per_class_ap")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for ax, group, label, title in zip(axes, ([r for r in sizes if r["measure"] == "area_640"],
                                              [r for r in sizes if r["measure"] == "short_side_640"], small),
                                       ("group", "group", "short_side_px"),
                                       ("Area quartiles", "Short-side quartiles", "Short side at 640 (px)")):
        bars(ax, [f"{r[label]} (n={r['objects']})" for r in group],
             {name: [r[f"recall_{name}"] for r in group] for name in MODELS}, "Fixed recall", title)
        ax.set_ylim(0, 1)
    save(fig, run / "plots/size_recall")
    compare("error_distribution", [key.replace("_", " ") for key in CATEGORIES],
            {name: [r["count"] for r in errors if r["condition"] == name] for name in MODELS}, "Count", "D1 detection assignments")

    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    for (name, curve), color in zip(curves.items(), colors):
        ax.step([r["fp_per_image"] for r in curve], [r["recall"] for r in curve], where="post", label=name, color=color)
    ax.set(xlabel="False positives per image", ylabel="Recall", xlim=(0, 1.05))
    ax.legend(frameon=False)
    save(fig, run / "plots/recall_fp")

    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    for name, color in zip(MODELS, colors):
        x = [row["batch1_median_ms"] for row in timing if row["condition"] == name and row["batch"] == 1]
        center = np.mean(x)
        ax.errorbar(center, metrics[name]["mAP50-95"], xerr=[[center-min(x)], [max(x)-center]],
                    marker="o", color=color, label=name)
    ax.set(xlabel="Batch-1 median end-to-end latency (ms)", ylabel="mAP50–95")
    ax.legend(frameon=False)
    save(fig, run / "plots/accuracy_latency")

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), layout="constrained")
    for ax, key, label, scale in zip(axes, ("parameters", "estimated_gflops"),
                                      ("Parameters (millions)", "Estimated GFLOPs"), (1e6, 1)):
        ax.bar(list(MODELS), [provenance[name][key]/scale for name in MODELS], color=colors)
        ax.set_ylabel(label)
    save(fig, run / "plots/architecture_cost")

    rng = np.random.default_rng(conf["SEED"])
    recovered = [o for o in objects if o["outcome"] == "recovered" and o["short_side_640"] <= 16]
    lost = [o for o in objects if o["outcome"] == "lost"]
    added = [p for p in changes if p["change"] == "added"]
    rolled = [o for o in objects if o["class"] == "rolled_pit"]
    for label, candidates in (("small-recovered", recovered), ("lost", lost), ("additional-fp", added), ("rolled-pit", rolled)):
        if not candidates:
            continue
        indices = np.arange(len(candidates)) if label == "rolled-pit" else rng.choice(len(candidates), min(4, len(candidates)), replace=False)
        for index, candidate in enumerate((candidates[i] for i in indices), 1):
            image = candidate["image"]
            with Image.open(DATASET / image) as source:
                pixels = source.convert("RGB")
            fig, axes = plt.subplots(2, 1, figsize=(10, 6), layout="constrained")
            for ax, name in zip(axes, MODELS):
                ax.imshow(pixels)
                for pred in predictions[name][image]:
                    if pred["score"] >= conf["CONFIDENCE"]:
                        x, y, x2, y2 = box(pred)
                        ax.add_patch(Rectangle((x, y), x2-x, y2-y, fill=False, edgecolor="#FEAB45", linewidth=1))
                if label != "additional-fp" or name == "A1":
                    x, y, x2, y2 = box(candidate)
                    ax.add_patch(Rectangle((x, y), x2-x, y2-y, fill=False, edgecolor="#31D9A0", linewidth=2))
                ax.set_title(f"{name} · {candidate.get(f'{name}_category', 'fixed-point predictions')}")
                ax.set_axis_off()
            fig.suptitle(f"{label.replace('-', ' ').capitalize()} · {candidate.get('object_id', Path(image).name)}")
            save(fig, run / "examples" / f"{label}-{index:02d}")


def report(run, summary, small, classes, rolled):
    metrics = summary["metrics"]
    lines = ["# A0 and A1 standalone validation", "", "Pending scientific review. A1 minus A0 is the matched comparison.",
             "B1 at 640 and E2 at 960 are historical references.",
             "", "| Condition | mAP50 | mAP50–95 |", "|---|---:|---:|"]
    for name, row in (metrics | summary["historical_references"]).items():
        lines.append(f"| {name} | {row['mAP50']:.2%} | {row['mAP50-95']:.2%} |")
    lines.extend(["", "| Sensitivity excluding rolled pit | A0 | A1 |", "|---|---:|---:|"])
    for label, key in (("AP50", "nine_class_AP50"), ("AP50–95", "nine_class_AP50-95")):
        lines.append(f"| {label} | {metrics['A0'][key]:.2%} | {metrics['A1'][key]:.2%} |")
    first = small[0]
    p2 = next(row for row in summary["levels"] if row["condition"] == "A1" and row["level"] == "P2")
    lines.extend(["", "## Small defects and paired outcomes", "",
                  f"Among {first['objects']} objects with a short side ≤16 px, fixed recall was "
                  f"{first['recall_A0']:.2%} for A0 and {first['recall_A1']:.2%} for A1. "
                  f"A1 recovered {first['recovered']} and lost {first['lost']}.",
                  "", f"All object outcomes: {summary['paired_outcomes']}.",
                  f"Shared, added and removed fixed-point false positives: {summary['false_positive_changes']}.",
                  f"P2 contributed {p2['correct']} correct detections at the fixed point, including "
                  f"{p2['correct_short_le16']} objects with short sides ≤16 px and {p2['A1_recovered'] or 0} "
                  f"recoveries. It contributed {p2['background']} background detections and {p2['duplicate']} duplicates.",
                  "", "## Priority classes", "", "| Class | AP50 A0 | AP50 A1 | AP50–95 A0 | AP50–95 A1 |",
                  "|---|---:|---:|---:|---:|"])
    for row in classes:
        if row["priority"]:
            lines.append(f"| {row['class']} | {row['AP50_A0']:.2%} | {row['AP50_A1']:.2%} | "
                         f"{row['AP50-95_A0']:.2%} | {row['AP50-95_A1']:.2%} |")
    lines.extend(["", "Rolled pit has four objects in two images. Per-object assignments and "
                  "class, IoU and confidence evidence are descriptive.", "",
                  "| Rolled-pit object | A0 outcome / class / IoU / score | A1 outcome / class / IoU / score |",
                  "|---|---|---|"])
    for obj in rolled:
        formatted = [" / ".join(str(obj[f"{name}_{k}"]) for k in ("category", "predicted_class", "iou", "score"))
                     for name in MODELS]
        lines.append(f"| {obj['object_id']} | {formatted[0]} | {formatted[1]} |")
    lines.extend(["", "## Paired image bootstrap and inference cost", "",
                  "Intervals describe validation-image uncertainty for the two fixed checkpoints. Training-seed variability is unmeasured.",
                  "AP intervals use draws with support for all classes in the respective mean.", "",
                  "| Metric | A1 − A0 | 95% interval | Valid draws |", "|---|---:|---:|---:|"])
    for row in summary["bootstrap"]:
        ci = f"[{row['ci_2.5']:+.2%}, {row['ci_97.5']:+.2%}]" if row["ci_2.5"] is not None else "undefined"
        lines.append(f"| {row['metric']} | {row['delta_A1_minus_A0']:+.2%} | {ci} | {row['valid_repetitions']} |")
    lines.extend(["", "Timing uses native uninstrumented FP32 inference and NMS in A0, A1, A1, A0 order. "
                  "The training durations and displayed training-memory peaks are historical.",
                  "", "| Model | Batch-1 median latency, range across passes (ms) | Batch-16 images/s, range |",
                  "|---|---:|---:|"])
    for name in MODELS:
        rows = [row for row in summary["timing"] if row["condition"] == name]
        latency = [row["batch1_median_ms"] for row in rows if row["batch"] == 1]
        throughput = [row["images_per_second"] for row in rows if row["batch"] == 16]
        lines.append(f"| {name} | {min(latency):.2f}–{max(latency):.2f} | {min(throughput):.2f}–{max(throughput):.2f} |")
    lines.extend([
                  "", "See [timing passes](timing_comparison.csv), [detection levels](detection_level_analysis.csv), "
                  "[object pairs](per_object_comparison.csv), and [same-budget recall](recall_fp_comparison.csv)."])
    (run / "report.md").write_text("\n".join(lines) + "\n")


def run_evaluation(run, conf, source, prior, truth, training, device):
    for name in ("predictions", "plots", "examples"):
        (run / name).mkdir()
    write_json(run / "source.json", source)
    write_json(run / "environment.json", device)
    (run / "effective_config.yaml").write_text(yaml.safe_dump(conf, sort_keys=False))
    write_json(run / "training_provenance.json", training)
    write_json(run / "summary.json", {"status": "started"})
    try:
        checkpoints = {name: ROOT / conf[f"{name}_CHECKPOINT"] for name in MODELS}
        predictions, records = {}, {}
        for name, checkpoint in checkpoints.items():
            predictions[name], records[name] = export_validation(
                run, conf, conf["IMAGE_SIZE"], truth, checkpoint, name=name, inspect_predictions=prediction_levels)
            args = read_json(run / name / "effective_config.json")
            if any(args[key] != conf[option] for option, (key, _) in NATIVE_OPTIONS.items()):
                raise RuntimeError(f"{name} native validator did not use the registered configuration.")
        order = [(name, checkpoints[name], conf["IMAGE_SIZE"]) for name in ("A0", "A1", "A1", "A0")]
        timing = measure_timing(run, conf, checkpoints["A0"], architecture_passes=order)
        objects, detections, curves, changes = paired_objects(run, conf, truth, predictions)
        rolled = [row for row in objects if row["class"] == "rolled_pit"]
        save_csv(run / "per_object_comparison.csv", objects)
        save_csv(run / "rolled_pit_analysis.csv", rolled)
        metrics, classes, sizes, small, errors, budgets = analyze(run, conf, objects, detections, curves, records)
        intervals = bootstrap(run, conf, truth, objects, records, metrics)
        levels = level_analysis(run, predictions, objects, detections)
        figures(run, conf, metrics, classes, sizes, small, errors, curves, objects, predictions, changes, timing, training)
        b1 = prior["metrics"]["B1"]["native"]
        # Historical values are from the reviewed E2 report, rounded to published precision.
        summary = {
            "status": "completed", "split": "val", "images": len(truth), "objects": len(objects), "classes": 10,
            "dataset_manifest_sha256": prior["dataset_manifest_sha256"],
            "d1_objects_sha256": sha256(PRIOR / "per_object.csv"), "upstream_commit": UPSTREAM_COMMIT,
            "checkpoint_sha256": {name: data["checkpoint_sha256"] for name, data in training.items()},
            "metrics": metrics, "native_parity_passed": True, "bootstrap": intervals, "timing": timing,
            "device": device, "levels": levels, "paired_outcomes": dict(Counter(o["outcome"] for o in objects)),
            "poorer_iou_among_retained": sum(o["poorer_correct_iou"] for o in objects),
            "false_positive_changes": dict(Counter(p["change"] for p in changes)),
            "added_false_positives_by_category": dict(Counter(p["category"] for p in changes if p["change"] == "added")),
            "added_false_positives_by_level": dict(Counter(p["level"] for p in changes if p["change"] == "added")),
            "duplicate_counts": {name: sum(p["category"] == "duplicate" for p in rows) for name, rows in detections.items()},
            "historical_references": {"B1 640": {"mAP50": b1["metrics/mAP50(B)"], "mAP50-95": b1["metrics/mAP50-95(B)"]},
                                      "E2 960": {"mAP50": .6799, "mAP50-95": .3423}},
        }
        report(run, summary, small, classes, rolled)
        write_json(run / "summary.json", summary)
    except (Exception, KeyboardInterrupt):
        traceback.print_exc()
        write_json(run / "summary.json", {"status": "failed"})
        raise SystemExit(1) from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/evaluation/age_yolo_a0_a1.yaml")
    parser.add_argument("--preflight", action="store_true", help="Verify inputs on the login node without creating a run.")
    args = parser.parse_args()
    conf = load_config(args.config)
    run = ROOT / "experiments" / conf["RUN_ID"]
    if run.exists() or run.is_symlink():
        parser.error(f"Run directory already exists: {run}")
    source = verify_source(args.config, conf)
    prior, truth = frozen_inputs()
    objects = [row for rows in truth.values() for row in rows]
    rolled = [row for row in objects if row["class"] == "rolled_pit"]
    if (prior["quartiles"]["area_640"] != [1055.859375, 6759.375, 16768.359375]
            or sum(row["short_side_640"] <= 16 for row in objects) != 35
            or (len(rolled), len({row["image"] for row in rolled})) != (4, 2)
            or len({row["class_id"] for row in objects}) != 10):
        raise ValueError("D1 subset counts, class support or boundaries differ.")
    training = verify_training(conf, prior)
    sys.path.insert(0, str(UPSTREAM))
    sys.dont_write_bytecode = True
    with TemporaryDirectory(prefix="flawtrace-a1-settings-") as temp:
        os.environ.update(YOLO_CONFIG_DIR=temp, YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false", MPLBACKEND="Agg")
        import torch
        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        verify_architectures(conf, training)
        if args.preflight:
            check_validation_images()
            print("Preflight passed: checkpoint architecture, training, source, dataset hashes, D1 objects and fresh run ID.")
            print(f"Configured CUDA device: {conf['DEVICE']}. Availability and index are checked at runtime.")
            return
        torch.set_num_threads(threads)
        device = initialize_cuda(conf["DEVICE"])
        props = torch.cuda.get_device_properties(device)
        gpu = {"device": str(device), "gpu_uuid": f"GPU-{str(props.uuid).removeprefix('GPU-')}",
               "gpu": props.name, "total_bytes": props.total_memory, "pbs_job_id": os.getenv("PBS_JOBID")}
        run.mkdir(parents=True, exist_ok=False)
        with (run / "terminal.log").open("x", buffering=1) as log:
            with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                print(f"{device} · {gpu['gpu']} · {gpu['gpu_uuid']}")
                run_evaluation(run, conf, source, prior, truth, training, gpu)


if __name__ == "__main__":
    main()
