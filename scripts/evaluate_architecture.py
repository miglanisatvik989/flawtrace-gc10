"""Compare frozen architecture checkpoints on the original validation split."""

import argparse
import hashlib
import os
import re
import shutil
import sys
import tarfile
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
    "A2": ("a2-age-yolo-p2-context-640-b32-01", "configs/models/age_yolo_p2_context.yaml",
           "d89418563912c8f240d069ec4a666451efd987177ccf69ba2cd1cdedd992db0d", (4, 8, 16, 32), 3451715, 10.007,
           273, 300, 3.685, 14.9),
}
REFERENCE_RUN = "experiments/a1-architecture-eval-01"
REFERENCE_ARCHIVE = "artifacts/A1/a1-architecture-eval-01.tar"
REFERENCE_SHA256 = "4a303104f75f3e5cc53fc73e2e5f1ad6ab24215e965095703aa55bb6c7a3d541"
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


def comparisons(conf):
    return (("A2", "A1"), ("A2", "A0")) if "A2_CHECKPOINT" in conf else (("A1", "A0"),)


def contrast(challenger, baseline):
    return f"{challenger}_minus_{baseline}"


def load_config(path):
    conf = yaml.safe_load(path.read_text())
    fixed = {key: value for key, (_, value) in NATIVE_OPTIONS.items()} | DESIGN
    models = MODELS if isinstance(conf, dict) and "A2_CHECKPOINT" in conf else ("A0", "A1")
    fixed.update({f"{model}_CHECKPOINT": f"experiments/{MODELS[model][0]}/weights/best.pt" for model in models})
    if not isinstance(conf, dict) or set(conf) != set(fixed) | {"RUN_ID", "DEVICE"}:
        raise ValueError("Expected the registered flat architecture evaluation settings.")
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
    augmented = ROOT / "data/gc10-det-aug42-x4"
    train_digest = sha256(augmented / "train.csv")
    augmentation = read_json(augmented / "augmentation.json")
    if (augmentation["train_manifest_sha256"] != train_digest
            or any(sha256(MANIFESTS / f"gc10-{split}.csv") != digest
                   for split, digest in augmentation["source_manifest_sha256"].items())):
        raise ValueError("Augmented training manifest or original split identities differ.")
    provenance = {}
    for name, (run_id, model_file, digest, strides, params, flops, best, epochs, hours, peak) in MODELS.items():
        if f"{name}_CHECKPOINT" not in conf:
            continue
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
        if (attempt["train_manifest_sha256"] != train_digest or attempt["augmentation"] != augmentation
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
                or (epochs < 300 and f"Best results observed at epoch {best}," not in log)
                or f"{epochs} epochs completed in {hours:.3f} hours." not in log or gpu_peak != peak):
            raise ValueError(f"{name} training results or early stopping differ.")
        if name == "A2":
            module = "scripts/context_sifhead.py"
            if (source["sha256"].get(module) != sha256(ROOT / module)
                    or tuple(float(winning[key]) for key in ("metrics/mAP50(B)", "metrics/mAP50-95(B)"))
                    != (.65791, .34447)):
                raise ValueError("A2 context module provenance or best-epoch AP differs.")
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
    reference = provenance["A0"]
    for name, item in provenance.items():
        first, second = reference["effective_config"]["args"], item["effective_config"]["args"]
        differences = {key for key in first.keys() | second.keys() if first.get(key) != second.get(key)}
        if differences - {"model", "device", "name", "save_dir"}:
            raise ValueError(f"{name} effective methodology differs: {sorted(differences)}")
        if name == "A1" and any(reference[key] != item[key] for key in ("trainer_sha256", "training_commit")):
            raise ValueError("A0 and A1 trainer provenance differs.")
    return provenance


def verify_architectures(conf, provenance):
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    for name, info in provenance.items():
        model = YOLO(str(ROOT / conf[f"{name}_CHECKPOINT"])).model.float().eval()
        configuration = yaml.safe_load(Path(info["architecture"]).read_text())
        if (any(model.yaml[key] != configuration[key] for key in ("nc", "scales", "backbone", "head"))
                or type(model.model[-1]).__name__ != configuration["head"][-1][2]
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


def reference_predictions(conf, truth, prior):
    reference = ROOT / REFERENCE_RUN
    archive = ROOT / REFERENCE_ARCHIVE
    if sha256(archive) != REFERENCE_SHA256:
        raise ValueError("A0/A1 evaluation archive SHA-256 differs from the reviewed archive.")
    hashes = {}
    inputs = ["summary.json", "effective_config.yaml"]
    for name in ("A0", "A1"):
        inputs.extend(f"{name}/{file}" for file in ("metrics.json", "per_class.csv", "effective_config.json", "run.json"))
        inputs.extend((f"predictions/{name}.json", f"predictions/{name}-ap.npz"))
    with tarfile.open(archive) as verified:
        for file in inputs:
            with verified.extractfile(f"{REFERENCE_RUN}/{file}") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if sha256(reference / file) != digest:
                raise ValueError(f"A0/A1 input differs from reviewed archive: {file}")
            hashes[file] = digest
    summary = read_json(reference / "summary.json")
    if (summary["status"] != "completed" or not summary["native_parity_passed"]
            or summary["dataset_manifest_sha256"] != prior["dataset_manifest_sha256"]
            or summary["upstream_commit"] != UPSTREAM_COMMIT
            or summary["d1_objects_sha256"] != sha256(PRIOR / "per_object.csv")
            or (summary["images"], summary["objects"], summary["classes"], summary["split"]) != (230, 345, 10, "val")):
        raise ValueError("The reviewed A0/A1 evaluation does not match the frozen validation inputs.")
    predictions, records = {}, {}
    for name in ("A0", "A1"):
        run = read_json(reference / name / "run.json")
        args = read_json(reference / name / "effective_config.json")
        if (summary["checkpoint_sha256"][name] != MODELS[name][2] or run["status"] != "completed"
                or run["checkpoint"]["sha256"] != MODELS[name][2]
                or any(args[key] != conf[option] for option, (key, _) in NATIVE_OPTIONS.items())):
            raise ValueError(f"The reviewed {name} evaluation differs from this protocol.")
        rows = read_json(reference / "predictions" / f"{name}.json")
        records[name] = {}
        with np.load(reference / "predictions" / f"{name}-ap.npz", allow_pickle=False) as data:
            images = data["images"].tolist()
            if rows.keys() != truth.keys() or set(images) != truth.keys() or len(images) != len(truth):
                raise ValueError(f"{name} prediction image identities differ from D1.")
            tp, scores, classes, targets = (data[key] for key in ("tp", "conf", "pred_cls", "target_cls"))
            pred_offset = target_offset = 0
            for image, n_pred, n_target in zip(images, data["prediction_counts"], data["target_counts"], strict=True):
                pred_span = slice(pred_offset, pred_offset + n_pred)
                target_span = slice(target_offset, target_offset + n_target)
                record = (tp[pred_span], scores[pred_span], classes[pred_span], targets[target_span])
                if (len(rows[image]) != n_pred or len(truth[image]) != n_target
                        or Counter(record[3]) != Counter(obj["class_id"] for obj in truth[image])
                        or not np.array_equal(record[1], [p["score"] for p in rows[image]])
                        or not np.array_equal(record[2], [p["class_id"] for p in rows[image]])):
                    raise ValueError(f"Inconsistent {name} prediction and AP exports for {image}.")
                records[name][image] = record
                pred_offset += n_pred
                target_offset += n_target
            if pred_offset != len(scores) or target_offset != len(targets):
                raise ValueError(f"Unexpected {name} prediction export lengths.")
        predictions[name] = rows
        checked, *_ = native_metrics(reference, name, records[name])
        if any(not np.isclose(checked[key], summary["metrics"][name][key], rtol=0, atol=1e-8) for key in checked):
            raise ValueError(f"The reviewed {name} summary disagrees with recalculated native metrics.")
    return predictions, records, hashes


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


def summarize_group(objects, names, pairs):
    row = {"objects": len(objects), "images": len({o["image"] for o in objects}),
           "recovered": sum(o["outcome"] == "recovered" for o in objects),
           "lost": sum(o["outcome"] == "lost" for o in objects)}
    for name in names:
        ious = [o[f"{name}_iou"] for o in objects if hits(o, name)]
        row.update({f"recall_{name}": ratio(len(ious), len(objects)),
                    f"localization_{name}": sum(o[f"{name}_category"] == "localization" for o in objects),
                    f"median_correct_iou_{name}": float(np.median(ious)) if ious else None})
    for challenger, baseline in pairs:
        label = contrast(challenger, baseline)
        row.update({f"{outcome}_{label}": sum(o[f"outcome_{label}"] == outcome for o in objects)
                    for outcome in ("recovered", "lost")})
        for key in ("recall", "localization", "median_correct_iou"):
            a, b = row[f"{key}_{baseline}"], row[f"{key}_{challenger}"]
            delta = b - a if a is not None and b is not None else None
            row[f"delta_{key}_{label}"] = delta
            if (challenger, baseline) == pairs[0]:
                row[f"delta_{key}"] = delta
    return row


def compare_false_positives(image, fixed, threshold, challenger, baseline):
    a0 = [row for row in fixed[baseline] if row["category"] != "correct"]
    a1 = [row for row in fixed[challenger] if row["category"] != "correct"]
    reference = [dict(row, object_id=str(i)) for i, row in enumerate(a0)]
    guesses = [dict(row, index=i) for i, row in enumerate(a1)]
    _, paired = match(reference, guesses, 0, threshold, threshold)
    counterparts = {row["index"]: int(row["object_id"]) for row in paired if row["category"] == "correct"}
    return [dict(row, image=image, change="shared" if i in counterparts else "added",
                 baseline_fp_index=counterparts.get(i)) for i, row in enumerate(a1)] + [
                     dict(row, image=image, change="removed", baseline_fp_index=i)
                     for i, row in enumerate(a0) if i not in counterparts.values()]


def paired_objects(run, conf, truth, predictions, *, pairs=None):
    objects, changes = [], []
    pairs = pairs or comparisons(conf)
    detections = {name: [] for name in predictions}
    curves = {name: [] for name in predictions}
    for image, targets in truth.items():
        matches, fixed = {}, {}
        for name in predictions:
            guesses = predictions[name][image]
            matches[name], fixed[name] = match(targets, guesses, conf["CONFIDENCE"], conf["MATCH_IOU"], conf["BACKGROUND_IOU"])
            detections[name].extend(dict(row, image=image) for row in fixed[name])
            curves[name].extend(match(targets, guesses, conf["NMS_CONFIDENCE_FLOOR"], conf["MATCH_IOU"],
                                      conf["BACKGROUND_IOU"])[1])
        for challenger, baseline in pairs:
            changes.extend(dict(row, comparison=contrast(challenger, baseline)) for row in
                           compare_false_positives(image, fixed, conf["MATCH_IOU"], challenger, baseline))
        for index, target in enumerate(targets):
            row = dict(target)
            for name in predictions:
                row.update({f"{name}_{key}": value for key, value in matches[name][index].items()})
            for challenger, baseline in pairs:
                label = contrast(challenger, baseline)
                pair = hits(row, baseline), hits(row, challenger)
                outcome = {(True, True): "retained", (False, True): "recovered",
                           (True, False): "lost", (False, False): "both_missed"}[pair]
                delta = row[f"{challenger}_iou"] - row[f"{baseline}_iou"] if all(pair) else None
                row[f"outcome_{label}"] = outcome
                row[f"delta_correct_iou_{label}"] = delta
                if (challenger, baseline) == pairs[0]:
                    row.update(outcome=outcome, delta_correct_iou=delta, poorer_correct_iou=delta is not None and delta < 0)
            for name in predictions:
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
    for name in predictions:
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


def native_metrics(run, name, records):
    names = yaml.safe_load(CONFIG.read_text())["names"]
    ids = list(range(len(names)))
    nine = [i for i in ids if names[i] != "rolled_pit"]
    ap, class_ids, p, r = per_class_ap(records.values())
    metrics = dict(zip(AP_KEYS, [*mean_ap(ap, class_ids, ids), *mean_ap(ap, class_ids, nine)]))
    native = read_json(run / name / "metrics.json")
    parity = {"mAP50": metrics["mAP50"], "mAP50-95": metrics["mAP50-95"],
              "precision": p.mean(), "recall": r.mean()}
    if not all(np.isclose(v, native[f"metrics/{key}(B)"], atol=1e-8, rtol=0) for key, v in parity.items()):
        raise ValueError(f"{name} prediction export failed native metric parity.")
    native_classes = read_csv(run / name / "per_class.csv")
    for i, class_id in enumerate(class_ids):
        native_class = native_classes[class_id]
        if not np.allclose([ap[i, 0], ap[i].mean()],
                           [float(native_class[k]) for k in ("AP50", "AP50-95")], atol=1e-8, rtol=0):
            raise ValueError(f"{name} class {class_id} failed native AP parity.")
    metrics.update(native_precision=float(p.mean()), native_recall=float(r.mean()))
    return metrics, ap, class_ids


def analyze(run, conf, objects, detections, curves, records, *, pairs=None):
    names = yaml.safe_load(CONFIG.read_text())["names"]
    pairs = pairs or comparisons(conf)
    metrics, classes, errors, sizes, small, budgets = {}, [], [], [], [], []
    for name in records:
        metrics[name], ap, class_ids = native_metrics(run, name, records[name])
        counts = Counter(d["category"] for d in detections[name])
        counts["missed"] = sum(o[f"{name}_category"] == "missed" for o in objects)
        tp = counts["correct"]
        metrics[name].update(fixed_precision=ratio(tp, len(detections[name])), fixed_recall=tp/len(objects),
                             tp=tp, fp=len(detections[name])-tp, fn=len(objects)-tp,
                             q1_recall=np.mean([hits(o, name) for o in objects if o["short_side_640_group"] == "Q1"]),
                             short_le16_recall=np.mean([hits(o, name) for o in objects if o["short_side_640"] <= 16]))
        errors.extend({"condition": name, "category": key, "count": counts[key]} for key in CATEGORIES)
        classes.extend({"condition": name, "class": names[class_id], "priority": names[class_id] in FOCUS,
                        "AP50": ap[i, 0], "AP50-95": ap[i].mean()}
                       for i, class_id in enumerate(class_ids))

    for field in GEOMETRY:
        for quartile in ("Q1", "Q2", "Q3", "Q4"):
            group = [o for o in objects if o[f"{field}_group"] == quartile]
            sizes.append({"measure": field, "group": quartile, **summarize_group(group, records, pairs)})
    for i, label in enumerate(("≤16", "(16, 24]", "(24, 32]", "(32, 64]", ">64")):
        group = [o for o in objects if np.searchsorted(SHORT_BOUNDS, o["short_side_640"], side="left") == i]
        small.append({"short_side_px": label, **summarize_group(group, records, pairs)})
    for budget in conf["FP_PER_IMAGE_BUDGETS"]:
        row = {"fp_per_image_budget": budget}
        for name in records:
            best = max((p for p in curves[name] if p["fp_per_image"] <= budget),
                       key=lambda point: (point["recall"], -point["fp_per_image"]))
            row.update({f"{key}_{name}": value for key, value in best.items()})
            row[f"budget_within_sweep_{name}"] = budget <= curves[name][-1]["fp_per_image"]
        for challenger, baseline in pairs:
            row[f"delta_recall_{contrast(challenger, baseline)}"] = row[f"recall_{challenger}"] - row[f"recall_{baseline}"]
        budgets.append(row)
    paired_classes = []
    for class_name in names:
        options = {r["condition"]: r for r in classes if r["class"] == class_name}
        group = [o for o in objects if o["class"] == class_name]
        row = {"class": class_name, "priority": class_name in FOCUS, **summarize_group(group, records, pairs)}
        for key in ("AP50", "AP50-95"):
            row.update({f"{key}_{name}": item[key] for name, item in options.items()})
            for challenger, baseline in pairs:
                row[f"delta_{key}_{contrast(challenger, baseline)}"] = options[challenger][key]-options[baseline][key]
        paired_classes.append(row)
    for filename, rows in (("metrics_comparison", [{"condition": name, **values} for name, values in metrics.items()]),
                           ("per_class_comparison", paired_classes), ("size_analysis", sizes),
                           ("small_defect_analysis", small), ("error_breakdown", errors),
                           ("recall_fp_comparison", budgets)):
        save_csv(run / f"{filename}.csv", rows)
    return metrics, paired_classes, sizes, small, errors, budgets


def bootstrap(run, conf, truth, objects, records, metrics, *, pairs=None, budget_recall=None):
    images = {image: i for i, image in enumerate(truth)}
    pairs = pairs or comparisons(conf)
    keys = (*BOOT_KEYS, "primary_budget_recall") if budget_recall else BOOT_KEYS
    names = list(records)
    counts = np.zeros((len(images), 3, 1 + len(names)), dtype=int)
    for obj in objects:
        groups = np.array([True, obj["short_side_640_group"] == "Q1", obj["short_side_640"] <= 16])
        counts[images[obj["image"]]] += groups[:, None] * [1, *(hits(obj, name) for name in names)]
    orders = {name: [(images[im], row) for im, row in table.items()] for name, table in records.items()}
    ids = list(range(10))
    rolled = yaml.safe_load(CONFIG.read_text())["names"].index("rolled_pit")
    nine = [i for i in ids if i != rolled]
    rng = np.random.default_rng(conf["SEED"])
    reps = conf["BOOTSTRAP_REPETITIONS"]
    deltas = {contrast(*pair): np.empty((reps, len(keys))) for pair in pairs}
    for i in range(reps):
        multiplicity = np.bincount(rng.integers(len(images), size=len(images)), minlength=len(images))
        aps = {}
        for name, order in orders.items():
            # Preserve native image order, including tie ordering, with a shared multiplicity.
            selected = [row for index, row in order for _ in range(multiplicity[index])]
            ap, classes, *_ = per_class_ap(selected)
            aps[name] = np.r_[mean_ap(ap, classes, ids), mean_ap(ap, classes, nine)]
        total, *hits_by_model = np.einsum("i,ijk->jk", multiplicity, counts).T
        hits_by_model = dict(zip(names, hits_by_model))
        budget = budget_recall(multiplicity) if budget_recall else None
        for challenger, baseline in pairs:
            change = hits_by_model[challenger] - hits_by_model[baseline]
            recalls = np.divide(change, total, out=np.full(3, np.nan), where=total > 0)
            extra = [budget[challenger] - budget[baseline]] if budget is not None else []
            deltas[contrast(challenger, baseline)][i] = np.r_[aps[challenger]-aps[baseline], recalls, extra]
        if (i+1) % 500 == 0:
            print(f"Paired bootstrap {i+1}/{reps}", flush=True)
    rows = []
    for challenger, baseline in pairs:
        label = contrast(challenger, baseline)
        for key, values in zip(keys, deltas[label].T):
            valid = values[np.isfinite(values)]
            low, high = np.quantile(valid, [.025, .975]) if len(valid) else (None, None)
            rows.append({"comparison": label, "metric": key, "delta": metrics[challenger][key]-metrics[baseline][key],
                         "ci_2.5": low, "ci_97.5": high, "valid_repetitions": len(valid),
                         "insufficient_support": reps-len(valid), "seed": conf["SEED"]})
    save_csv(run / "bootstrap_results.csv", rows)
    if len(pairs) > 1:
        np.savez_compressed(run / "predictions" / "bootstrap_deltas.npz", **deltas)
    else:
        np.save(run / "predictions" / "bootstrap_deltas.npy", next(iter(deltas.values())))
    return rows


def level_analysis(run, conf, predictions, objects, detections):
    object_sizes = {o["object_id"]: o["short_side_640"] for o in objects}
    rows = []
    pairs = comparisons(conf)
    for name in predictions:
        floor = Counter(p["level"] for items in predictions[name].values() for p in items)
        fixed = Counter((p["level"], p["category"]) for p in detections[name])
        small = Counter(p["level"] for p in detections[name] if p["category"] == "correct"
                        and object_sizes[p["object_id"]] <= 16)
        recovered = {contrast(c, b): Counter(o[f"{name}_matched_level"] for o in objects
                                             if o[f"outcome_{contrast(c, b)}"] == "recovered")
                     for c, b in pairs if c == name}
        levels = [f"P{stride.bit_length()-1}" for stride in MODELS[name][3]]
        for level in (*levels, "ambiguous", "unattributed"):
            counts = {category: fixed[level, category] for category in CATEGORIES if category != "missed"}
            rows.append({"condition": name, "level": level, "post_nms_at_floor": floor[level],
                         "fixed_predictions": sum(counts.values()), "correct_short_le16": small[level],
                         **{f"recovered_{label}": recovered.get(label, {}).get(level, 0) if label in recovered else None
                            for label in (contrast(*pair) for pair in pairs)}, **counts})
    save_csv(run / "detection_level_analysis.csv", rows)
    return rows


def figures(run, conf, metrics, classes, sizes, small, errors, curves, objects, predictions, changes,
            timing=None, provenance=None, *, pairs=None):
    pairs = pairs or comparisons(conf)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    colors = ("#255F85", "#B24A38", "#397754")
    challenger = pairs[0][0]
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, filename):
        for extension in ("pdf", "png"):
            fig.savefig(filename.with_suffix(f".{extension}"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    def bars(ax, labels, values, ylabel, title):
        x = np.arange(len(labels))
        for i, name in enumerate(metrics):
            ax.bar(x + .25*(i-(len(metrics)-1)/2), values[name], .25, label=name, color=colors[i])
        ax.set_xticks(x, labels, rotation=30 if len(labels) > 4 else 0, ha="right" if len(labels) > 4 else "center")
        ax.set(ylabel=ylabel, title=title)
        ax.legend(frameon=False)

    def compare(filename, labels, values, ylabel, title):
        fig, ax = plt.subplots(figsize=(max(6, .75*len(labels)), 4), layout="constrained")
        bars(ax, labels, values, ylabel, title)
        save(fig, run / "plots" / filename)

    compare("overall_accuracy", ["AP50", "AP50–95", "Fixed precision", "Fixed recall"],
            {name: [metrics[name][k] for k in ("mAP50", "mAP50-95", "fixed_precision", "fixed_recall")] for name in metrics},
            "Proportion", "Matched validation accuracy")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), layout="constrained")
    for ax, key in zip(axes, ("AP50", "AP50-95")):
        bars(ax, [r["class"].replace("_", " ") for r in classes],
             {name: [r[f"{key}_{name}"] for r in classes] for name in metrics}, "Average precision", key)
    save(fig, run / "plots/per_class_ap")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for ax, group, label, title in zip(axes, ([r for r in sizes if r["measure"] == "area_640"],
                                              [r for r in sizes if r["measure"] == "short_side_640"], small),
                                       ("group", "group", "short_side_px"),
                                       ("Area quartiles", "Short-side quartiles", "Short side at 640 (px)")):
        bars(ax, [f"{r[label]} (n={r['objects']})" for r in group],
             {name: [r[f"recall_{name}"] for r in group] for name in metrics}, "Fixed recall", title)
        ax.set_ylim(0, 1)
    save(fig, run / "plots/size_recall")
    compare("error_distribution", [key.replace("_", " ") for key in CATEGORIES],
            {name: [r["count"] for r in errors if r["condition"] == name] for name in metrics}, "Count", "D1 detection assignments")

    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    for (name, curve), color in zip(curves.items(), colors):
        ax.step([r["fp_per_image"] for r in curve], [r["recall"] for r in curve], where="post", label=name, color=color)
    ax.set(xlabel="False positives per image", ylabel="Recall", xlim=(0, 1.05))
    ax.legend(frameon=False)
    save(fig, run / "plots/recall_fp")

    if timing is not None:
        fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
        for name, color in zip(metrics, colors):
            x = [row["batch1_median_ms"] for row in timing if row["condition"] == name and row["batch"] == 1]
            center = np.mean(x)
            ax.errorbar(center, metrics[name]["mAP50-95"], xerr=[[center-min(x)], [max(x)-center]],
                        marker="o", color=color, label=name)
        ax.set(xlabel="Batch-1 median end-to-end latency (ms)", ylabel="mAP50–95")
        ax.legend(frameon=False)
        save(fig, run / "plots/accuracy_latency")

    if provenance is not None:
        fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), layout="constrained")
        for ax, key, label, scale in zip(axes, ("parameters", "estimated_gflops"),
                                          ("Parameters (millions)", "Estimated GFLOPs"), (1e6, 1)):
            ax.bar(list(metrics), [provenance[name][key]/scale for name in metrics], color=colors[:len(metrics)])
            ax.set_ylabel(label)
        save(fig, run / "plots/architecture_cost")

    rng = np.random.default_rng(conf["SEED"])
    fusion = pairs[0][0] == "F2"
    recovered = [o for o in objects if o["outcome"] == "recovered" and (fusion or o["short_side_640"] <= 16)]
    recovered_label = "recovered" if fusion else "small-recovered"
    lost = [o for o in objects if o["outcome"] == "lost"]
    added = [p for p in changes if p["change"] == "added" and p["comparison"] == contrast(*pairs[0])]
    rolled = [o for o in objects if o["class"] == "rolled_pit"]
    for label, candidates in ((recovered_label, recovered), ("lost", lost), ("additional-fp", added), ("rolled-pit", rolled)):
        if not candidates:
            continue
        indices = np.arange(len(candidates)) if label == "rolled-pit" else rng.choice(len(candidates), min(4, len(candidates)), replace=False)
        for index, candidate in enumerate((candidates[i] for i in indices), 1):
            image = candidate["image"]
            with Image.open(DATASET / image) as source:
                pixels = source.convert("RGB")
            fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3*len(metrics)), layout="constrained")
            for ax, name in zip(axes, metrics):
                ax.imshow(pixels)
                for pred in predictions[name][image]:
                    if pred["score"] >= conf["CONFIDENCE"]:
                        x, y, x2, y2 = box(pred)
                        ax.add_patch(Rectangle((x, y), x2-x, y2-y, fill=False, edgecolor="#FEAB45", linewidth=1))
                if label != "additional-fp" or name == challenger:
                    x, y, x2, y2 = box(candidate)
                    ax.add_patch(Rectangle((x, y), x2-x, y2-y, fill=False, edgecolor="#31D9A0", linewidth=2))
                ax.set_title(f"{name} · {candidate.get(f'{name}_category', 'fixed-point predictions')}")
                ax.set_axis_off()
            fig.suptitle(f"{label.replace('-', ' ').capitalize()} · {candidate.get('object_id', Path(image).name)}")
            save(fig, run / "examples" / f"{label}-{index:02d}")


def partial_coverage(run, conf, objects, predictions):
    from evaluate_e2 import coverage

    fields = ("iou", "gt_coverage", "area_ratio", "prediction_coverage")
    details, summaries = [], []
    for obj in objects:
        row = {key: obj[key] for key in ("image", "object_id", "class", "short_side_640_group")}
        for name in predictions:
            values = coverage(obj, predictions[name][obj["image"]])
            row.update({f"{name}_{key}": value for key, value in zip(fields, values)})
            row[f"{name}_category"] = obj[f"{name}_category"]
        for pair in comparisons(conf):
            row[f"outcome_{contrast(*pair)}"] = obj[f"outcome_{contrast(*pair)}"]
        details.append(row)
    for challenger, baseline in comparisons(conf):
        label = contrast(challenger, baseline)
        groups = {
            "all": details,
            "recovered": [row for row in details if row[f"outcome_{label}"] == "recovered"],
            "lost": [row for row in details if row[f"outcome_{label}"] == "lost"],
            "localization_either": [row for row in details if any(
                row[f"{name}_category"] == "localization" for name in (challenger, baseline))],
        }
        for group, rows in groups.items():
            paired = [r for r in rows if all(r[f"{name}_iou"] is not None for name in (challenger, baseline))]
            summary = {"comparison": label, "group": group, "objects": len(rows), "paired_candidates": len(paired)}
            for name in predictions:
                present = [r for r in rows if r[f"{name}_iou"] is not None]
                summary[f"candidates_{name}"] = len(present)
                summary[f"missing_candidates_{name}"] = len(rows) - len(present)
                for key in fields:
                    values = [r[f"{name}_{key}"] for r in present if r[f"{name}_{key}"] is not None]
                    summary[f"median_{key}_{name}"] = float(np.median(values)) if values else None
            for key in fields:
                changes = [r[f"{challenger}_{key}"]-r[f"{baseline}_{key}"] for r in paired
                           if all(r[f"{name}_{key}"] is not None for name in (challenger, baseline))]
                summary[f"paired_{key}_objects"] = len(changes)
                summary[f"paired_median_delta_{key}"] = float(np.median(changes)) if changes else None
            summaries.append(summary)
    save_csv(run / "partial_defect_per_object.csv", details)
    save_csv(run / "partial_defect_analysis.csv", summaries)
    return summaries


def report(run, conf, summary, small, classes, rolled):
    metrics = summary["metrics"]
    pairs = comparisons(conf)
    lines = ["# " + ", ".join(metrics) + " standalone validation", "", "Pending independent scientific review.",
             f"Primary comparison is {pairs[0][0]} minus {pairs[0][1]}.",
             "", "| Condition | mAP50 | mAP50–95 | Nine-class AP50 | Nine-class AP50–95 | Native P / R | Fixed P / R | TP / FP / FN |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for name, row in metrics.items():
        aps = " | ".join(f"{row[key]:.2%}" for key in AP_KEYS)
        native = f"{row['native_precision']:.2%} / {row['native_recall']:.2%}"
        fixed = f"{row['fixed_precision']:.2%} / {row['fixed_recall']:.2%}"
        lines.append(f"| {name} | {aps} | {native} | {fixed} | {row['tp']} / {row['fp']} / {row['fn']} |")
    if len(pairs) > 1:
        lines.extend(["", "A0 and A1 accuracy comes from the verified A1 evaluation archive. "
                      "All timing passes were run on the GPU assigned to this comparison."])
    lines.extend(["", "## Small defects and paired outcomes", ""])
    first = small[0]
    for challenger, baseline in pairs:
        label = contrast(challenger, baseline)
        lines.append(f"- {challenger} vs {baseline} at short sides ≤16 px ({first['objects']} objects) had "
                     f"{first[f'recall_{challenger}']:.2%} vs {first[f'recall_{baseline}']:.2%} recall, "
                     f"{first[f'recovered_{label}']} recovered and {first[f'lost_{label}']} lost. "
                     f"All-size outcomes were {summary['paired_outcomes_by_comparison'][label]}.")
    lines.extend(["", "## Priority classes", "",
                  "| Class | " + " | ".join(f"{name} AP50 / AP50–95" for name in metrics) + " |",
                  "|---|" + "---:|" * len(metrics)])
    for row in classes:
        if row["priority"]:
            values = " | ".join(f"{row[f'AP50_{name}']:.2%} / {row[f'AP50-95_{name}']:.2%}" for name in metrics)
            lines.append(f"| {row['class']} | {values} |")
    lines.extend(["", "Rolled pit has four objects in two images. Per-object assignments are descriptive.", "",
                  "| Rolled-pit object | " + " | ".join(f"{name} outcome / class / IoU / score / level" for name in metrics) + " |",
                  "|---|" + "---|" * len(metrics)])
    for obj in rolled:
        formatted = [" / ".join(str(obj[f"{name}_{key}"]) for key in
                                ("category", "predicted_class", "iou", "score", "matched_level")) for name in metrics]
        lines.append("| " + " | ".join((obj["object_id"], *formatted)) + " |")
    lines.extend(["", "## Paired image bootstrap and inference cost", "",
                  "Intervals describe validation-image uncertainty for fixed checkpoints from one training seed.",
                  "AP intervals use draws with support for all classes in the respective mean.", "",
                  "| Comparison | Metric | Difference (pp) | 95% interval (pp) | Valid draws |",
                  "|---|---|---:|---:|---:|"])
    for row in summary["bootstrap"]:
        ci = f"[{100*row['ci_2.5']:+.2f}, {100*row['ci_97.5']:+.2f}]" if row["ci_2.5"] is not None else "undefined"
        lines.append(f"| {row['comparison']} | {row['metric']} | {100*row['delta']:+.2f} | {ci} | {row['valid_repetitions']} |")
    lines.extend(["", "| Model | Mean batch-1 median latency (ms) | Mean batch-16 throughput (images/s) |",
                  "|---|---:|---:|"])
    for name in metrics:
        rows = [row for row in summary["timing"] if row["condition"] == name]
        latency = [row["batch1_median_ms"] for row in rows if row["batch"] == 1]
        throughput = [row["images_per_second"] for row in rows if row["batch"] == 16]
        lines.append(f"| {name} | {np.mean(latency):.2f} | {np.mean(throughput):.2f} |")
    links = ["[timing passes](timing_comparison.csv)", "[detection levels](detection_level_analysis.csv)",
             "[error categories](error_breakdown.csv)", "[object pairs](per_object_comparison.csv)",
             "[same-budget recall](recall_fp_comparison.csv)"]
    if len(pairs) > 1:
        links.append("[partial-defect coverage](partial_defect_analysis.csv)")
    lines.extend(["", "See " + ", ".join(links) + "."])
    (run / "report.md").write_text("\n".join(lines) + "\n")


def run_evaluation(run, conf, source, prior, truth, training, device, reference=None):
    for name in ("predictions", "plots", "examples"):
        (run / name).mkdir()
    write_json(run / "source.json", source)
    write_json(run / "environment.json", device)
    (run / "effective_config.yaml").write_text(yaml.safe_dump(conf, sort_keys=False))
    write_json(run / "training_provenance.json", training)
    write_json(run / "summary.json", {"status": "started"})
    try:
        checkpoints = {name: ROOT / conf[f"{name}_CHECKPOINT"] for name in training}
        predictions, records = {}, {}
        if reference is not None:
            predictions, records, hashes = reference
            for file in hashes:
                destination = run / (file if file.split("/", 1)[0] in ("A0", "A1", "predictions")
                                     else f"reference/{file}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / REFERENCE_RUN / file, destination)
                if sha256(destination) != hashes[file]:
                    raise ValueError(f"Reviewed input changed while staging: {file}")
            write_json(run / "reference_sha256.json", {"archive": REFERENCE_SHA256, "inputs": hashes})
        for name, checkpoint in checkpoints.items():
            if name in predictions:
                continue
            predictions[name], records[name] = export_validation(
                run, conf, conf["IMAGE_SIZE"], truth, checkpoint, name=name, inspect_predictions=prediction_levels)
            args = read_json(run / name / "effective_config.json")
            if any(args[key] != conf[option] for option, (key, _) in NATIVE_OPTIONS.items()):
                raise RuntimeError(f"{name} native validator did not use the registered configuration.")
        # Verify accuracy exports before timing. Timing reruns inference without instrumenting NMS.
        for name in records:
            native_metrics(run, name, records[name])
        order = [(name, checkpoints[name], conf["IMAGE_SIZE"])
                 for name in (*checkpoints, *reversed(checkpoints))]
        timing = measure_timing(run, conf, checkpoints["A0"], architecture_passes=order)
        objects, detections, curves, changes = paired_objects(run, conf, truth, predictions)
        rolled = [row for row in objects if row["class"] == "rolled_pit"]
        save_csv(run / "per_object_comparison.csv", objects)
        save_csv(run / "rolled_pit_analysis.csv", rolled)
        metrics, classes, sizes, small, errors, budgets = analyze(run, conf, objects, detections, curves, records)
        intervals = bootstrap(run, conf, truth, objects, records, metrics)
        levels = level_analysis(run, conf, predictions, objects, detections)
        coverage = partial_coverage(run, conf, objects, predictions) if reference is not None else []
        figures(run, conf, metrics, classes, sizes, small, errors, curves, objects, predictions, changes, timing, training)
        b1 = prior["metrics"]["B1"]["native"]
        # Historical values are from the reviewed E2 report, rounded to published precision.
        summary = {
            "status": "completed", "split": "val", "images": len(truth), "objects": len(objects), "classes": 10,
            "dataset_manifest_sha256": prior["dataset_manifest_sha256"],
            "d1_objects_sha256": sha256(PRIOR / "per_object.csv"), "upstream_commit": UPSTREAM_COMMIT,
            "checkpoint_sha256": {name: data["checkpoint_sha256"] for name, data in training.items()},
            "metrics": metrics, "native_parity_passed": True, "bootstrap": intervals, "timing": timing,
            "device": device, "levels": levels, "partial_defect_coverage": coverage,
            "paired_outcomes": dict(Counter(o["outcome"] for o in objects)),
            "paired_outcomes_by_comparison": {
                contrast(*pair): dict(Counter(o[f"outcome_{contrast(*pair)}"] for o in objects))
                for pair in comparisons(conf)
            },
            "poorer_iou_among_retained": sum(o["poorer_correct_iou"] for o in objects),
            "false_positive_changes": {
                contrast(*pair): dict(Counter(p["change"] for p in changes if p["comparison"] == contrast(*pair)))
                for pair in comparisons(conf)
            },
            "duplicate_counts": {name: sum(p["category"] == "duplicate" for p in rows) for name, rows in detections.items()},
            "historical_references": {"B1 640": {"mAP50": b1["metrics/mAP50(B)"], "mAP50-95": b1["metrics/mAP50-95(B)"]},
                                      "E2 960": {"mAP50": .6799, "mAP50-95": .3423}},
        }
        report(run, conf, summary, small, classes, rolled)
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
    with TemporaryDirectory(prefix="flawtrace-architecture-settings-") as temp:
        os.environ.update(YOLO_CONFIG_DIR=temp, YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false", MPLBACKEND="Agg")
        import torch

        reference = None
        if "A2_CHECKPOINT" in conf:
            from context_sifhead import register_context_sifhead

            register_context_sifhead()
            reference = reference_predictions(conf, truth, prior)
        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        verify_architectures(conf, training)
        if args.preflight:
            check_validation_images()
            print("Preflight passed: checkpoint architecture, training, source, dataset hashes, D1 objects and fresh run ID.")
            if reference is not None:
                print("Verified A0/A1 predictions against the reviewed archive and reproduced native metrics.")
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
                run_evaluation(run, conf, source, prior, truth, training, gpu, reference)


if __name__ == "__main__":
    main()
