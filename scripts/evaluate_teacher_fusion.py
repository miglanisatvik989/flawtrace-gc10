"""Evaluate same-resolution A0/A1 weighted boxes fusion from archived predictions."""

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import yaml

from analyze_errors import box, box_iou, match, read_csv, save_csv
from evaluate import CONFIG, MANIFESTS, ROOT, UPSTREAM, UPSTREAM_COMMIT, Tee, prepare_split, sha256, write_json
from evaluate_architecture import (
    DESIGN, NATIVE_OPTIONS, REFERENCE_ARCHIVE, REFERENCE_RUN, REFERENCE_SHA256,
    analyze, bootstrap, contrast, figures, paired_objects, per_class_ap, reference_predictions,
)
from evaluate_fusion import fused_records
from evaluate_resolution import PRIOR, frozen_inputs

PAIRS = (("F2", "F0"), ("F2", "F1"))
COORDINATES = ("x1", "y1", "x2", "y2")
WBF_REFERENCE = "651884c5c7df13a920d286404b47c6fa86944782"


def load_config(path):
    conf = yaml.safe_load(path.read_text())
    required = {key: value for key, (_, value) in NATIVE_OPTIONS.items()}
    required.update({key: value for key, value in DESIGN.items() if key not in ("WARMUP_BATCHES", "LATENCY_IMAGES")})
    required.update(WBF_IOU=.55, WBF_MODEL_WEIGHTS=[1, 1], WBF_CONFIDENCE="avg", PRIMARY_FP_BUDGET=.50)
    if not isinstance(conf, dict) or set(conf) != set(required) | {"RUN_ID"}:
        raise ValueError("Expected the registered flat S1-Fusion settings.")
    if any(type(conf[key]) is not type(value) or conf[key] != value for key, value in required.items()):
        raise ValueError("One or more S1-Fusion settings differ from the registered protocol.")
    if not isinstance(conf["RUN_ID"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", conf["RUN_ID"]):
        raise ValueError("Expected a simple run ID.")
    return conf


def weighted_boxes(first, second, width, height, conf):
    # Normalize for the reference algorithm, which assumes coordinates in [0, 1].
    scale = np.array([width, height, width, height])
    floor = conf["NMS_CONFIDENCE_FLOOR"]
    classes = {}
    for pred in (*first, *second):
        if pred["score"] < floor:
            continue
        coords = np.clip(np.asarray(box(pred), dtype=float) / scale, 0, 1)
        if coords[0] >= coords[2] or coords[1] >= coords[3]:
            continue
        classes.setdefault(pred["class_id"], []).append((pred["score"], coords))
    fused = []
    for cls, candidates in classes.items():
        order = np.argsort([score for score, _ in candidates])[::-1]
        sums, weights, counts, centers = [], [], [], []
        for index in order:
            score, coords = candidates[index]
            overlaps = box_iou([coords], centers)[0]
            if len(overlaps) and overlaps.max() > conf["WBF_IOU"]:
                j = int(overlaps.argmax())
                sums[j] += score * coords
                weights[j] += score
                counts[j] += 1
                centers[j] = sums[j] / weights[j]
            else:
                sums.append(np.asarray(score * coords, dtype=np.float32))
                weights.append(score)
                counts.append(1)
                centers.append(coords)
        for coords, total, count in zip(centers, weights, counts, strict=True):
            # Standard avg, no overflow, equal weights for two models.
            mean = float(np.float32(total / count)) if count > 1 else total
            score = mean * min(2, count) / 2
            if score >= floor:
                fused.append(dict(zip(COORDINATES, (coords * scale).tolist()), class_id=cls, score=score))
    order = np.argsort([pred["score"] for pred in fused])[::-1]
    return [fused[i] for i in order[:conf["MAX_DETECTIONS"]]]


def native_truth(run, truth):
    """Use the pinned validation label transform without constructing or running a model."""
    import torch
    from ultralytics.models.yolo.detect import DetectionValidator

    mapped = {}
    with TemporaryDirectory(prefix="native-labels-", dir=run) as temporary:
        staging = Path(temporary)
        prepare_split("val", MANIFESTS / "gc10-val.csv", staging)
        image_path = str(staging / "images/val")
        validator = DetectionValidator(save_dir=staging, args={
            "imgsz": 640, "rect": True, "batch": 16, "workers": 0, "split": "val", "plots": False})
        validator.device, validator.stride = torch.device("cpu"), 32
        validator.data = {**yaml.safe_load(CONFIG.read_text()), "channels": 3}
        for batch in validator.get_dataloader(image_path, 16):
            for index, path in enumerate(batch["im_file"]):
                image = f"images/val/{Path(path).name}"
                prepared = validator._prepare_batch(index, batch)
                coords = prepared["bbox"].tolist()
                classes = prepared["cls"].tolist()
                if image in mapped or classes != [obj["class_id"] for obj in truth[image]]:
                    raise ValueError(f"Native labels do not agree with D1 for {image}.")
                mapped[image] = [dict(obj, **dict(zip(COORDINATES, xyxy))) for obj, xyxy in zip(truth[image], coords, strict=True)]
    if mapped.keys() != truth.keys():
        raise ValueError("Native validation images differ from D1.")
    return mapped


def budget_recall_function(truth, predictions, conf):
    """Apply the primary FP cap inside each paired image bootstrap draw."""
    orders = {}
    total = np.array([len(rows) for rows in truth.values()])
    for condition, items in predictions.items():
        ranked = [(index, row) for index, (image, targets) in enumerate(truth.items())
                  for row in match(targets, items[image], conf["NMS_CONFIDENCE_FLOOR"],
                                   conf["MATCH_IOU"], conf["BACKGROUND_IOU"])[1]]
        ranked.sort(key=lambda pair: -pair[1]["score"])
        image_index = np.asarray([i for i, _ in ranked], dtype=int)
        correct = np.asarray([r["category"] == "correct" for _, r in ranked], dtype=bool)
        scores = np.asarray([r["score"] for _, r in ranked])
        ends = np.flatnonzero(np.r_[scores[1:] != scores[:-1], True]) if len(scores) else np.empty(0, dtype=int)
        orders[condition] = image_index, correct, ends

    def recall(multiplicity):
        objects = multiplicity @ total
        cap = conf["PRIMARY_FP_BUDGET"] * multiplicity.sum()
        results = {}
        for condition, (image_index, correct, ends) in orders.items():
            weights = multiplicity[image_index]
            fp = np.cumsum(weights * ~correct)[ends]
            tp = np.cumsum(weights * correct)[ends]
            index = np.searchsorted(fp, cap, side="right") - 1
            results[condition] = (tp[index] if index >= 0 else 0) / objects if objects else np.nan
        return results

    return recall


def write_native(run, name, table):
    ap, ids, precision, recall = per_class_ap(table.values())
    files = run / name
    files.mkdir()
    write_json(files / "metrics.json", {f"metrics/{key}(B)": float(value) for key, value in zip(
        ("mAP50", "mAP50-95", "precision", "recall"), (ap[:, 0].mean(), ap.mean(), precision.mean(), recall.mean()))})
    save_csv(files / "per_class.csv", [{"class_id": int(cid), "AP50": float(row[0]), "AP50-95": float(row.mean())}
                                     for cid, row in zip(ids, ap, strict=True)])


def save_ap(path, table):
    np.savez_compressed(path, images=np.array(list(table)),
                        prediction_counts=[len(row[1]) for row in table.values()],
                        target_counts=[len(row[3]) for row in table.values()],
                        **dict(zip(("tp", "conf", "pred_cls", "target_cls"),
                                   (np.concatenate(column) for column in zip(*table.values())))))


def write_report(run, summary, budgets, classes, rolled):
    metrics = summary["metrics"]
    lines = ["# S1-Fusion — Same-resolution complementary detector analysis", "",
             "CPU-only validation analysis for independent scientific review. F0 is A0, F1 is A1 and F2 is WBF.",
             "", "| Condition | mAP50 | mAP50–95 | Nine-class mAP50 / mAP50–95 | Fixed P / R | TP / FP / FN |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, m in metrics.items():
        lines.append(f"| {name} | {m['mAP50']:.2%} | {m['mAP50-95']:.2%} | "
                     f"{m['nine_class_AP50']:.2%} / {m['nine_class_AP50-95']:.2%} | "
                     f"{m['fixed_precision']:.2%} / {m['fixed_recall']:.2%} | {m['tp']} / {m['fp']} / {m['fn']} |")
    lines.extend(["", "## Matched FP budgets", "", "| FP/image cap | F0 TP | F1 TP | F2 TP |", "|---|---:|---:|---:|"])
    for row in budgets:
        lines.append("| " + " | ".join(str(row[key]) for key in ("fp_per_image_budget", "tp_F0", "tp_F1", "tp_F2")) + " |")
    lines.extend(["", "### Paired bootstrap", "",
                  "| Comparison | Metric | Δ (pp) | 95% interval (pp) | Valid draws |", "|---|---|---:|---:|---:|"])
    for row in summary["bootstrap"]:
        ci = "undefined" if row["ci_2.5"] is None else f"[{100*row['ci_2.5']:+.2f}, {100*row['ci_97.5']:+.2f}]"
        lines.append(f"| {row['comparison']} | {row['metric']} | {100*row['delta']:+.2f} | {ci} | {row['valid_repetitions']} |")
    lines.extend(["", "The primary 0.50 FP/image cap was applied inside each bootstrap draw. The paired image draws "
                  "describe validation sampling for these frozen checkpoints. Full-class AP excludes draws missing a class.",
                  "", "## Priority classes", "",
                  "| Class | F0 AP50 / AP50–95 | F1 AP50 / AP50–95 | F2 AP50 / AP50–95 |",
                  "|---|---:|---:|---:|"])
    for row in classes:
        if row["priority"]:
            values = " | ".join(f"{row[f'AP50_{name}']:.2%} / {row[f'AP50-95_{name}']:.2%}" for name in metrics)
            lines.append(f"| {row['class']} | {values} |")
    lines.extend(["", "Rolled pit has four objects across two images. Outcomes are descriptive.", "",
                  "| Object | F0 outcome | F1 outcome | F2 outcome |", "|---|---|---|---|"])
    for obj in rolled:
        lines.append("| " + " | ".join([obj["object_id"], *(obj[f"{name}_category"] for name in metrics)]) + " |")
    timing = summary["timing"]
    lines.extend(["", f"Median CPU WBF overhead was {timing['median_ms']:.3f} ms/image "
                  f"(p95 {timing['p95_ms']:.3f} ms). This covers only the offline fusion operation.", "",
                  "See the [object comparison](per_object_comparison.csv), [small-defect groups](small_defect_analysis.csv), "
                  "[error counts](error_breakdown.csv), [budget details](recall_fp_comparison.csv), and [examples](examples/)."])
    (run / "report.md").write_text("\n".join(lines) + "\n")


def run_fusion(run, path, conf, truth, prior, references):
    import torch

    archived, native, hashes = references
    outputs = {"F0": archived["A0"], "F1": archived["A1"], "F2": {}}
    records = {"F0": native["A0"], "F1": native["A1"]}
    for directory in ("predictions", "plots", "examples"):
        (run / directory).mkdir()
    (run / "effective_config.yaml").write_text(yaml.safe_dump(conf, sort_keys=False))
    write_json(run / "input_sha256.json", {"archive": REFERENCE_ARCHIVE, "archive_sha256": REFERENCE_SHA256,
                                           "inputs": hashes, "D1": sha256(PRIOR / "per_object.csv"),
                                           "manifests": prior["dataset_manifest_sha256"]})
    dependencies = (Path(__file__), ROOT / "scripts/evaluate_architecture.py", ROOT / "scripts/analyze_errors.py",
                    ROOT / "scripts/evaluate_fusion.py", ROOT / "scripts/evaluate_resolution.py",
                    ROOT / "scripts/evaluate.py", ROOT / "scripts/check_env.py")
    (run / "source").mkdir()
    for file in dependencies:
        shutil.copyfile(file, run / "source" / file.name)
    shutil.copyfile(path, run / "source" / "teacher_fusion.yaml")
    write_json(run / "source.json", {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                                    "upstream": UPSTREAM_COMMIT, "wbf_reference": WBF_REFERENCE,
                                    "config_sha256": sha256(path),
                                    "sha256": {str(p.relative_to(ROOT)): sha256(p) for p in dependencies}})
    write_json(run / "environment.json", {"host": platform.node(), "python": platform.python_version(),
                                          "platform": platform.platform(), "torch": torch.__version__,
                                          "numpy": np.__version__, "torch_threads": torch.get_num_threads(),
                                          "execution": "CPU only"})
    times = []
    for row in read_csv(MANIFESTS / "gc10-val.csv"):
        image = row["image"]
        start = time.perf_counter_ns()
        outputs["F2"][image] = weighted_boxes(outputs["F0"][image], outputs["F1"][image],
                                               int(row["width"]), int(row["height"]), conf)
        times.append({"image": image, "fusion_ms": (time.perf_counter_ns()-start)/1e6})
    save_csv(run / "fusion_timing.csv", times)
    timing = np.asarray([r["fusion_ms"] for r in times])
    print("Preparing native validation labels on CPU without model inference", flush=True)
    mapped = native_truth(run, truth)
    parity = {}
    for label, reference in (("F0", "A0"), ("F1", "A1")):
        rebuilt = fused_records(outputs[label], mapped, records[label])
        mismatches = sum(np.count_nonzero(rebuilt[image][0].numpy() != row[0]) for image, row in records[label].items())
        if mismatches:
            raise ValueError(f"{label} rebuilt native TP flags differ at {mismatches} prediction/IoU entries.")
        parity[label] = {"tp_flag_mismatches": int(mismatches)}
        (run / label).mkdir()
        for filename in ("metrics.json", "per_class.csv"):
            shutil.copyfile(ROOT / REFERENCE_RUN / reference / filename, run / label / filename)
    print("Both single-model TP arrays reproduced with the pinned native validator", flush=True)
    records["F2"] = fused_records(outputs["F2"], mapped, records["F0"])
    write_native(run, "F2", records["F2"])
    for label in outputs:
        write_json(run / "predictions" / f"{label}.json", outputs[label])
        save_ap(run / "predictions" / f"{label}-ap.npz", records[label])
    objects, detections, curves, changes = paired_objects(run, conf, truth, outputs, pairs=PAIRS)
    metrics, classes, sizes, small, errors, budgets = analyze(run, conf, objects, detections, curves, records, pairs=PAIRS)
    rolled = [row for row in objects if row["class"] == "rolled_pit"]
    save_csv(run / "per_object_comparison.csv", objects)
    save_csv(run / "rolled_pit_analysis.csv", rolled)
    for row in budgets:
        for label in outputs:
            row[f"tp_{label}"] = round(row[f"recall_{label}"] * len(objects))
    save_csv(run / "recall_fp_comparison.csv", budgets)
    budget_recall = budget_recall_function(truth, outputs, conf)
    point = next(row for row in budgets if row["fp_per_image_budget"] == conf["PRIMARY_FP_BUDGET"])
    for label, recall in budget_recall(np.ones(len(truth), dtype=int)).items():
        assert np.isclose(recall, point[f"recall_{label}"], atol=1e-14, rtol=0)
        metrics[label]["primary_budget_recall"] = recall
    intervals = bootstrap(run, conf, truth, objects, records, metrics, pairs=PAIRS, budget_recall=budget_recall)
    figures(run, conf, metrics, classes, sizes, small, errors, curves, objects, outputs, changes, pairs=PAIRS)
    summary = {
        "status": "completed", "run_id": conf["RUN_ID"], "images": len(truth), "objects": len(objects),
        "metrics": metrics, "bootstrap": intervals, "primary_budget": point, "budgets": budgets, "native_parity": parity,
        "paired_outcomes": {contrast(*pair): dict(Counter(row[f"outcome_{contrast(*pair)}"] for row in objects)) for pair in PAIRS},
        "single_model_overlap": dict(Counter(
            ("detected_by_both" if row["F1_category"] == "correct" else "A0_only")
            if row["F0_category"] == "correct" else
            ("A1_only" if row["F1_category"] == "correct" else "missed_by_both") for row in objects)),
        "timing": {"median_ms": float(np.median(timing)), "p95_ms": float(np.quantile(timing, .95)),
                   "mean_ms": float(timing.mean()), "total_ms": float(timing.sum())},
    }
    write_report(run, summary, budgets, classes, rolled)
    write_json(run / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/evaluation/teacher_fusion.yaml")
    args = parser.parse_args()
    conf = load_config(args.config)
    run = ROOT / "experiments" / conf["RUN_ID"]
    if run.exists() or run.is_symlink():
        parser.error(f"Run directory already exists: {run}")
    sys.path.insert(0, str(UPSTREAM))
    with TemporaryDirectory(prefix="flawtrace-fusion-settings-") as settings:
        os.environ.update(YOLO_CONFIG_DIR=settings, YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false", CUDA_VISIBLE_DEVICES="")
        import torch

        torch.set_num_threads(2)
        prior, truth = frozen_inputs()
        references = reference_predictions(conf, truth, prior)
        run.mkdir(parents=True, exist_ok=False)
        with (run / "terminal.log").open("x", buffering=1) as log:
            with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                write_json(run / "summary.json", {"status": "started"})
                try:
                    run_fusion(run, args.config.resolve(), conf, truth, prior, references)
                except (Exception, KeyboardInterrupt):
                    traceback.print_exc()
                    write_json(run / "summary.json", {"status": "failed"})
                    raise SystemExit(1) from None


if __name__ == "__main__":
    main()
