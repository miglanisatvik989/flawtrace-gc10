"""Compare cross-resolution fusion on frozen E1 validation predictions, on CPU."""

import argparse
import json
import os
import platform
import random
import re
import sys
import time
import traceback
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from tempfile import TemporaryDirectory
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from analyze_errors import CATEGORIES, box, box_iou, match, ratio, read_csv, save_csv
from evaluate import DATASET, ROOT, UPSTREAM, UPSTREAM_COMMIT, Tee, sha256, write_json
from evaluate_resolution import GEOMETRY, MATCH_SETTINGS, frozen_inputs, native_ap, read_json, score_curve

CONDITIONS = ("F0", "F1", "F2", "F3")
FOCUS = ("inclusion", "silk_spot", "oil_spot", "crease", "rolled_pit")


def retained(first, second, condition):
    if condition == "F0":
        return first
    if condition == "F1":
        return second
    if condition == "F2":
        pool = sorted(first + second, key=lambda p: -p["score"])
        keep = []
        cutoff = .7
    else:
        pool = sorted(second, key=lambda p: -p["score"])
        keep = list(first)
        cutoff = .5
    overlaps = box_iou([box(p) for p in pool], [box(p) for p in keep + pool])
    eligible = np.array([p["class_id"] for p in pool], dtype=int)[:, None] == np.array(
        [p["class_id"] for p in keep + pool], dtype=int)
    blocked = (overlaps > cutoff if condition == "F2" else overlaps >= cutoff) & eligible
    indices = list(range(len(keep)))
    offset = len(keep)
    for index, prediction in enumerate(pool):
        if len(keep) == 300:
            break
        if not blocked[index, indices].any():
            keep.append(prediction)
            indices.append(offset + index)
    return keep


def read_inputs(input_run, truth):
    predictions, records = {}, {}
    for size in (640, 960):
        predictions[size] = {
            image: [dict(p, source_resolution=size, prediction_id=f"{image}:{size}:{i}")
                    for i, p in enumerate(rows)]
            for image, rows in read_json(input_run / f"predictions/{size}.json").items()
        }
        with np.load(input_run / f"predictions/{size}-ap.npz") as ap:
            images = ap["images"].tolist()
            pred_edges = np.r_[0, ap["prediction_counts"].cumsum()]
            target_edges = np.r_[0, ap["target_counts"].cumsum()]
            if set(images) != truth.keys() or len(images) != len(truth) or predictions[size].keys() != truth.keys():
                raise ValueError(f"Unexpected image membership at {size}px.")
            records[size] = {}
            for i, image in enumerate(images):
                sl = slice(pred_edges[i], pred_edges[i + 1])
                ts = slice(target_edges[i], target_edges[i + 1])
                records[size][image] = tuple(ap[key][sl] for key in ("tp", "conf", "pred_cls")) + (ap["target_cls"][ts],)
                rows = predictions[size][image]
                if (len(rows) > 300 or [p["score"] for p in rows] != ap["conf"][sl].tolist()
                        or [p["class_id"] for p in rows] != ap["pred_cls"][sl].tolist()
                        or Counter(ap["target_cls"][ts].tolist()) != Counter(o["class_id"] for o in truth[image])):
                    raise ValueError(f"Prediction or target records differ for {image} at {size}px.")
    return predictions, records


def combined(records):
    return [np.concatenate(col) for col in zip(*records.values())]


def fused_records(predictions, truth, order):
    import torch
    from ultralytics.engine.validator import BaseValidator
    from ultralytics.utils.metrics import box_iou as native_iou

    context = SimpleNamespace(iouv=torch.linspace(.5, .95, 10, device="cpu"))
    records = {}
    for image in order:
        targets, guesses = truth[image], predictions[image]
        gt_classes = torch.tensor([o["class_id"] for o in targets], device="cpu")
        pred_classes = torch.tensor([p["class_id"] for p in guesses], device="cpu")
        gt_boxes = torch.tensor([box(o) for o in targets], device="cpu").reshape(-1, 4)
        pred_boxes = torch.tensor([box(p) for p in guesses], device="cpu").reshape(-1, 4)
        tp = BaseValidator.match_predictions(context, pred_classes, gt_classes, native_iou(gt_boxes, pred_boxes))
        records[image] = (tp, np.asarray([p["score"] for p in guesses], dtype=np.float32),
                          pred_classes.numpy(), gt_classes.numpy())
    return records


def bootstrap(run, conf, truth, paired, records, metrics):
    images = list(truth)
    positions = {image: i for i, image in enumerate(images)}
    # Columns are total objects, Q1 objects, and correct counts per condition.
    counts = np.zeros((len(images), 2 + 2 * len(CONDITIONS)), dtype=int)
    for obj in paired:
        row = counts[positions[obj["image"]]]
        q1 = obj["short_side_640_group"] == "Q1"
        row[:2] += [1, q1]
        for index, condition in enumerate(CONDITIONS):
            hit = obj[f"{condition}_category"] == "correct"
            row[2 + 2*index:4 + 2*index] += [hit, q1 and hit]
    ordered = {c: [(positions[image], record) for image, record in records[c].items()]
               for c in ("F0", "F2", "F3")}
    reps = conf["BOOTSTRAP_REPETITIONS"]
    rng = np.random.default_rng(conf["SEED"])
    changes = {c: np.empty((reps, 4)) for c in ("F2", "F3")}
    for iteration in range(reps):
        multiplicity = np.bincount(rng.integers(len(images), size=len(images)), minlength=len(images))
        aps = {c: native_ap([np.concatenate(col) for col in zip(*(
            record for index, record in ordered[c] for _ in range(multiplicity[index])))]) for c in ordered}
        totals = multiplicity @ counts
        for c, index in (("F2", 6), ("F3", 8)):
            changes[c][iteration] = [*(aps[c] - aps["F0"]), (totals[index] - totals[2]) / totals[0],
                                      (totals[index + 1] - totals[3]) / totals[1] if totals[1] else np.nan]
        if (iteration + 1) % 500 == 0:
            print(f"Paired bootstrap {iteration + 1}/{reps}", flush=True)
    rows = []
    for c, values in changes.items():
        np.save(run / "predictions" / f"{c}-bootstrap.npy", values)
        for index, key in enumerate(("mAP50", "mAP50-95", "fixed_recall", "q1_recall")):
            valid = values[np.isfinite(values[:, index]), index]
            low, high = np.quantile(valid, [.025, .975]) if len(valid) else (None, None)
            rows.append({"condition": c, "metric": key, "delta": metrics[c][key] - metrics["F0"][key],
                         "ci_2.5": low, "ci_97.5": high, "valid_repetitions": len(valid)})
    return rows


def evaluate_conditions(run, conf, truth, predictions, native, reference):
    from ultralytics.utils.metrics import ap_per_class

    metrics, errors, groups, classes, sweeps, budget_rows, timing = {}, [], [], [], {}, [], []
    all_outputs, assignments, records = {}, {}, {}
    paired = {o["object_id"]: dict(o) for rows in truth.values() for o in rows}
    prior_aps = {f"F{i}": reference["metrics"][str(size)] for i, size in enumerate((640, 960))}
    budgets = (.10, .25, .50, 1.00)
    for condition in CONDITIONS:
        print(f"Evaluating {condition} on CPU", flush=True)
        output, elapsed = {}, []
        for image in truth:
            start = time.perf_counter_ns()
            output[image] = retained(predictions[640][image], predictions[960][image], condition)
            elapsed.append((time.perf_counter_ns() - start) / 1e6)
        all_outputs[condition] = output
        timing.append({"condition": condition, "images": len(elapsed), "median_cpu_ms": np.median(elapsed),
                       "p95_cpu_ms": np.quantile(elapsed, .95)})
        write_json(run / "predictions" / f"{condition}.json", output)
        if condition in prior_aps:
            records[condition] = native[640 if condition == "F0" else 960]
        else:
            records[condition] = fused_records(output, truth, native[640])
        _, _, precision, recall, _, ap, class_ids, *_ = ap_per_class(*combined(records[condition]), plot=False)
        metrics[condition] = {"mAP50": ap[:, 0].mean(), "mAP50-95": ap.mean(),
                              "native_precision": precision.mean(), "native_recall": recall.mean()}
        if condition in prior_aps and any(
            abs(metrics[condition][key] - prior_aps[condition][key]) > 1e-8 for key in ("mAP50", "mAP50-95")
        ):
            raise ValueError(f"{condition} AP parity with E1 failed.")

        detections, floor = [], []
        for image, targets in truth.items():
            outcome, matched = match(targets, output[image], *MATCH_SETTINGS)
            detections.extend(dict(p, image=image) for p in matched)
            floor.extend(match(targets, output[image], .001, *MATCH_SETTINGS[1:])[1])
            for obj, result in zip(targets, outcome):
                paired[obj["object_id"]][f"{condition}_correct_prediction"] = None
                paired[obj["object_id"]].update({f"{condition}_{key}": value for key, value in result.items()})
            for p in matched:
                if p["category"] == "correct":
                    paired[p["object_id"]][f"{condition}_correct_prediction"] = p["prediction_id"]
        assignments[condition] = detections
        write_json(run / "predictions" / f"{condition}-fixed-point.json", detections)
        objects = list(paired.values())
        counts = Counter(d["category"] for d in detections)
        counts["missed"] = sum(o[f"{condition}_category"] == "missed" for o in objects)
        tp = counts["correct"]
        q1 = [o for o in objects if o["short_side_640_group"] == "Q1"]
        metrics[condition].update(tp=tp, fp=len(detections)-tp, fn=len(objects)-tp,
                                  fixed_precision=ratio(tp,len(detections)), fixed_recall=tp/len(objects),
                                  q1_recall=sum(o[f"{condition}_category"] == "correct" for o in q1)/len(q1))
        errors.extend({"condition": condition, "category": key, "count": counts[key]} for key in CATEGORIES)
        for field in GEOMETRY:
            for quartile in ("Q1", "Q2", "Q3", "Q4"):
                subset = [o for o in objects if o[f"{field}_group"] == quartile]
                groups.append({"condition": condition, "measure": field, "group": quartile, "objects": len(subset),
                               "images": len({o["image"] for o in subset}),
                               "recall": ratio(sum(o[f"{condition}_category"] == "correct" for o in subset),len(subset)),
                               "localization": sum(o[f"{condition}_category"] == "localization" for o in subset)})
        for index, cls in enumerate(class_ids):
            subset = [o for o in objects if o["class_id"] == cls]
            classes.append({"condition": condition, "class": subset[0]["class"], "objects": len(subset),
                            "images": len({o["image"] for o in subset}), "AP50": ap[index,0],
                            "AP50-95": ap[index].mean(),
                            "fixed_recall": sum(o[f"{condition}_category"] == "correct" for o in subset)/len(subset)})
        sweeps[condition] = score_curve(floor, len(objects), len(truth))
        for budget in budgets:
            best = max((p for p in sweeps[condition] if p["fp_per_image"] <= budget),
                       key=lambda row: (row["recall"], -row["fp_per_image"]))
            budget_rows.append({"condition": condition, "fp_per_image_budget": budget, **best,
                                "budget_within_sweep": budget <= sweeps[condition][-1]["fp_per_image"]})
    return metrics, errors, groups, classes, sweeps, budget_rows, timing, paired, all_outputs, assignments, records


def compare_objects(paired, outputs, assignments):
    summary = {}
    f0_fp = {p["prediction_id"] for p in assignments["F0"] if p["category"] != "correct"}
    for condition in CONDITIONS[1:]:
        kept = {p["prediction_id"] for rows in outputs[condition].values() for p in rows}
        added_fp = {p["prediction_id"] for p in assignments[condition] if p["category"] != "correct"} - f0_fp
        outcomes = Counter()
        for obj in paired.values():
            baseline, candidate = (obj[f"{c}_category"] == "correct" for c in ("F0", condition))
            status = {(True, True): "retained", (True, False): "lost",
                      (False, True): "recovered", (False, False): "both_missed"}[baseline, candidate]
            obj[f"{condition}_vs_F0"] = status
            outcomes[status] += 1
            complementary = obj["F0_category"] != "correct" and obj["F1_category"] == "correct"
            obj[f"{condition}_complementary_960_retained"] = (
                obj["F1_correct_prediction"] in kept if complementary else None)
        summary[condition] = {key: outcomes[key] for key in ("retained", "lost", "recovered", "both_missed")}
        summary[condition]["new_false_positives"] = len(added_fp)
        summary[condition]["complementary_960_retained"] = sum(
            obj[f"{condition}_complementary_960_retained"] is True for obj in paired.values())
    return summary


def figures(run, metrics, classes, groups, errors, curves, paired, outputs, assignments, seed):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    colors = ("#2166ac", "#b35806", "#4d9221", "#762a83")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, path):
        for ext in ("png", "pdf"):
            fig.savefig(path.with_suffix(f".{ext}"), dpi=250, bbox_inches="tight")
        plt.close(fig)

    def bars(name, labels, values, ylabel, limit=None):
        fig, ax = plt.subplots(figsize=(max(6, len(labels)), 4))
        x = np.arange(len(labels))
        for i, c in enumerate(CONDITIONS):
            ax.bar(x + .19*(i-1.5), values[c], .19, color=colors[i], label=c)
        ax.set_xticks(x, [label.replace("_", " ") for label in labels], rotation=28, ha="right")
        ax.set_ylabel(ylabel)
        if limit:
            ax.set_ylim(0, limit)
        ax.legend(frameon=False)
        save(fig, run / "plots" / name)

    keys = ("mAP50", "mAP50-95", "fixed_precision", "fixed_recall")
    bars("accuracy", keys, {c: [metrics[c][k] for k in keys] for c in CONDITIONS}, "Fraction", 1)
    labels = [row["class"] for row in classes if row["condition"] == "F0"]
    for key in ("AP50", "AP50-95"):
        bars(f"class_{key}", labels, {c: [r[key] for r in classes if r["condition"] == c] for c in CONDITIONS}, key, 1)
    for field in GEOMETRY:
        bars(f"recall_{field}", ("Q1","Q2","Q3","Q4"),
             {c: [r["recall"] for r in groups if r["condition"] == c and r["measure"] == field] for c in CONDITIONS},
             "Fixed-point recall", 1)
    bars("errors", CATEGORIES, {c: [r["count"] for r in errors if r["condition"] == c] for c in CONDITIONS}, "Assignments")
    fig, ax = plt.subplots(figsize=(6, 4))
    for c, color in zip(CONDITIONS, colors):
        ax.step([r["fp_per_image"] for r in curves[c]], [r["recall"] for r in curves[c]], where="post", color=color, label=c)
    ax.set(xlabel="False positives per image", ylabel="Recall", xlim=(0, 1), ylim=(0, 1))
    ax.legend(frameon=False)
    save(fig, run / "plots" / "recall_fp")

    rng = random.Random(seed)
    for c in ("F2", "F3"):
        original_fp = {p["prediction_id"] for p in assignments["F0"] if p["category"] != "correct"}
        candidates = {"recovered": [o for o in paired.values() if o[f"{c}_vs_F0"] == "recovered"],
                      "lost": [o for o in paired.values() if o[f"{c}_vs_F0"] == "lost"],
                      "new_fp": [p for p in assignments[c] if p["category"] != "correct"
                                 and p["prediction_id"] not in original_fp]}
        for outcome, rows in candidates.items():
            for i, highlight in enumerate(rng.sample(rows, min(2, len(rows))), 1):
                image = highlight["image"]
                with Image.open(DATASET / image) as original:
                    pixels = original.convert("RGB")
                fig, axes = plt.subplots(2, 1, figsize=(9, 7), layout="constrained")
                for ax, method in zip(axes, ("F0", c)):
                    ax.imshow(pixels)
                    for row in outputs[method][image]:
                        if row["score"] >= .25:
                            x, y, x2, y2 = box(row)
                            color = "#ffb34a" if row["source_resolution"] == 960 else "#42cef5"
                            ax.add_patch(Rectangle((x, y), x2-x, y2-y, fill=False, edgecolor=color))
                    x, y, x2, y2 = box(highlight)
                    ax.add_patch(Rectangle((x, y), x2-x, y2-y, fill=False, edgecolor="#55efbf", linewidth=2))
                    ax.set_title(method)
                    ax.set_axis_off()
                fig.suptitle(f"{outcome.replace('_',' ').title()} · {Path(image).name}")
                save(fig, run / "examples" / f"{c}-{outcome}-{i}")


def write_report(run, metrics, comparisons, intervals, oracle, classes, timing):
    lines = ["# E1A offline fusion · pending scientific review", "",
             "| Condition | mAP50 | mAP50–95 | Fixed precision | Fixed recall | TP / FP / FN |",
             "|---|---:|---:|---:|---:|---:|"]
    for c, row in metrics.items():
        lines.append(f"| {c} | {row['mAP50']:.2%} | {row['mAP50-95']:.2%} | {row['fixed_precision']:.2%} | "
                     f"{row['fixed_recall']:.2%} | {row['tp']} / {row['fp']} / {row['fn']} |")
    lines.extend(["", f"Oracle union detected {oracle['union_tp']} of {oracle['objects']} objects ({oracle['recall']:.2%}). "
                  "This retrospective diagnostic uses ground truth and is excluded from AP rankings.", ""])
    for c in ("F2", "F3"):
        row = comparisons[c]
        lines.append(f"- {c} recovered {row['recovered']}, lost {row['lost']} and retained {row['retained']} "
                     f"baseline detections. New fixed-point false positives: {row['new_false_positives']}. "
                     f"Complementary 960 prediction IDs retained: {row['complementary_960_retained']} of 17.")
    lines.extend(["", "| Condition | Metric | Change (points) | Paired 95% interval (points) |",
                  "|---|---|---:|---:|"])
    for row in intervals:
        lines.append(f"| {row['condition']} | {row['metric']} | {100*row['delta']:+.2f} | "
                     f"{100*row['ci_2.5']:+.2f} to {100*row['ci_97.5']:+.2f} |")
    lines.extend(["", "## Priority classes", ""])
    for name in FOCUS:
        selected = [r for r in classes if r["class"] == name]
        lines.append(f"- {name.replace('_',' ').title()} ({selected[0]['objects']} objects). Fixed recall "
                     + ", ".join(f"{r['condition']} {r['fixed_recall']:.1%}" for r in selected) + ".")
    lines.extend(["", "Rolled pit has four objects across two images. Its results are descriptive.", "",
                  "## Latency", "",
                  "| Condition | Median CPU fusion (ms) | p95 CPU fusion (ms) | Estimated combined latency (ms) |",
                  "|---|---:|---:|---:|"])
    for row in timing:
        lines.append(f"| {row['condition']} | {row['median_cpu_ms']:.3f} | {row['p95_cpu_ms']:.3f} | "
                     f"{row['estimated_total_ms']:.2f} |")
    lines.extend(["", "GPU cost uses the average of E1’s two batch-1 end-to-end medians per resolution. "
                  "The combined estimate adds the required view costs and median CPU overhead. Live end-to-end fusion latency remains unmeasured.",
                  "", "The paired intervals describe validation-image uncertainty for one checkpoint. "
                  "FP/image comparisons use shared upper budgets with discrete confidence thresholds."])
    (run / "report.md").write_text("\n".join(lines) + "\n")


def run_experiment(run, conf, input_run, predictions, native, reference, truth, prior):
    metrics, errors, groups, classes, curves, budgets, timing, paired, outputs, assignments, records = (
        evaluate_conditions(run, conf, truth, predictions, native, reference))
    comparisons = compare_objects(paired, outputs, assignments)
    outcomes = comparisons["F1"]
    oracle = {"control_tp": metrics["F0"]["tp"], "complementary_tp": outcomes["recovered"],
              "union_tp": metrics["F0"]["tp"] + outcomes["recovered"], "objects": len(paired)}
    oracle["recall"] = oracle["union_tp"] / oracle["objects"]
    if (oracle["control_tp"], oracle["complementary_tp"], oracle["union_tp"], oracle["objects"]) != (235, 17, 252, 345):
        raise ValueError(f"Oracle differs from accepted E1 outcomes: {oracle}")
    write_json(run / "oracle_analysis.json", oracle)
    write_json(run / "predictions" / "recall_fp_curves.json", curves)
    stage_times = read_csv(input_run / "timing_comparison.csv")
    estimates = {size: np.mean([float(r["batch1_median_ms"]) for r in stage_times
                               if int(r["resolution"]) == size and int(r["batch"]) == 1]) for size in (640,960)}
    for row in timing:
        c = row["condition"]
        row["estimated_view_ms"] = estimates[640] if c == "F0" else estimates[960] if c == "F1" else sum(estimates.values())
        row["estimated_total_ms"] = row["estimated_view_ms"] + row["median_cpu_ms"]
    for rows, metrics_keys, group_keys in ((classes, ("AP50","AP50-95","fixed_recall"), ("class",)),
                                           (groups, ("recall","localization"), ("measure","group"))):
        control = {tuple(row[k] for k in group_keys): row for row in rows if row["condition"] == "F0"}
        for row in rows:
            for key in metrics_keys:
                base = control[tuple(row[k] for k in group_keys)][key]
                row[f"delta_{key}_vs_F0"] = row[key] - base if row[key] is not None and base is not None else None
    for row in errors:
        row["delta_vs_F0"] = row["count"] - next(r["count"] for r in errors
                                               if r["condition"] == "F0" and r["category"] == row["category"])
    for name, rows in (("metrics_comparison", [{"condition": c, **metrics[c]} for c in CONDITIONS]),
                       ("per_class_comparison", classes), ("per_object_comparison", list(paired.values())),
                       ("size_analysis", groups), ("error_breakdown", errors), ("recall_fp_comparison", budgets),
                       ("fusion_timing", timing)):
        save_csv(run / f"{name}.csv", rows)
    intervals = bootstrap(run, conf, truth, paired.values(), records, metrics)
    save_csv(run / "bootstrap_results.csv", intervals)
    figures(run, metrics, classes, groups, errors, curves, paired, outputs, assignments, conf["SEED"])
    write_report(run, metrics, comparisons, intervals, oracle, classes, timing)
    write_json(run / "summary.json", {
        "status": "completed", "device": "cpu", "split": "val", "images": len(truth), "objects": len(paired),
        "upstream_commit": UPSTREAM_COMMIT, "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(), "numpy": np.__version__,
        "checkpoint_sha256": prior["checkpoint_sha256"]["B1"], "dataset_manifest_sha256": prior["dataset_manifest_sha256"],
        "metrics": metrics, "E1_control_ap_parity": True, "comparisons_vs_F0": comparisons,
        "oracle": oracle, "bootstrap": intervals, "timing": timing,
        "rolled_pit": [row for row in paired.values() if row["class"] == "rolled_pit"],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/fusion.yaml")
    args = parser.parse_args()
    conf = yaml.safe_load(args.config.read_text())
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", conf["RUN_ID"]):
        parser.error("RUN_ID must be a single directory name.")
    if conf["BOOTSTRAP_REPETITIONS"] != 5000 or conf["SEED"] != 42:
        parser.error("Use 5,000 paired repetitions and seed 42.")
    run = ROOT / "experiments" / conf["RUN_ID"]
    if run.exists() or run.is_symlink():
        parser.error(f"Choose an unused run ID. Already exists: {run}")
    prior, truth = frozen_inputs()
    input_run = ROOT / conf["INPUT_RUN"]
    reference = read_json(input_run / "summary.json")
    if (reference["status"] != "completed" or not reference["control_ap_parity_passed"]
            or reference["checkpoint_sha256"] != prior["checkpoint_sha256"]["B1"]
            or reference["dataset_manifest_sha256"] != prior["dataset_manifest_sha256"]
            or reference["d1_objects_sha256"] != sha256(ROOT / "experiments/d1-baseline-01/per_object.csv")):
        raise ValueError("E1 inputs do not match the frozen baseline or D1 identities.")
    predictions, native = read_inputs(input_run, truth)
    source = read_json(ROOT / ".cache/source.json")
    for name in ("scripts/evaluate_fusion.py", "scripts/evaluate_resolution.py", "scripts/analyze_errors.py",
                 str(args.config.resolve().relative_to(ROOT))):
        if sha256(ROOT / name) != source["sha256"].get(name):
            raise ValueError("Run scripts/record_source.py before execution.")
    if source["upstream_commit"] != UPSTREAM_COMMIT:
        raise ValueError("Expected provenance for the pinned upstream.")
    run.mkdir(exist_ok=False)
    for directory in ("plots", "examples", "predictions"):
        (run / directory).mkdir()
    hashes = {str(path.relative_to(input_run)): sha256(path) for path in (
        input_run / "summary.json", input_run / "timing_comparison.csv",
        *(input_run / "predictions" / f"{size}{suffix}" for size in (640,960) for suffix in (".json","-ap.npz")))}
    write_json(run / "source.json", source)
    write_json(run / "input_sha256.json", hashes)
    (run / "effective_config.yaml").write_text(yaml.safe_dump({
        **conf, "DEVICE": "cpu", "SPLIT": "val", "AP_FLOOR": .001, "FIXED_CONFIDENCE": .25,
        "MATCH_IOU": .5, "BACKGROUND_IOU": .1, "F2_NMS_IOU": .7, "F3_BLOCK_IOU": .5,
        "MAX_DETECTIONS": 300, "FP_PER_IMAGE_BUDGETS": [.1,.25,.5,1.],
    }, sort_keys=False))
    os.environ.update(CUDA_VISIBLE_DEVICES="", YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false")
    sys.path.insert(0, str(UPSTREAM))
    sys.dont_write_bytecode = True
    with (run / "terminal.log").open("x", buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            with TemporaryDirectory(prefix="settings-", dir=run) as settings:
                os.environ["YOLO_CONFIG_DIR"] = settings
                try:
                    run_experiment(run, conf, input_run, predictions, native, reference, truth, prior)
                except Exception:
                    traceback.print_exc()
                    raise SystemExit(1) from None
            print(f"E1A artifacts: {run}")


if __name__ == "__main__":
    main()
