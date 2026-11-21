"""Evaluate the completed E2 checkpoint and compare all four B1/E2 conditions."""

import argparse
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

from analyze_errors import CATEGORIES, box, box_iou, match, ratio, save_csv
from evaluate import DATASET, ROOT, UPSTREAM_COMMIT, Tee, sha256, write_json
from evaluate_fusion import combined, read_inputs
from evaluate_resolution import (
    GEOMETRY, MATCH_SETTINGS, PRIOR, export_validation, frozen_inputs, measure_timing, native_ap,
    read_json,
)

CONDITIONS = {"A": (640, 640), "B": (640, 960), "C": (960, 640), "D": (960, 960)}
COMPARISONS = (("D", "A"), ("D", "B"), ("C", "A"), ("D", "C"))
E2_CHECKPOINT = "experiments/e2-train-960-b16-02/weights/best.pt"
E2_SHA256 = "ed58d685cdc5b2cd5d3f6db800eb5101ecc73b0a960bc7235ac77ddb1be69f33"
FOCUS = ("inclusion", "silk_spot", "oil_spot", "crease", "welding_line", "rolled_pit")


def verify_inputs(conf):
    prior, truth = frozen_inputs()
    checkpoint = (ROOT / conf["CHECKPOINT"]).resolve(strict=True)
    if checkpoint != (ROOT / E2_CHECKPOINT).resolve() or sha256(checkpoint) != E2_SHA256:
        raise ValueError("Expected the frozen E2 best.pt.")
    trained = checkpoint.parents[1]
    attempt, = trained.glob("attempt-*.json")
    attempt = read_json(attempt)
    if (read_json(trained / "run.json")["status"] != "completed" or attempt["status"] != "completed"
            or attempt["checkpoint_sha256"]["best.pt"] != E2_SHA256
            or attempt["train_manifest_sha256"] != sha256(ROOT / "data/gc10-det-aug42-x4/train.csv")
            or attempt["augmentation"]["source_manifest_sha256"]["val"] != prior["dataset_manifest_sha256"]["val"]):
        raise ValueError("E2 training or dataset provenance differs.")
    b1_run = ROOT / conf["B1_RUN"]
    b1 = read_json(b1_run / "summary.json")
    if (b1["status"] != "completed" or b1["dataset_manifest_sha256"] != prior["dataset_manifest_sha256"]
            or b1["checkpoint_sha256"] != prior["checkpoint_sha256"]["B1"]
            or b1["d1_objects_sha256"] != sha256(PRIOR / "per_object.csv")
            or b1["upstream_commit"] != UPSTREAM_COMMIT or not b1["control_ap_parity_passed"]):
        raise ValueError("E1 predictions differ from frozen B1 or D1 inputs.")
    paths = [b1_run / "summary.json", b1_run / "timing_comparison.csv", b1_run / "timing_device.json",
             *(b1_run / "predictions" / f"{size}{ext}" for size in (640, 960) for ext in (".json", "-ap.npz"))]
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in paths}
    reviewed = read_json(ROOT / "experiments/e1a-fusion-01/input_sha256.json")
    for name in ("summary.json", "timing_comparison.csv",
                 *(f"predictions/{size}{ext}" for size in (640, 960) for ext in (".json", "-ap.npz"))):
        if hashes[str((b1_run / name).relative_to(ROOT))] != reviewed[name]:
            raise ValueError(f"Reused E1 artifact differs from reviewed input: {name}")
    for size, expected in (("640", ("69.5288", "34.1224")), ("960", ("41.0997", "17.2407"))):
        if tuple(f"{100*b1['metrics'][size][key]:.4f}" for key in ("mAP50", "mAP50-95")) != expected:
            raise ValueError(f"Unexpected archived B1 AP at {size}px.")
    guesses, records = read_inputs(b1_run, truth)
    return prior, truth, checkpoint, b1, guesses, records, hashes


def coverage(obj, predictions):
    candidates = [p for p in predictions if p["class_id"] == obj["class_id"] and p["score"] >= MATCH_SETTINGS[0]]
    if not candidates:
        return (None,) * 4
    actual = np.asarray(box(obj))
    ious = box_iou([actual], [box(p) for p in candidates])[0]
    index = int(ious.argmax())
    pred = np.asarray(box(candidates[index]))
    pred_area = float(np.maximum(0, pred[2:] - pred[:2]).prod())
    true_area = float((actual[2:] - actual[:2]).prod())
    overlap = float(np.maximum(0, np.minimum(pred[2:], actual[2:]) - np.maximum(pred[:2], actual[:2])).prod())
    return float(ious[index]), overlap / true_area, pred_area / true_area, ratio(overlap, pred_area)


def diagnostics(run, conf, truth, predictions, records, expected):
    from ultralytics.utils.metrics import ap_per_class
    from evaluate_resolution import score_curve

    metrics, classes, errors, sizes, curves, budgets = {}, [], [], [], {}, []
    objects = {obj["object_id"]: dict(obj) for rows in truth.values() for obj in rows}
    for label, (train_size, inference) in CONDITIONS.items():
        _, _, p, r, _, ap, class_ids, *_ = ap_per_class(*combined(records[label]), plot=False)
        values = {"mAP50": ap[:, 0].mean(), "mAP50-95": ap.mean(), "native_precision": p.mean(),
                  "native_recall": r.mean()}
        if any(abs(values[key] - expected[label][key]) > 1e-8 for key in ("mAP50", "mAP50-95")):
            raise ValueError(f"{label} AP records failed native parity.")
        detections, floor = [], []
        for image, targets in truth.items():
            guesses = predictions[label][image]
            matched, selected = match(targets, guesses, *MATCH_SETTINGS)
            detections.extend(dict(row, image=image) for row in selected)
            floor.extend(match(targets, guesses, .001, *MATCH_SETTINGS[1:])[1])
            for target, result in zip(targets, matched):
                obj = objects[target["object_id"]]
                obj.update({f"{label}_{key}": value for key, value in result.items()})
                if result["category"] == "localization":
                    measurements = coverage(obj, guesses)
                else:
                    measurements = (None,) * 4
                for key, value in zip(("best_same_class_iou", "gt_coverage", "area_ratio", "containment"), measurements):
                    obj[f"{label}_{key}"] = value
        counts = Counter(row["category"] for row in detections)
        def is_hit(obj):
            return obj[f"{label}_category"] == "correct"

        counts["missed"] = sum(obj[f"{label}_category"] == "missed" for obj in objects.values())
        tp = counts["correct"]
        q1 = [obj for obj in objects.values() if obj["short_side_640_group"] == "Q1"]
        metrics[label] = {"training_size": train_size, "inference_size": inference, **values,
                          "tp": tp, "fp": len(detections) - tp, "fn": len(objects) - tp,
                          "fixed_precision": ratio(tp, len(detections)), "fixed_recall": tp / len(objects),
                          "q1_recall": sum(map(is_hit, q1)) / len(q1)}
        for i, cls in enumerate(class_ids):
            group = [obj for obj in objects.values() if obj["class_id"] == cls]
            classes.append({"condition": label, "class": group[0]["class"], "objects": len(group),
                            "images": len({o["image"] for o in group}), "AP50": ap[i, 0], "AP50-95": ap[i].mean(),
                            "fixed_recall": sum(map(is_hit, group)) / len(group)})
        for field in GEOMETRY:
            for quartile in ("Q1", "Q2", "Q3", "Q4"):
                group = [obj for obj in objects.values() if obj[f"{field}_group"] == quartile]
                sizes.append({"condition": label, "measure": field, "group": quartile, "objects": len(group),
                              "images": len({o["image"] for o in group}), "recall": ratio(sum(map(is_hit, group)), len(group)),
                              "localization": sum(obj[f"{label}_category"] == "localization" for obj in group)})
        errors.extend({"condition": label, "category": name, "count": counts[name]} for name in CATEGORIES)
        curves[label] = score_curve(floor, len(objects), len(truth))
        for budget in conf["FP_PER_IMAGE_BUDGETS"]:
            best = max((p for p in curves[label] if p["fp_per_image"] <= budget),
                       key=lambda p: (p["recall"], -p["fp_per_image"]))
            budgets.append({"condition": label, "fp_per_image_budget": budget, **best,
                            "budget_within_sweep": budget <= curves[label][-1]["fp_per_image"]})
        write_json(run / "predictions" / f"{label}-fixed-point.json", detections)
    for obj in objects.values():
        for c in CONDITIONS:
            a, b = obj["A_category"] == "correct", obj[f"{c}_category"] == "correct"
            obj[f"{c}_vs_A"] = {(True, True): "retained", (True, False): "lost",
                                (False, True): "recovered", (False, False): "both_missed"}[a, b]
        obj["missed_all"] = all(obj[f"{c}_category"] != "correct" for c in CONDITIONS)
    for table, identity, keys in ((classes, ("class",), ("AP50", "AP50-95", "fixed_recall")),
                                  (sizes, ("measure", "group"), ("recall", "localization"))):
        base = {tuple(r[k] for k in identity): r for r in table if r["condition"] == "A"}
        for row in table:
            for key in keys:
                a = base[tuple(row[k] for k in identity)][key]
                row[f"delta_{key}_vs_A"] = row[key] - a if a is not None and row[key] is not None else None
    for name, rows in (("metrics_comparison", [{"condition": c, **r} for c, r in metrics.items()]),
                       ("per_class_comparison", classes), ("per_object_comparison", list(objects.values())),
                       ("size_analysis", sizes), ("error_breakdown", errors), ("recall_fp_comparison", budgets)):
        save_csv(run / f"{name}.csv", rows)
    write_json(run / "predictions" / "recall_fp_curves.json", curves)
    return metrics, classes, errors, sizes, curves, list(objects.values())


def paired_bootstrap(run, conf, truth, objects, records, metrics):
    keys = ("mAP50", "mAP50-95", "fixed_recall", "q1_recall")
    indices = {image: i for i, image in enumerate(truth)}
    counts = np.zeros((len(truth), 2 + 2 * len(CONDITIONS)), dtype=int)
    for obj in objects:
        row = counts[indices[obj["image"]]]
        q1 = obj["short_side_640_group"] == "Q1"
        row[:2] += [1, q1]
        for i, c in enumerate(CONDITIONS):
            hit = obj[f"{c}_category"] == "correct"
            row[2 + 2*i:4 + 2*i] += [hit, q1 and hit]
    orders = {c: [(indices[image], row) for image, row in table.items()] for c, table in records.items()}
    rng = np.random.default_rng(conf["SEED"])
    reps = conf["BOOTSTRAP_REPETITIONS"]
    changes = {f"{c}-{base}": np.empty((reps, 4)) for c, base in COMPARISONS}
    for i in range(reps):
        multiplicity = np.bincount(rng.integers(len(truth), size=len(truth)), minlength=len(truth))
        total, q1_total, *hits = multiplicity @ counts
        values = {}
        for index, c in enumerate(CONDITIONS):
            selected = [record for number, record in orders[c] for _ in range(multiplicity[number])]
            ap = native_ap([np.concatenate(col) for col in zip(*selected)])
            values[c] = np.array([*ap, hits[2*index] / total,
                                  hits[2*index+1] / q1_total if q1_total else np.nan])
        for c, base in COMPARISONS:
            changes[f"{c}-{base}"][i] = values[c] - values[base]
        if (i + 1) % 500 == 0:
            print(f"Paired bootstrap {i + 1}/{reps}", flush=True)
    rows = []
    for c, base in COMPARISONS:
        comparison = f"{c}-{base}"
        array = changes[comparison]
        np.save(run / "predictions" / f"{comparison}-bootstrap.npy", array)
        for key, draws in zip(keys, array.T):
            valid = draws[np.isfinite(draws)]
            low, high = np.quantile(valid, [.025, .975]) if len(valid) else (None, None)
            rows.append({"comparison": comparison, "metric": key, "delta": metrics[c][key] - metrics[base][key],
                         "ci_2.5": low, "ci_97.5": high, "valid_repetitions": len(valid)})
    save_csv(run / "bootstrap_results.csv", rows)
    return rows


def figures(run, conf, metrics, classes, errors, sizes, curves, objects, predictions):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    colors = ("#2166ac", "#67a9cf", "#b35806", "#762a83")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, location):
        for ext in ("png", "pdf"):
            fig.savefig(location.with_suffix(f".{ext}"), dpi=250, bbox_inches="tight")
        plt.close(fig)

    def bars(name, labels, values, ylabel, limit=None):
        fig, ax = plt.subplots(figsize=(max(6, len(labels)), 4))
        x = np.arange(len(labels))
        for i, c in enumerate(CONDITIONS):
            ax.bar(x + .19*(i-1.5), [np.nan if v is None else v for v in values[c]], .19, color=colors[i], label=c)
        ax.set_xticks(x, [label.replace("_", " ") for label in labels], rotation=28, ha="right")
        ax.set_ylabel(ylabel)
        if limit:
            ax.set_ylim(0, limit)
        ax.legend(frameon=False)
        save(fig, run / "plots" / name)

    keys = ("mAP50", "mAP50-95", "fixed_precision", "fixed_recall")
    bars("accuracy", keys, {c: [metrics[c][key] for key in keys] for c in CONDITIONS}, "Fraction", 1)
    labels = [row["class"] for row in classes if row["condition"] == "A"]
    for key in ("AP50", "AP50-95"):
        bars(f"class_{key}", labels, {c: [r[key] for r in classes if r["condition"] == c] for c in CONDITIONS}, key, 1)
    for field in GEOMETRY:
        bars(f"recall_{field}", ("Q1", "Q2", "Q3", "Q4"),
             {c: [r["recall"] for r in sizes if r["condition"] == c and r["measure"] == field] for c in CONDITIONS},
             "Fixed recall", 1)
    bars("errors", CATEGORIES, {c: [r["count"] for r in errors if r["condition"] == c] for c in CONDITIONS}, "Assignments")
    fig, ax = plt.subplots(figsize=(6, 4))
    for c, color in zip(CONDITIONS, colors):
        ax.step([r["fp_per_image"] for r in curves[c]], [r["recall"] for r in curves[c]], where="post", color=color, label=c)
    ax.set(xlabel="False positives per image", ylabel="Recall", xlim=(0, max(conf["FP_PER_IMAGE_BUDGETS"])), ylim=(0, 1))
    ax.legend(frameon=False)
    save(fig, run / "plots" / "recall_fp")

    rng = random.Random(conf["SEED"])
    for c in ("C", "D"):
        for outcome in ("recovered", "lost"):
            group = [obj for obj in objects if obj[f"{c}_vs_A"] == outcome]
            for index, obj in enumerate(rng.sample(group, min(3, len(group))), 1):
                with Image.open(DATASET / obj["image"]) as original:
                    pixels = original.convert("RGB")
                fig, axes = plt.subplots(2, 1, figsize=(9, 7), layout="constrained")
                for ax, condition in zip(axes, ("A", c)):
                    ax.imshow(pixels)
                    for p in predictions[condition][obj["image"]]:
                        if p["score"] >= MATCH_SETTINGS[0]:
                            x, y, x2, y2 = box(p)
                            ax.add_patch(Rectangle((x, y), x2-x, y2-y, fill=False, edgecolor="#ffb34a"))
                    x, y, x2, y2 = box(obj)
                    ax.add_patch(Rectangle((x, y), x2-x, y2-y, fill=False, edgecolor="#55efbf", linewidth=2))
                    ax.set_title(f"{condition} · {obj[f'{condition}_category']}")
                    ax.set_axis_off()
                fig.suptitle(f"{outcome.title()} {obj['class']} · {Path(obj['image']).name}")
                save(fig, run / "examples" / f"{c}-{outcome}-{index:02}")


def report(run, metrics, classes, intervals, outcomes, partial, missed):
    lines = ["# E2 validation comparison · pending scientific review", "",
             "| Condition | Train / inference | mAP50 | mAP50–95 | Native P / R | Fixed P / R | TP / FP / FN |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for c, (train_size, size) in CONDITIONS.items():
        row = metrics[c]
        lines.append(f"| {c} | {train_size} / {size} | {row['mAP50']:.2%} | {row['mAP50-95']:.2%} | "
                     f"{row['native_precision']:.2%} / {row['native_recall']:.2%} | "
                     f"{row['fixed_precision']:.2%} / {row['fixed_recall']:.2%} | {row['tp']} / {row['fp']} / {row['fn']} |")
    lines.extend(["", "## Paired differences", "",
                  "| Comparison | Metric | Change (percentage points) | 95% interval (points) |", "|---|---|---:|---:|"])
    for row in intervals:
        interval = "Undefined" if row["ci_2.5"] is None else f"{100*row['ci_2.5']:+.2f} to {100*row['ci_97.5']:+.2f}"
        lines.append(f"| {row['comparison']} | {row['metric']} | {100*row['delta']:+.2f} | {interval} |")
    lines.extend(["", "D–A is the primary comparison of the two matched pipelines. D–B holds inference at 960. "
                  "C–A holds inference at 640. D–C varies inference size within the E2 checkpoint. Across-checkpoint "
                  "comparisons include the physical batch and accumulation change. The bootstrap quantifies "
                  "validation-image uncertainty for these checkpoints. Training-seed variation remains unmeasured.", "",
                  f"Objects missed by all four conditions: **{missed}**.", ""])
    for c in ("B", "C", "D"):
        lines.append(f"- {c} versus A. Recovered {outcomes[c]['recovered']}, lost {outcomes[c]['lost']}, "
                     f"retained {outcomes[c]['retained']}.")
    lines.extend(["", "## Partial-defect diagnostics", "",
                  "| Condition | Localization errors measured | Median GT coverage | Median predicted / GT area | Contained, area ratio < 0.5 |",
                  "|---|---:|---:|---:|---:|"])
    for c, row in partial.items():
        medians = [f"{row[key]:.3f}" if row[key] is not None else "—" for key in ("median_gt_coverage", "median_area_ratio")]
        lines.append(f"| {c} | {row['objects']} | {' | '.join(medians)} | {row['partial']} |")
    lines.extend(["", "For each localization error, the highest-IoU same-class box above the fixed confidence floor "
                  "supplies the coverage and area ratios. Contained means at least 80% of predicted area lies inside "
                  "the ground truth. These are descriptive diagnostics.", "", "## Priority classes", ""])
    for name in FOCUS:
        selected = [r for r in classes if r["class"] == name]
        lines.append(f"- {name.replace('_', ' ').title()} ({selected[0]['objects']} objects). Fixed recall "
                     + ", ".join(f"{r['condition']} {r['fixed_recall']:.1%}" for r in selected) + ".")
    lines.extend(["", "Rolled pit has four objects across two images, so its results are descriptive.", "",
                  "E2 timing is in the timing CSV. It uses the repeated E1 procedure on the recorded GPU. "
                  "A direct comparison to B1 timing requires verified hardware and measurement comparability."])
    (run / "report.md").write_text("\n".join(lines) + "\n")


def run_experiment(run, conf, prior, truth, checkpoint, b1, guesses, native, hashes):
    predictions = {"A": guesses[640], "B": guesses[960]}
    records = {"A": native[640], "B": native[960]}
    expected = {"A": b1["metrics"]["640"], "B": b1["metrics"]["960"]}
    for c in ("C", "D"):
        size = CONDITIONS[c][1]
        predictions[c], records[c] = export_validation(run, conf, size, truth, checkpoint=checkpoint)
        output = read_json(run / str(size) / "metrics.json")
        expected[c] = {key: output[f"metrics/{key}(B)"] for key in ("mAP50", "mAP50-95")}
    for c in ("A", "B"):
        size = CONDITIONS[c][1]
        write_json(run / "predictions" / f"{c}-input.json", {"resolution": size, "input_run": conf["B1_RUN"],
                                                            "reused": True})
    metrics, classes, errors, sizes, curves, objects = diagnostics(run, conf, truth, predictions, records, expected)
    timing = measure_timing(run, conf, checkpoint=checkpoint)
    intervals = paired_bootstrap(run, conf, truth, objects, records, metrics)
    partial = {}
    outcomes = {c: dict(Counter(obj[f"{c}_vs_A"] for obj in objects)) for c in ("B", "C", "D")}
    for c in CONDITIONS:
        rows = [o for o in objects if o[f"{c}_gt_coverage"] is not None]
        partial[c] = {
            "objects": len(rows),
            "lost_vs_A": sum(o[f"{c}_vs_A"] == "lost" for o in rows),
            "median_gt_coverage": float(np.median([o[f"{c}_gt_coverage"] for o in rows])) if rows else None,
            "median_area_ratio": float(np.median([o[f"{c}_area_ratio"] for o in rows])) if rows else None,
            "partial": sum((o[f"{c}_containment"] or 0) >= .8 and o[f"{c}_area_ratio"] < .5 for o in rows),
        }
    for row in outcomes.values():
        for key in ("recovered", "lost", "retained", "both_missed"):
            row.setdefault(key, 0)
    missed = sum(obj["missed_all"] for obj in objects)
    figures(run, conf, metrics, classes, errors, sizes, curves, objects, predictions)
    report(run, metrics, classes, intervals, outcomes, partial, missed)
    write_json(run / "summary.json", {
        "status": "completed", "metrics": metrics, "images": len(truth), "objects": len(objects), "split": "val",
        "B1_checkpoint_sha256": b1["checkpoint_sha256"], "E2_checkpoint_sha256": sha256(checkpoint),
        "dataset_manifest_sha256": prior["dataset_manifest_sha256"], "d1_objects_sha256": b1["d1_objects_sha256"],
        "quartiles": prior["quartiles"], "upstream_commit": UPSTREAM_COMMIT, "reused_input_sha256": hashes,
        "outcomes_vs_A": outcomes, "missed_by_all": missed, "bootstrap": intervals,
        "partial_detection": partial, "timing": timing, "timing_device": read_json(run / "timing_device.json"),
        "rolled_pit": [obj for obj in objects if obj["class"] == "rolled_pit"], "pbs_job_id": os.getenv("PBS_JOBID"),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/evaluation/age_yolo_e2.yaml")
    parser.add_argument("--preflight", action="store_true", help="Verify inputs without CUDA or creating a run.")
    args = parser.parse_args()
    conf = yaml.safe_load(args.config.read_text())
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", conf["RUN_ID"]) or not re.fullmatch(r"cuda:(0|[1-9][0-9]*)", conf["DEVICE"]):
        parser.error("Use a simple run ID and an authorized logical CUDA index.")
    if (conf["BOOTSTRAP_REPETITIONS"], conf["SEED"]) != (5000, 42):
        parser.error("Use 5,000 paired repetitions with seed 42.")
    if not (0 < conf["WARMUP_BATCHES"] < 15 and 0 < conf["LATENCY_IMAGES"] <= 230 - conf["WARMUP_BATCHES"]):
        parser.error("Invalid warmup or latency sampling counts.")
    if not conf["FP_PER_IMAGE_BUDGETS"] or any(not np.isfinite(v) or v <= 0 for v in conf["FP_PER_IMAGE_BUDGETS"]):
        parser.error("Use finite positive FP/image budgets.")
    run = ROOT / "experiments" / conf["RUN_ID"]
    if run.exists() or run.is_symlink():
        parser.error(f"Choose an unused run ID. Already exists: {run}")
    prior, truth, checkpoint, b1, guesses, native, hashes = verify_inputs(conf)
    if args.preflight:
        print(f"Verified E2 best.pt, frozen B1 predictions and 345 objects on 230 validation images. Fresh path: {run}")
        return
    source = read_json(ROOT / ".cache/source.json")
    paths = ("scripts/evaluate_e2.py", "scripts/evaluate_resolution.py", "scripts/evaluate.py",
             "scripts/evaluate_fusion.py", "scripts/analyze_errors.py", "scripts/check_env.py", "submit_e2_eval.sh",
             str(args.config.resolve().relative_to(ROOT)))
    if source["upstream_commit"] != UPSTREAM_COMMIT or any(sha256(ROOT / name) != source["sha256"].get(name) for name in paths):
        raise ValueError("Source differs from deployment record. Run scripts/record_source.py before submitting.")
    run.mkdir(exist_ok=False)
    for directory in ("predictions", "plots", "examples"):
        (run / directory).mkdir()
    write_json(run / "source.json", source)
    with (run / "effective_config.yaml").open("x") as stream:
        yaml.safe_dump({**conf, "SPLIT": "val", "CONDITIONS": {k: list(v) for k, v in CONDITIONS.items()},
                        "BATCH": 16, "WORKERS": 8, "RECT": True, "FP16": False,
                        "NMS_CONFIDENCE_FLOOR": .001, "NMS_IOU": .7, "MAX_DETECTIONS": 300, "AGNOSTIC_NMS": False,
                        "TEST_TIME_AUGMENTATION": False, "CONFIDENCE": .25, "MATCH_IOU": .5, "BACKGROUND_IOU": .1,
                        "TIMING_ORDER": [640, 960, 960, 640]}, stream, sort_keys=False)
    with (run / "terminal.log").open("x", buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            with TemporaryDirectory(prefix="settings-", dir=run) as settings:
                os.environ["YOLO_CONFIG_DIR"] = settings
                try:
                    run_experiment(run, conf, prior, truth, checkpoint, b1, guesses, native, hashes)
                except Exception:
                    traceback.print_exc()
                    raise SystemExit(1) from None
            print(f"E2 evaluation artifacts: {run}", flush=True)


if __name__ == "__main__":
    main()
