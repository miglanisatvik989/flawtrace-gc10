"""Audit the unchanged GC10 release and compare it with the recorded manifests."""

import argparse
import csv
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "gc10-det"
MANIFESTS = ROOT / "data" / "manifests"
CONFIG = ROOT / "configs" / "gc10.yaml"
UPSTREAM_CONFIG = ROOT / "upstream" / "AGE-YOLO" / "dataset" / "gc10_data.yaml"
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
EDGE_TOLERANCE = 1e-6


def class_names(path):
    config = yaml.safe_load(path.read_text())
    names = config["names"]
    if isinstance(names, dict):
        names = [names[index] for index in range(10)]
    if len(names) != 10 or len(set(names)) != 10 or config.get("nc", 10) != 10:
        raise ValueError(f"Expected ten unique classes in {path}")
    return names


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inspect_label(path, relative, classes, issues):
    count = 0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeError:
        issues["unreadable_labels"].append(relative)
        return count
    for line_number, line in enumerate(lines, 1):
        fields = line.split()
        if not fields:
            continue
        count += 1
        location = f"{relative}:{line_number}"
        if len(fields) != 5:
            issues["malformed_rows"].append(location)
            continue
        try:
            class_id = int(fields[0])
        except ValueError:
            class_id = -1
        if 0 <= class_id < len(classes):
            classes[class_id] += 1
        else:
            issues["invalid_class_ids"].append(location)
        try:
            x, y, w, h = map(float, fields[1:])
        except ValueError:
            issues["invalid_boxes"].append(location)
            continue
        edges = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
        if (not all(map(math.isfinite, (x, y, w, h))) or min(w, h) <= 0
                or max(w, h) > 1 or not (0 <= x <= 1 and 0 <= y <= 1)
                or min(edges) < -EDGE_TOLERANCE or max(edges) > 1 + EDGE_TOLERANCE):
            issues["invalid_boxes"].append(location)
        elif min(edges) < 0 or max(edges) > 1:
            issues["rounding_only_edges"].append(location)
    if count == 0:
        issues["empty_labels"].append(relative)
    return count


def audit_split(split, names, byte_groups, pixel_groups):
    images_dir, labels_dir = (DATA / category / split for category in ("images", "labels"))
    images = [p for p in sorted(images_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    labels = {p.stem: p for p in sorted(labels_dir.iterdir()) if p.is_file() and p.suffix == ".txt"}
    stems = Counter(path.stem for path in images)
    rows, dimensions, counts = [], Counter(), [0] * len(names)
    issues = {name: [] for name in (
        "missing_labels", "orphan_labels", "empty_labels", "unreadable_labels",
        "duplicate_image_stems", "unexpected_entries", "malformed_rows",
        "invalid_class_ids", "invalid_boxes", "rounding_only_edges", "corrupt_images",
    )}
    issues["orphan_labels"] = sorted(path.relative_to(DATA).as_posix() for stem, path in labels.items() if stem not in stems)
    issues["duplicate_image_stems"] = sorted(stem for stem, count in stems.items() if count > 1)
    expected = set(images) | set(labels.values())
    issues["unexpected_entries"] = sorted(
        path.relative_to(DATA).as_posix() for folder in (images_dir, labels_dir)
        for path in folder.iterdir() if path not in expected
    )
    label_rows = {}
    for stem, label in labels.items():
        name = label.relative_to(DATA).as_posix()
        label_rows[stem] = (name, digest(label), inspect_label(label, name, counts, issues))
    for image in images:
        name = image.relative_to(DATA).as_posix()
        image_digest = digest(image)
        byte_groups[image_digest].append(name)
        label_name, label_digest, objects = label_rows.get(image.stem, ("", "", 0))
        if not label_name:
            issues["missing_labels"].append(name)
        width = height = ""
        try:
            with Image.open(image) as decoded:
                decoded.load()
                width, height = decoded.size
                dimensions[f"{width}x{height}"] += 1
                pixels = hashlib.sha256(decoded.convert("RGB").tobytes()).hexdigest()
                pixel_groups[f"{width}x{height}:{pixels}"].append(name)
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
            issues["corrupt_images"].append(name)
        rows.append((name, label_name, image_digest, label_digest, width, height, objects))
    statistics = {
        "images": len(images), "label_files": len(labels),
        "objects": sum(record[2] for record in label_rows.values()),
        "per_class": dict(zip(names, counts)),
        "dimensions": dict(dimensions),
        "issues": issues,
    }
    return statistics, rows


def duplicate_summary(groups):
    within = {split: [] for split in SPLITS}
    across = []
    for key, files in sorted(groups.items()):
        if len(files) < 2:
            continue
        partitions = defaultdict(list)
        for path in files:
            partitions[path.split("/")[1]].append(path)
        for split, paths in partitions.items():
            if len(paths) > 1:
                within[split].append({"hash": key, "images": paths})
        if len(partitions) > 1:
            across.append({"hash": key, "images": files})
    return {"within_split": within, "across_splits": across}


def record(path, contents, write):
    if write:
        path.write_bytes(contents)
    elif not path.is_file() or path.read_bytes() != contents:
        raise SystemExit(f"Dataset audit differs from {path.relative_to(ROOT)}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifests", action="store_true", help="Record an explicitly reviewed dataset audit.")
    args = parser.parse_args()
    if not DATA.is_dir():
        parser.exit(1, "Prepare GC10 first with scripts/prepare_data.py.\n")
    names = class_names(UPSTREAM_CONFIG)
    config = yaml.safe_load(CONFIG.read_text())
    if any((CONFIG.parent / config[split]).resolve() != DATA / "images" / split for split in SPLITS):
        parser.exit(1, "FlawTrace YAML paths do not match the audited splits.\n")
    if any(class_names(path) != names for path in (CONFIG, DATA / "gc10_data.yaml")):
        parser.exit(1, "GC10 class mapping differs from the pinned upstream YAML.\n")
    provenance = json.loads((MANIFESTS / "gc10-provenance.json").read_text())
    for name, expected in provenance["archive_metadata_sha256"].items():
        if digest(DATA / name) != expected:
            parser.exit(1, f"Archive metadata checksum differs: {name}\n")
    byte_groups, pixel_groups = defaultdict(list), defaultdict(list)
    statistics = {}
    observed_test = Counter()
    for split in SPLITS:
        statistics[split], rows = audit_split(split, names, byte_groups, pixel_groups)
        if split == "test":
            observed_test.update(Path(row[0]).stem for row in rows)
        output = io.StringIO()
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(("image", "label", "image_sha256", "label_sha256", "width", "height", "objects"))
        writer.writerows(rows)
        record(MANIFESTS / f"gc10-{split}.csv", output.getvalue().encode(), args.write_manifests)
        stats = statistics[split]
        findings = ", ".join(f"{key}={len(values)}" for key, values in stats["issues"].items() if values)
        print(f"{split}: {stats['images']} images, {stats['label_files']} label files, "
              f"{stats['objects']} objects. Findings: {findings or 'none'}.")
    duplicates = {"file_sha256": duplicate_summary(byte_groups), "decoded_rgb": duplicate_summary(pixel_groups)}
    listed_test = Counter((DATA / "testlist_gc10.txt").read_text().splitlines())
    test_list = {
        "entries": listed_test.total(),
        "missing_from_test_images": sorted((listed_test - observed_test).elements()),
        "missing_from_author_list": sorted((observed_test - listed_test).elements()),
    }
    result = {"splits": statistics, "duplicates": duplicates, "author_test_list": test_list}
    record(MANIFESTS / "gc10-audit.json", (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(), args.write_manifests)
    for method, summary in duplicates.items():
        within = ", ".join(f"{split}={len(summary['within_split'][split])}" for split in SPLITS)
        print(f"{method} duplicate groups: within splits ({within}), across splits={len(summary['across_splits'])}.")
    print(f"Author test list: {test_list['entries']} entries, "
          f"{len(test_list['missing_from_test_images']) + len(test_list['missing_from_author_list'])} differences.")
    print("Recorded audit and split manifests." if args.write_manifests else "Audit and split manifests match.")


if __name__ == "__main__":
    main()
