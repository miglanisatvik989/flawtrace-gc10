"""Prepare or verify the deterministic AGE-YOLO GC10 training augmentation."""

import argparse
import csv
import hashlib
import json
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from check_data import CONFIG, DATA, MANIFESTS, ROOT, SPLITS, class_names, digest


DERIVED = ROOT / "data" / "gc10-det-aug42-x4"
DATA_CONFIG = ROOT / "configs" / "gc10_augmented.yaml"
SEED = 42
MULTIPLIER = 4
FIELDS = ("image", "label", "image_sha256", "label_sha256", "objects", "source_image")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def pixel_hash(data):
    with Image.open(BytesIO(data)) as image:
        return (image.size, sha256(image.convert("RGB").tobytes()))


def transform(image, labels, parameters):
    scale, flip, saturation, value = parameters
    height, width = image.shape[:2]
    image = cv2.resize(image, (int(width * scale), int(height * scale)))
    labels = labels.copy()
    if flip:
        image = image[:, ::-1].copy()
        labels[:, 1] = 1.0 - labels[:, 1]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= saturation
    hsv[:, :, 2] *= value
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR), labels


def make_variants(job):
    row, variants = job
    image = cv2.imread(str(DATA / row["image"]))
    if image is None:
        raise ValueError(f"Could not decode {row['image']}")
    label_text = (DATA / row["label"]).read_text()
    labels = np.array([line.split() for line in label_text.splitlines() if line.strip()], dtype=float).reshape(-1, 5)
    if len(labels) != int(row["objects"]):
        raise ValueError(f"Object count differs for {row['image']}")
    for number, parameters in enumerate(variants, 1):
        transformed, boxes = transform(image, labels, parameters)
        stem = f"{Path(row['image']).stem}_aug{number}"
        success, encoded = cv2.imencode(".jpg", transformed)
        if not success:
            raise RuntimeError(f"Could not encode {stem}")
        label = "".join(
            f"{int(b[0])} {' '.join(f'{coord:.6f}' for coord in b[1:])}\n" for b in boxes
        ).encode()
        # Match the pinned script: variants of unlabeled backgrounds have no TXT.
        yield (
            f"images/train/{stem}.jpg", f"labels/train/{stem}.txt" if label else "",
            encoded.tobytes(), label, len(boxes), row["image"],
        )


def source_rows():
    rows = {}
    for split in SPLITS:
        with (MANIFESTS / f"gc10-{split}.csv").open(newline="") as stream:
            rows[split] = sorted(csv.DictReader(stream), key=lambda row: row["image"])
        for field, category in (("image", "images"), ("label", "labels")):
            expected = {row[field] for row in rows[split]}
            actual = {path.relative_to(DATA).as_posix() for path in (DATA / category / split).iterdir()}
            if expected != actual:
                raise ValueError(f"Original {category}/{split} membership differs from the audit.")
    return rows


def prepare(destination, verify):
    audit = json.loads((MANIFESTS / "gc10-audit.json").read_text())
    rows = source_rows()
    index = []
    heldout_bytes, heldout_pixels = set(), set()
    for split in SPLITS:
        if not verify:
            for category in ("images", "labels"):
                (destination / category / split).mkdir(parents=True)
        for row in rows[split]:
            for field in ("image", "label"):
                relative = row[field]
                original = DATA / relative
                target = destination / relative
                if digest(original) != row[f"{field}_sha256"]:
                    raise ValueError(f"Original checksum differs: {relative}")
                if not verify:
                    shutil.copyfile(original, target)
                if digest(target) != row[f"{field}_sha256"]:
                    raise ValueError(f"Copied checksum differs: {relative}")
            if split == "train":
                index.append(tuple(row[key] for key in FIELDS[:-1]) + (row["image"],))
            else:
                heldout_bytes.add(row["image_sha256"])
                heldout_pixels.add(pixel_hash((DATA / row["image"]).read_bytes()))

    rng = random.Random(SEED)
    jobs = [
        (row, [(rng.uniform(0.8, 1.2), rng.random() < 0.5, rng.uniform(0.8, 1.2), rng.uniform(0.8, 1.2))
               for _ in range(MULTIPLIER - 1)])
        for row in rows["train"]
    ]
    cv2.setNumThreads(1)
    seen = {row[0] for row in index}
    # Batch submissions so encoded images cannot accumulate without bound.
    with ThreadPoolExecutor(max_workers=6) as pool:
        for start in tqdm(range(0, len(jobs), 24), desc="Verifying AUG42" if verify else "Preparing AUG42"):
            for variants in pool.map(lambda job: list(make_variants(job)), jobs[start:start + 24]):
                for image, label, pixels, annotations, objects, source in variants:
                    image_hash = sha256(pixels)
                    if image in seen or image_hash in heldout_bytes or pixel_hash(pixels) in heldout_pixels:
                        raise ValueError(f"Name collision or exact held-out duplicate: {image}")
                    seen.add(image)
                    label_hash = sha256(annotations) if label else ""
                    for relative, contents, expected in ((image, pixels, image_hash), (label, annotations, label_hash)):
                        if not relative:
                            continue
                        path = destination / relative
                        if not verify:
                            path.write_bytes(contents)
                        elif digest(path) != expected:
                            raise ValueError(f"Regenerated checksum differs: {relative}")
                    index.append((image, label, image_hash, label_hash, objects, source))

    expected_count = len(rows["train"]) * MULTIPLIER
    if len(index) != expected_count or len(index) != audit["splits"]["train"]["images"] * MULTIPLIER:
        raise ValueError("Unexpected augmented training count.")
    for split in SPLITS:
        for field, category in (("image", "images"), ("label", "labels")):
            offset = FIELDS.index(field)
            expected = ({row[offset] for row in index if row[offset]} if split == "train"
                        else {row[field] for row in rows[split]})
            actual = {path.relative_to(destination).as_posix() for path in (destination / category / split).iterdir()}
            if expected != actual:
                raise ValueError(f"Derived {category}/{split} membership differs.")

    manifest = destination / "train.csv"
    from io import StringIO
    output = StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(FIELDS)
    writer.writerows(sorted(index))
    contents = output.getvalue().encode()
    summary = {
        "seed": SEED, "multiplier": MULTIPLIER, "train_images": len(index),
        "train_label_files": sum(bool(row[1]) for row in index),
        "train_objects": sum(int(row[4]) for row in index),
        "val_images": len(rows["val"]), "test_images": len(rows["test"]),
        "train_manifest_sha256": sha256(contents),
        "source_manifest_sha256": {split: digest(MANIFESTS / f"gc10-{split}.csv") for split in SPLITS},
        "augmentation_script_sha256": digest(Path(__file__)),
        "opencv": cv2.__version__, "numpy": np.__version__,
    }
    metadata = destination / "augmentation.json"
    if verify:
        if manifest.read_bytes() != contents or json.loads(metadata.read_text()) != summary:
            raise ValueError("Regenerated augmentation manifests differ.")
    else:
        manifest.write_bytes(contents)
        metadata.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Regenerate in memory and verify existing bytes.")
    args = parser.parse_args()
    if class_names(DATA_CONFIG) != class_names(CONFIG):
        parser.exit(1, "Augmented YAML class mapping differs from the original.\n")
    if args.verify:
        prepare(DERIVED, verify=True)
    else:
        if DERIVED.exists():
            parser.exit(1, f"{DERIVED} already exists. Use --verify.\n")
        with TemporaryDirectory(prefix=".gc10-aug-", dir=DERIVED.parent) as temporary:
            derived = Path(temporary) / "dataset"
            prepare(derived, verify=False)
            derived.rename(DERIVED)


if __name__ == "__main__":
    main()
