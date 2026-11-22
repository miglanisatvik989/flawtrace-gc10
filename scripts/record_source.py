"""Capture deployment provenance on the login node before submitting a run."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def training_files(config, data_yaml, model_yaml):
    """Files that must match the source record at training startup."""
    files = (config, data_yaml, model_yaml, ROOT / "upstream/AGE-YOLO/ultralytics/cfg/default.yaml",
             ROOT / "data/manifests/gc10-provenance.json", ROOT / "submit.sh",
             *(ROOT / "scripts" / name for name in (
                 "train.py", "evaluate.py", "check_env.py", "record_source.py", "context_sifhead.py")))
    return source_paths(files)


def source_paths(files):
    resolved = {path.resolve() for path in files}
    return sorted(str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path) for path in resolved)


def architecture_eval_files(config, conf):
    models = {
        "A0_CHECKPOINT": "upstream/AGE-YOLO/ultralytics/cfg/models/age_yolo.yaml",
        "A1_CHECKPOINT": "configs/models/age_yolo_p2.yaml",
    }
    scripts = ("evaluate_architecture", "evaluate_resolution", "evaluate", "analyze_errors",
               "check_env", "record_source", "train")
    launcher = "submit_architecture_eval.sh"
    if "A2_CHECKPOINT" in conf:
        models["A2_CHECKPOINT"] = "configs/models/age_yolo_p2_context.yaml"
        scripts += ("context_sifhead", "evaluate_e2", "evaluate_fusion")
        launcher = "submit_a2_eval.sh"
    upstream = ROOT / "upstream/AGE-YOLO/ultralytics"
    return source_paths([
        config, *(ROOT / path for path in models.values()),
        *(ROOT / Path(conf[key]).parents[1] / Path(model).name
          for key, model in models.items()),
        *(ROOT / "scripts" / f"{script}.py" for script in scripts),
        *(ROOT / path for path in (launcher, "configs/gc10.yaml",
                                  "data/manifests/gc10-val.csv", "data/manifests/gc10-train.csv",
                                  "experiments/d1-baseline-01/summary.json",
                                  "experiments/d1-baseline-01/per_object.csv")),
        *(upstream / path for path in ("cfg/default.yaml", "nn/tasks.py", "nn/autobackend.py",
                                      "nn/modules/age_yolo_modules.py", "engine/validator.py",
                                      "models/yolo/detect/val.py", "utils/metrics.py", "utils/ops.py")),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/hpc.yaml")
    parser.add_argument("--architecture-eval", action="store_true", help="Record both architecture-evaluation models.")
    args = parser.parse_args()
    config = args.config.resolve(strict=True)
    conf = yaml.safe_load(config.read_text())
    if args.architecture_eval:
        if not isinstance(conf["RUN_ID"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", conf["RUN_ID"]):
            parser.error("Expected a simple architecture evaluation run ID.")
        required = architecture_eval_files(config, conf)
        destination = ROOT / ".cache" / f"source-{conf['RUN_ID']}.json"
    else:
        required = training_files(config, ROOT / conf["DATA_CONFIG"], ROOT / conf["MODEL_CONFIG"])
        destination = ROOT / ".cache/source.json"
    upstream = "upstream/AGE-YOLO"
    commit = git("-C", upstream, "rev-parse", "HEAD").strip()
    if commit != git("rev-parse", f"HEAD:{upstream}").strip() or git("-C", upstream, "status", "--porcelain"):
        raise ValueError("Deploy the clean, pinned AGE-YOLO submodule.")
    tracked = set(git("ls-files", "-z", "--cached").split("\0")) - {""}
    untracked = {name for name in git("ls-files", "-zo", "--exclude-standard").split("\0")
                 if Path(name).suffix in {".py", ".sh", ".yaml", ".yml", ".md", ".txt"}}
    files = sorted(tracked | untracked | set(required))
    hashes = {}
    for name in files:
        path = ROOT / name
        if path.is_file():
            with path.open("rb") as stream:
                hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
        elif name in required:
            raise FileNotFoundError(f"Required training input is missing: {name}")
    record = {
        "commit": git("rev-parse", "HEAD").strip(), "upstream_commit": commit,
        "status": git("status", "--porcelain", "-uall"), "patch": git("diff", "--binary", "HEAD"),
        "sha256": hashes,
        "untracked_source": {
            name: (ROOT / name).read_text() for name in sorted(untracked)
        },
    }
    destination.parent.mkdir(exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(destination)
    print(destination)


if __name__ == "__main__":
    main()
