"""Capture deployment provenance on the login node before submitting a run."""

import argparse
import hashlib
import json
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
             *(ROOT / "scripts" / name for name in ("train.py", "evaluate.py", "check_env.py", "record_source.py")))
    resolved = {path.resolve() for path in files}
    return sorted(str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path) for path in resolved)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/hpc.yaml")
    args = parser.parse_args()
    config = args.config.resolve(strict=True)
    conf = yaml.safe_load(config.read_text())
    required = training_files(config, ROOT / conf["DATA_CONFIG"], ROOT / conf["MODEL_CONFIG"])
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
    destination = ROOT / ".cache/source.json"
    destination.parent.mkdir(exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(destination)
    print(destination)


if __name__ == "__main__":
    main()
