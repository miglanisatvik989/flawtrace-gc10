"""Capture deployment provenance on the login node before submitting a run."""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def main():
    upstream = "upstream/AGE-YOLO"
    commit = git("-C", upstream, "rev-parse", "HEAD").strip()
    if commit != git("rev-parse", f"HEAD:{upstream}").strip() or git("-C", upstream, "status", "--porcelain"):
        raise ValueError("Deploy the clean, pinned AGE-YOLO submodule.")
    files = git("ls-files", "-z", "--cached", "--others", "--exclude-standard").split("\0")
    hashes = {}
    for name in filter(None, files):
        path = ROOT / name
        if path.is_file():
            with path.open("rb") as stream:
                hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    record = {
        "commit": git("rev-parse", "HEAD").strip(), "upstream_commit": commit,
        "status": git("status", "--porcelain", "-uall"), "patch": git("diff", "--binary", "HEAD"),
        "sha256": hashes,
        "untracked_source": {
            name: (ROOT / name).read_text() for name in git("ls-files", "-zo", "--exclude-standard").split("\0")
            if Path(name).suffix in {".py", ".sh", ".yaml", ".md", ".txt"}
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
