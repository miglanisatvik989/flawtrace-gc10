"""Fetch and extract only the GC10 byte range of the authors' pinned archive."""

import hashlib
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import requests


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DESTINATION = DATA / "gc10-det"
PROVENANCE = DATA / "manifests" / "gc10-provenance.json"
CHUNK_SIZE = 32_000_000


def download_part(url, start, end, archive_bytes, destination):
    content_range = f"bytes {start}-{end}/{archive_bytes}"
    for attempt in range(3):
        try:
            with requests.get(
                url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                stream=True, timeout=(15, 60),
            ) as response:
                if response.status_code != 206 or response.headers.get("Content-Range") != content_range:
                    raise RuntimeError(f"Server did not honor GC10 range {content_range}.")
                with destination.open("wb") as output:
                    for chunk in response.iter_content(1024 * 1024):
                        output.write(chunk)
            if destination.stat().st_size != end - start + 1:
                raise IOError(f"Incomplete download for {content_range}.")
            return destination
        except (OSError, requests.RequestException):
            if attempt == 2:
                raise


def download_prefix(source, directory, output):
    size = source["gc10_prefix_bytes"]
    with ThreadPoolExecutor(max_workers=12) as pool:
        parts = [
            pool.submit(
                download_part, source["archive_url"], start,
                min(start + CHUNK_SIZE, size) - 1, source["archive_bytes"],
                directory / f"{start:010d}.part",
            )
            for start in range(0, size, CHUNK_SIZE)
        ]
        with output.open("wb") as combined:
            for part in parts:
                with part.result().open("rb") as chunk:
                    shutil.copyfileobj(chunk, combined)


def main():
    if DESTINATION.exists():
        raise SystemExit(f"{DESTINATION} already exists. Run scripts/check_data.py to verify it.")
    if not shutil.which("bsdtar"):
        raise SystemExit("Install bsdtar (libarchive-tools on Ubuntu) before preparing GC10.")
    source = json.loads(PROVENANCE.read_text())
    prefix = DATA / ".downloads" / "gc10-prefix.bin"
    prefix.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="gc10-", dir=prefix.parent) as temporary:
        directory = Path(temporary)
        payload = prefix if prefix.exists() else directory / "gc10-prefix.bin"
        if not payload.exists():
            print(f"Downloading {source['gc10_prefix_bytes']:,} GC10 archive bytes.", flush=True)
            download_prefix(source, directory, payload)
        with payload.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if payload.stat().st_size != source["gc10_prefix_bytes"] or digest != source["gc10_prefix_sha256"]:
            raise SystemExit(f"GC10 archive prefix checksum or size mismatch: {payload}")
        if payload != prefix:
            payload.rename(prefix)

        archive = directory / "gc10.rar"
        shutil.copyfile(prefix, archive)
        with archive.open("ab") as stream:
            stream.write(bytes.fromhex(source["rar_end_header_hex"]))
        subprocess.run(["bsdtar", "-xf", str(archive), "-C", str(directory)], check=True)
        extracted = directory / "gc10"
        if sorted(p.name for p in extracted.iterdir()) != source["gc10_root_entries"]:
            raise SystemExit("Unexpected GC10 archive structure.")
        extracted.rename(DESTINATION)
    print(f"Extracted verified GC10 archive members to {DESTINATION}")


if __name__ == "__main__":
    main()
