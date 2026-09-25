#!/usr/bin/env python3
"""First-run model downloader.

Reads models.lock, downloads any missing pinned artifacts, verifies SHA-256,
and extracts the InsightFace buffalo_l pack. Runtime must make zero outbound
calls; this script is intended for setup/first-run only.
"""
import hashlib
import json
import os
import sys
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK_PATH = os.path.join(ROOT, "models.lock")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as out:
        while True:
            chunk = resp.read(8192)
            if not chunk:
                break
            out.write(chunk)


def main() -> int:
    if not os.path.exists(LOCK_PATH):
        print(f"ERROR: {LOCK_PATH} not found", file=sys.stderr)
        return 1

    with open(LOCK_PATH) as f:
        lock = json.load(f)

    if "models" not in lock:
        print("ERROR: models.lock missing 'models' array", file=sys.stderr)
        return 1

    for entry in lock["models"]:
        rel_path = entry["path"]
        full_path = os.path.join(ROOT, rel_path)
        expected_sha = entry["sha256"]
        url = entry["download_url"]

        if os.path.exists(full_path):
            print(f"Found {rel_path}; verifying checksum...")
            if _sha256(full_path) == expected_sha:
                print(f"  OK {rel_path}")
            else:
                print(f"  CHECKSUM MISMATCH {rel_path}; re-downloading...")
                os.remove(full_path)

        if not os.path.exists(full_path):
            print(f"Downloading {entry['model_version']} from {url} ...")
            _download(url, full_path)
            actual_sha = _sha256(full_path)
            if actual_sha != expected_sha:
                print(
                    f"ERROR: checksum mismatch for {rel_path}\n"
                    f"  expected: {expected_sha}\n"
                    f"  actual:   {actual_sha}",
                    file=sys.stderr,
                )
                return 1
            print(f"  OK downloaded {rel_path}")

        # Extract InsightFace model packs.
        if full_path.endswith(".zip"):
            extract_to = os.path.join(ROOT, entry.get("extract_to", rel_path[:-4]))
            marker = os.path.join(extract_to, ".extracted")
            if not os.path.exists(extract_to) or not os.path.exists(marker):
                print(f"Extracting {rel_path} -> {extract_to} ...")
                os.makedirs(extract_to, exist_ok=True)
                with zipfile.ZipFile(full_path, "r") as z:
                    z.extractall(extract_to)
                with open(marker, "w") as f:
                    f.write(full_path)
                print(f"  OK extracted {rel_path}")
            else:
                print(f"  Already extracted {rel_path}")

    print("All pinned models present and integrity-verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
