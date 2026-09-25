#!/usr/bin/env python3
"""Verify pinned model artifacts against models.lock.

This script is intended to be called at startup and by setup; it fails closed
on any missing file or SHA-256 mismatch.
"""
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK_PATH = os.path.join(ROOT, "models.lock")

BUFFALO_L_EXPECTED_ONNX = {
    "det_10g.onnx",
    "w600k_r50.onnx",
    "1k3d68.onnx",
    "2d106det.onnx",
    "genderage.onnx",
}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not os.path.exists(LOCK_PATH):
        print(f"ERROR: lock file not found: {LOCK_PATH}", file=sys.stderr)
        return 1

    with open(LOCK_PATH) as f:
        lock = json.load(f)

    if "models" not in lock:
        print("ERROR: models.lock missing 'models' array", file=sys.stderr)
        return 1

    failed = False
    for entry in lock["models"]:
        rel_path = entry["path"]
        full_path = os.path.join(ROOT, rel_path)
        expected_sha = entry["sha256"]

        if not os.path.exists(full_path):
            print(f"MISSING {rel_path}", file=sys.stderr)
            failed = True
            continue

        actual_sha = _sha256(full_path)
        if actual_sha != expected_sha:
            print(
                f"MISMATCH {rel_path}\n"
                f"  expected: {expected_sha}\n"
                f"  actual:   {actual_sha}",
                file=sys.stderr,
            )
            failed = True
        else:
            print(f"OK {rel_path}")

        if full_path.endswith(".zip"):
            extract_to = os.path.join(ROOT, entry.get("extract_to", rel_path[:-4]))
            if not os.path.isdir(extract_to):
                print(f"MISSING extraction directory {extract_to}", file=sys.stderr)
                failed = True
                continue
            present = {name for name in os.listdir(extract_to) if name.endswith(".onnx")}
            missing = BUFFALO_L_EXPECTED_ONNX - present
            if missing:
                print(
                    f"INCOMPLETE buffalo_l extraction: missing {missing}",
                    file=sys.stderr,
                )
                failed = True
            else:
                print(f"  buffalo_l extraction complete ({len(present)} .onnx files)")

    if failed:
        print("Model verification FAILED", file=sys.stderr)
        return 1

    print("Model verification PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
