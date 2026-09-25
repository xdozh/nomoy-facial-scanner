#!/usr/bin/env python3
"""Reproducible Gate E / T6 / T7 governance and security evaluator."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

os.environ.update({
    "CUDA_VISIBLE_DEVICES": "",
    "HF_HUB_OFFLINE": "1",
    "ULTRALYTICS_OFFLINE": "true",
    "YOLO_OFFLINE": "true",
    "NO_PROXY": "*",
})

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2

import app
import audit
import demo

EVIDENCE = ROOT / "outputs" / "gate_e"
METRICS = EVIDENCE / "metrics.json"
LOG = EVIDENCE / "representative_run_log.jsonl"
VIDEO = EVIDENCE / "representative_output.mp4"
EXCERPTS = EVIDENCE / "safe_log_excerpts.json"
SAMPLE = ROOT / "samples" / "test.mp4"


def check(name: str, passed: bool, detail: Any, criterion: str) -> dict[str, Any]:
    return {"name": name, "criterion": criterion, "passed": bool(passed), "detail": detail}


def clean_error(call: Any, expected: type[BaseException], contains: str) -> dict[str, Any]:
    try:
        call()
    except Exception as exc:
        message = str(exc)
        return {
            "raised": type(exc).__name__,
            "message": message,
            "expected_type": expected.__name__,
            "clean": isinstance(exc, expected) and contains.lower() in message.lower()
            and "traceback" not in message.lower() and "File \"" not in message,
            "cause_type": type(exc.__cause__).__name__ if exc.__cause__ else None,
        }
    return {"raised": None, "message": None, "expected_type": expected.__name__, "clean": False}


def make_clip(dst: Path, frames: int = 30) -> int:
    cap = cv2.VideoCapture(str(SAMPLE))
    if not cap.isOpened():
        raise RuntimeError("sample video cannot be opened")
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    written = 0
    while written < frames:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
    writer.release()
    cap.release()
    return written


def event_key(event: dict[str, Any], frame: int | None = None) -> tuple[Any, ...]:
    return (
        event.get("event"),
        event.get("frame", frame),
        event.get("track_id"),
        event.get("identity"),
        event.get("confidence"),
    )


def main() -> int:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    for path in (METRICS, LOG, VIDEO, EXCERPTS):
        path.unlink(missing_ok=True)
    checks: list[dict[str, Any]] = []

    # Integrity must be established before any inference.
    verified = audit.verify_models(ROOT / "models.lock", root=ROOT)
    expected_locked = json.loads((ROOT / "models.lock").read_text(encoding="utf-8"))["models"]
    integrity_ok = len(verified) == len(expected_locked) and all(
        verified[item["path"]]["sha256"] == item["sha256"] for item in expected_locked
    )
    checks.append(check("real model integrity verified first", integrity_ok, verified, "Gate E/T7"))

    emitted: list[tuple[Any, ...]] = []
    outbound: list[str] = []
    original_blocked = audit._blocked
    original_tracker_update = demo.PersonTracker.update
    original_recognizer_update = demo.FaceRecognizer.update

    def recording_block(*args: Any, **kwargs: Any) -> Any:
        outbound.append(repr(args[1] if len(args) > 1 else args[0] if args else kwargs))
        raise audit.OfflineViolationError("Outbound network access is disabled")

    def tracker_update(self: Any, *args: Any, **kwargs: Any) -> Any:
        tracks, events = original_tracker_update(self, *args, **kwargs)
        frame = kwargs.get("frame_index")
        emitted.extend(event_key(dict(item), frame) for item in events)
        return tracks, events

    def recognizer_update(self: Any, *args: Any, **kwargs: Any) -> Any:
        recognitions, events = original_recognizer_update(self, *args, **kwargs)
        frame = kwargs.get("frame_index")
        emitted.extend(event_key(dict(item), frame) for item in events)
        return recognitions, events

    with tempfile.TemporaryDirectory(prefix="gate-e-") as td:
        temp = Path(td)
        clip = temp / "clip.mp4"
        clip_frames = make_clip(clip)
        audit._blocked = recording_block
        demo.PersonTracker.update = tracker_update
        demo.FaceRecognizer.update = recognizer_update
        started = time.perf_counter()
        try:
            result = demo.process_video(clip, consent=True, output_video=VIDEO, output_log=LOG)
            integrated_error = None
        except Exception as exc:
            result = None
            integrated_error = f"{type(exc).__name__}: {exc}"
        finally:
            audit._blocked = original_blocked
            demo.PersonTracker.update = original_tracker_update
            demo.FaceRecognizer.update = original_recognizer_update
        elapsed = time.perf_counter() - started

        records = audit.verify_log(LOG) if LOG.is_file() else []
        completions = [r for r in records if r.get("event") == "frame_complete"]
        logged_events = [event_key(r) for r in records if r.get("event") in audit.ALLOWED_EVENTS]
        recognition_emitted = [e for e in emitted if e[0] in {"match", "reidentify"}]
        recognition_logged = [e for e in logged_events if e[0] in {"match", "reidentify"}]
        lifecycle_emitted = [e for e in emitted if e[0] in {"track_birth", "track_death"}]
        lifecycle_logged = [e for e in logged_events if e[0] in {"track_birth", "track_death"}]
        startup = records[0] if records else {}
        startup_models = startup.get("model_versions", {})
        startup_has_versions_hashes = bool(startup_models) and all(
            isinstance(value, dict) and bool(value.get("version")) and len(value.get("sha256", "")) == 64
            for value in startup_models.values()
        )
        checks.extend([
            check("representative integrated process completes offline", integrated_error is None and result is not None and not outbound, {"frames": clip_frames, "elapsed_seconds": elapsed, "error": integrated_error, "outbound_attempts": outbound}, "T7"),
            check("JSONL hash chain verifies", bool(records), {"records": len(records), "final_hash": records[-1].get("record_hash") if records else None}, "T6"),
            check("frame_complete sequence is contiguous and complete", [r.get("frame") for r in completions] == list(range(clip_frames)), {"input_frames": clip_frames, "completion_frames": len(completions), "first": completions[0].get("frame") if completions else None, "last": completions[-1].get("frame") if completions else None}, "T6"),
            check("recognition events are complete", recognition_logged == recognition_emitted, {"emitted": len(recognition_emitted), "logged": len(recognition_logged), "exact_match": recognition_logged == recognition_emitted}, "T6"),
            check("track lifecycle events are complete", lifecycle_logged == lifecycle_emitted, {"emitted": len(lifecycle_emitted), "logged": len(lifecycle_logged), "exact_match": lifecycle_logged == lifecycle_emitted}, "T6"),
            check("startup metadata contains model versions and hashes", startup_has_versions_hashes, startup_models, "T6/Gate E"),
        ])

        # Tamper only a disposable copy of one artifact and its copied lock.
        tamper_root = temp / "tamper_root"
        artifact = tamper_root / "models" / "fixture.bin"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"verified fixture")
        lock = tamper_root / "models.lock"
        lock.write_text(json.dumps({"schema": "models.lock/v1", "models": [{
            "path": "models/fixture.bin", "model_version": "fixture-v1",
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        }]}), encoding="utf-8")
        audit.verify_models(lock, root=tamper_root)
        artifact.write_bytes(b"tampered fixture")
        old_root, old_lock = demo.ROOT, demo.DEFAULT_MODEL_LOCK
        demo.ROOT, demo.DEFAULT_MODEL_LOCK = tamper_root, lock
        tamper = clean_error(demo._model_versions, demo.DemoError, "checksum mismatch")
        demo.ROOT, demo.DEFAULT_MODEL_LOCK = old_root, old_lock
        checks.append(check("checksum tamper causes startup refusal", tamper["clean"], tamper, "T7"))

        # Sparse file exercises the real 500 MiB threshold without allocation.
        oversized = temp / "oversized.mp4"
        with oversized.open("wb") as handle:
            handle.truncate(audit.MAX_VIDEO_BYTES + 1)
        oversize = clean_error(lambda: audit.validate_video(oversized), audit.InputValidationError, "500MB")
        extension = temp / "bad.exe"
        extension.write_bytes(b"not video")
        ext_error = clean_error(lambda: demo.process_video(extension, consent=True, output_video=temp / "x.mp4", output_log=temp / "x.jsonl"), demo.DemoError, "Unsupported")
        decode = temp / "invalid.mp4"
        decode.write_bytes(b"not a video")
        decode_error = clean_error(lambda: demo.process_video(decode, consent=True, output_video=temp / "y.mp4", output_log=temp / "y.jsonl"), demo.DemoError, "decoded")
        consent_error = clean_error(lambda: demo.process_video(clip, consent=False, output_video=temp / "z.mp4", output_log=temp / "z.jsonl"), demo.DemoError, "consent")
        checks.extend([
            check("oversize sparse input cleanly rejected", oversize["clean"], {**oversize, "logical_bytes": oversized.stat().st_size, "allocated_blocks": oversized.stat().st_blocks}, "T7"),
            check("extension error is clean", ext_error["clean"], ext_error, "Gate E"),
            check("decode error is clean", decode_error["clean"], decode_error, "Gate E"),
            check("consent enforced before processing", consent_error["clean"] and not (temp / "z.jsonl").exists(), consent_error, "Gate E"),
        ])

        # Public CLI must not expose Python traceback details.
        cli = subprocess.run(
            [sys.executable, str(ROOT / "demo.py"), "--video", str(extension), "--no-browser"],
            cwd=ROOT, capture_output=True, text=True, check=False,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
        )
        cli_text = cli.stdout + cli.stderr
        cli_clean = cli.returncode != 0 and "Traceback (most recent call last)" not in cli_text and "File \"" not in cli_text
        checks.append(check("public CLI errors contain no raw traceback", cli_clean, {"returncode": cli.returncode, "stdout": cli.stdout[-500:], "stderr": cli.stderr[-1000:]}, "Gate E"))

        # save_audit=False currently has no retention semantics: record this as a governance defect.
        persistence_path = temp / "not_explicitly_saved.jsonl"
        logger = audit.AuditLogger(persistence_path, {}, consent=True, save=False)
        logger.record_frame(0, 0.0, [{"event": "match", "track_id": 1, "identity": "TEST_IDENTITY", "confidence": 0.9}])
        logger.close()
        retained = persistence_path.exists() and "TEST_IDENTITY" in persistence_path.read_text(encoding="utf-8")
        checks.append(check("no identity persistence unless explicitly saved", not retained, {"save_flag": False, "file_retained_after_close": persistence_path.exists(), "identity_retained": retained}, "Gate E"))

    # Safe excerpts avoid identity values while proving record shape and chaining.
    safe = []
    for record in records[:3] + records[-2:]:
        safe.append({key: ("[REDACTED]" if key == "identity" and value else value) for key, value in record.items()})
    EXCERPTS.write_text(json.dumps(safe, indent=2), encoding="utf-8")

    passed = all(item["passed"] for item in checks)
    metrics = {
        "gate": "E/T6/T7",
        "verdict": "PASS" if passed else "FAIL",
        "generated_at_epoch": time.time(),
        "environment": {"python": sys.version, "platform": platform.platform(), "opencv": cv2.__version__},
        "commands": [
            ".venv/bin/python scripts/verify_models.py",
            ".venv/bin/python -m pytest -q tests/test_audit.py tests/test_integration.py",
            ".venv/bin/python tests/gate_e_eval.py",
        ],
        "checks": checks,
        "t6": {"records": len(records), "frames": len(completions), "recognition_events": len(recognition_logged), "lifecycle_events": len(lifecycle_logged), "startup_has_versions_and_hashes": startup_has_versions_hashes},
        "t7": {"offline_outbound_attempts": outbound, "integrated_error": integrated_error, "tamper_refused": tamper["clean"], "oversize_rejected": oversize["clean"]},
        "failed_checks": [item["name"] for item in checks if not item["passed"]],
        "evidence": [str(path.relative_to(ROOT)) for path in (METRICS, EXCERPTS, LOG, VIDEO)],
    }
    METRICS.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"verdict": metrics["verdict"], "failed_checks": metrics["failed_checks"], "metrics": str(METRICS.relative_to(ROOT))}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
