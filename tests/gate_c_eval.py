#!/usr/bin/env python3
"""Gate C / T4 QA evaluator — face recognition on the synthetic sample video.

This evaluator is owned by QA.  It exercises the real ``face.py`` boundary with
InsightFace buffalo_l on ``samples/recognition_test.mp4``.  Deterministic
tracks/IDs are generated directly from ``scripts/generate_synthetic_assets.py``
so that Gate C evaluates face recognition independently of the already-certified
tracking gate.

Scope rule: only ``tests/gate_c_eval.py``, ``tests/results.md``, and
``outputs/gate_c/`` are modified by this evaluator.
"""

from __future__ import annotations

import json
import math
import os
import platform
import socket
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

os.environ.update({
    "CUDA_VISIBLE_DEVICES": "",
    "HF_HUB_OFFLINE": "1",
    "NO_PROXY": "*",
})

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from face import FaceRecognizer, UNKNOWN, THRESHOLD_MIN, THRESHOLD_MAX
from scripts.verify_models import main as verify_models

SAMPLE_VIDEO = ROOT / "samples" / "recognition_test.mp4"
CONFIG_PATH = ROOT / "config.yaml"
OUTPUT_DIR = ROOT / "outputs" / "gate_c"
FRAMES_DIR = OUTPUT_DIR / "frames"

VIDEO_FPS = 10
FRAMES_EXPECTED = 120
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
SPRITE_SIZE = 260
FACE_IN_SPRITE = (82, 28, 178, 124)  # x1, y1, x2, y2 of pasted 96x96 face

ENROLLED_NAMES = ["alex", "blair", "casey"]
PERSON_TARGETS = ["alex", "blair", "casey", UNKNOWN]


# ---------------------------------------------------------------------------
# Ground truth: exact reproduction of generate_synthetic_assets.py
# ---------------------------------------------------------------------------
def recognition_positions(frame_index: int) -> list[tuple[int, int]]:
    """Return the four 260x260 sprite top-left corners for *frame_index*."""
    positions = [(20, 100), (340, 330), (660, 100), (980, 330)]
    return [
        (
            base_x + int(12 * math.sin((frame_index + person_index * 13) / 10)),
            base_y + int(8 * math.cos((frame_index + person_index * 7) / 12)),
        )
        for person_index, (base_x, base_y) in enumerate(positions)
    ]


def intended_boxes(frame_index: int) -> list[list[float]]:
    """Return the four synthetic sprite boxes for *frame_index*."""
    return [
        [float(x), float(y), float(x + SPRITE_SIZE), float(y + SPRITE_SIZE)]
        for x, y in recognition_positions(frame_index)
    ]


def face_box_in_frame(sprite_box: list[float]) -> list[float]:
    """Return the embedded 96x96 face box in frame coordinates."""
    sx, sy = sprite_box[0], sprite_box[1]
    fx1, fy1, fx2, fy2 = FACE_IN_SPRITE
    return [sx + fx1, sy + fy1, sx + fx2, sy + fy2]


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------
def annotate_frame(
    frame: np.ndarray,
    frame_index: int,
    boxes: list[list[float]],
    identities: list[str],
    confidences: list[float],
    correct_flags: list[bool],
) -> np.ndarray:
    """Render boxes, identities, confidences, and a synthetic-data header."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Header background
    cv2.rectangle(annotated, (0, 0), (w, 78), (235, 235, 235), -1)
    cv2.putText(
        annotated,
        f"SYNTHETIC GATE C QA | frame {frame_index:03d} | recognition",
        (24, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (10, 10, 10),
        2,
    )
    cv2.putText(
        annotated,
        "Green box: correct identity | Red box: wrong identity",
        (24, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (10, 10, 10),
        1,
    )

    for person_index, (box, identity, conf, correct) in enumerate(
        zip(boxes, identities, confidences, correct_flags)
    ):
        x1, y1, x2, y2 = map(int, box)
        color = (0, 180, 0) if correct else (0, 0, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        target = PERSON_TARGETS[person_index]
        label = f"P{person_index} {identity}"
        if identity != UNKNOWN:
            label += f" {conf:.2f}"
        if identity != target:
            label += f" (want {target})"
        cv2.putText(
            annotated,
            label,
            (x1, max(110, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

        # Also mark the embedded face region (small blue dot / box) for traceability.
        fx1, fy1, fx2, fy2 = map(int, face_box_in_frame(box))
        cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), (255, 0, 0), 1)

    return annotated


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def main() -> dict[str, Any]:
    if verify_models() != 0:
        raise SystemExit("Model integrity verification failed; inference refused")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    # Fail-closed socket guard installed *before* the InsightFace backend is
    # constructed by FaceRecognizer, and kept active for the whole evaluation.
    attempted_connections: list[str] = []
    original_connect = socket.socket.connect
    original_create_connection = socket.create_connection

    def blocked_connect(sock, address):
        attempted_connections.append(str(address))
        raise RuntimeError(f"Outbound network blocked during Gate C evaluation: {address}")

    def blocked_create_connection(address, *args, **kwargs):
        attempted_connections.append(str(address))
        raise RuntimeError(f"Outbound network blocked during Gate C evaluation: {address}")

    socket.socket.connect = blocked_connect
    socket.create_connection = blocked_create_connection

    all_events: list[dict[str, Any]] = []

    try:
        # Load the production config and use the real local InsightFace backend.
        recognizer = FaceRecognizer(str(CONFIG_PATH))

        # Validate the production tuning band and minimum face size required by
        # the build spec.
        if not (THRESHOLD_MIN <= recognizer.match_threshold <= THRESHOLD_MAX):
            raise SystemExit(
                f"match_threshold {recognizer.match_threshold} outside allowed band "
                f"[{THRESHOLD_MIN}, {THRESHOLD_MAX}]"
            )
        if recognizer.face_min_size_px < 48:
            raise SystemExit(
                f"face_min_size_px {recognizer.face_min_size_px} is below the 48 px requirement"
            )

        cap = cv2.VideoCapture(str(SAMPLE_VIDEO))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {SAMPLE_VIDEO}")

        frame_count_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = cap.get(cv2.CAP_PROP_FPS)

        frames: list[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()

        if len(frames) != FRAMES_EXPECTED or (width, height) != (VIDEO_WIDTH, VIDEO_HEIGHT):
            raise RuntimeError(
                f"Unexpected video properties: decoded={len(frames)}, "
                f"metadata={frame_count_meta}, size={width}x{height}"
            )

        # Timed recognition pass.  Tracks are generated deterministically from
        # the generator equations, isolating this gate from tracking behavior.
        timed_start = time.perf_counter()
        frame_records: list[dict[str, Any]] = []
        for frame_index in range(len(frames)):
            boxes = intended_boxes(frame_index)
            tracks = [
                {
                    "track_id": person_index,
                    "bbox": box,
                    "age": frame_index + 1,
                    "state": "active",
                }
                for person_index, box in enumerate(boxes)
            ]
            results, events = recognizer.update(frames[frame_index], tracks, frame_index)
            all_events.extend(events)
            # results order matches tracks order because track IDs are 0..3 sorted.
            frame_records.append({
                "frame": frame_index,
                "boxes": boxes,
                "results": results,
                "events": events,
            })
        elapsed = time.perf_counter() - timed_start
        recognition_fps = len(frames) / elapsed if elapsed > 0 else 0.0
    finally:
        socket.socket.connect = original_connect
        socket.create_connection = original_create_connection

    # Continue evidence generation offline (video encode does not need sockets).

    # Per-person metrics.
    visible_frames: dict[int, int] = defaultdict(int)
    correct_frames: dict[int, int] = defaultdict(int)
    identity_changes: dict[int, int] = defaultdict(int)
    last_identity: dict[int, str] = {}
    confidence_sum: dict[int, float] = defaultdict(float)
    confidence_count: dict[int, int] = defaultdict(int)
    false_accept_frames: list[int] = []
    per_frame_details: list[dict[str, Any]] = []

    for record in frame_records:
        frame_index = record["frame"]
        frame_identities: list[str] = []
        frame_confidences: list[float] = []
        frame_correct: list[bool] = []
        for person_index, (box, result) in enumerate(
            zip(record["boxes"], record["results"])
        ):
            visible_frames[person_index] += 1
            identity = result["identity"]
            confidence = result["confidence"]
            frame_identities.append(identity)
            frame_confidences.append(confidence)
            target = PERSON_TARGETS[person_index]
            correct = identity == target
            frame_correct.append(correct)
            if correct:
                correct_frames[person_index] += 1
            confidence_sum[person_index] += confidence
            confidence_count[person_index] += 1

            if identity != UNKNOWN and person_index == 3:
                false_accept_frames.append(frame_index)

            if person_index in last_identity and last_identity[person_index] != identity:
                identity_changes[person_index] += 1
            last_identity[person_index] = identity

        per_frame_details.append({
            "frame": frame_index,
            "identities": frame_identities,
            "confidences": frame_confidences,
            "correct": frame_correct,
        })

    # Annotated evidence video: every frame at 10 FPS.
    writer = cv2.VideoWriter(
        str(OUTPUT_DIR / "annotated_recognition.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        VIDEO_FPS,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create annotated recognition evidence video")

    contact_frames: list[np.ndarray] = []
    for record in frame_records:
        frame_index = record["frame"]
        annotated = annotate_frame(
            frames[frame_index],
            frame_index,
            record["boxes"],
            [r["identity"] for r in record["results"]],
            [r["confidence"] for r in record["results"]],
            [
                r["identity"] == PERSON_TARGETS[person_index]
                for person_index, r in enumerate(record["results"])
            ],
        )
        writer.write(annotated)
        if frame_index % 10 == 0:
            cv2.imwrite(str(FRAMES_DIR / f"frame_{frame_index:03d}.jpg"), annotated)
            contact_frames.append(cv2.resize(annotated, (426, 240)))

    writer.release()

    # Contact sheet.
    if contact_frames:
        rows = [
            np.hstack(contact_frames[i : i + 3])
            for i in range(0, len(contact_frames), 3)
        ]
        if rows:
            cv2.imwrite(str(OUTPUT_DIR / "contact_sheet.jpg"), np.vstack(rows))

    # Compile metrics.
    recall_per_person: dict[int, float] = {}
    avg_confidence: dict[int, float | None] = {}
    flicker_per_person: dict[int, int] = {}
    for person_index in range(4):
        total = visible_frames[person_index]
        recall_per_person[person_index] = (
            correct_frames[person_index] / total if total else 0.0
        )
        avg_confidence[person_index] = (
            confidence_sum[person_index] / confidence_count[person_index]
            if confidence_count[person_index]
            else None
        )
        flicker_per_person[person_index] = identity_changes[person_index]

    # Acceptance criteria.
    enrolled_recalls_pass = all(
        recall_per_person[i] >= 0.80 for i in range(3)
    )
    zero_false_accepts = len(false_accept_frames) == 0
    threshold_in_band = THRESHOLD_MIN <= recognizer.match_threshold <= THRESHOLD_MAX
    min_face_size_ok = recognizer.face_min_size_px >= 48
    zero_flicker = all(v == 0 for v in flicker_per_person.values())

    gate_c_pass = (
        enrolled_recalls_pass
        and zero_false_accepts
        and threshold_in_band
        and min_face_size_ok
    )

    # T4 adds explicit flicker measurement and zero-false-accept requirement.
    t4_pass = zero_false_accepts and zero_flicker

    report: dict[str, Any] = {
        "synthetic_stopgap": True,
        "video": {
            "path": str(SAMPLE_VIDEO),
            "decoded_frames": len(frames),
            "metadata_frames": frame_count_meta,
            "width": width,
            "height": height,
            "source_fps": source_fps,
        },
        "methodology": {
            "config": str(CONFIG_PATH),
            "recognizer": "face.FaceRecognizer(config.yaml) with real InsightFaceBackend",
            "model_dir": str(ROOT / "models" / "buffalo_l"),
            "tracks": "deterministic from generate_synthetic_assets.py equations, IDs 0..3",
            "face_size_in_crop": f"{FACE_IN_SPRITE[2] - FACE_IN_SPRITE[0]}x"
                                 f"{FACE_IN_SPRITE[3] - FACE_IN_SPRITE[1]}",
            "match_threshold": recognizer.match_threshold,
            "face_min_size_px": recognizer.face_min_size_px,
            "vote_window": 20,
            "face_check_interval_frames": recognizer.face_check_interval_frames,
            "enrolled_target_mapping": {
                0: "alex",
                1: "blair",
                2: "casey",
                3: "UNKNOWN",
            },
        },
        "model_integrity": "PASSED",
        "runtime": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "offline": True,
            "outbound_connections_attempted": attempted_connections,
        },
        "performance": {
            "timed_frames": len(frames),
            "elapsed_seconds": elapsed,
            "recognition_fps": recognition_fps,
            "decode_excluded": True,
            "render_excluded": True,
        },
        "recognition_events": {
            "total_match_events": sum(
                1 for e in all_events if e.get("event") == "match"
            ),
            "total_reidentify_events": sum(
                1 for e in all_events if e.get("event") == "reidentify"
            ),
            "events": all_events,
        },
        "recognition_metrics": {
            "recall_per_person": {str(i): recall_per_person[i] for i in range(4)},
            "correct_frames_per_person": {str(i): correct_frames[i] for i in range(4)},
            "visible_frames_per_person": {str(i): visible_frames[i] for i in range(4)},
            "avg_confidence_per_person": {str(i): avg_confidence[i] for i in range(4)},
            "flicker_per_person": {str(i): flicker_per_person[i] for i in range(4)},
            "false_accept_frames": false_accept_frames,
            "false_accept_count": len(false_accept_frames),
        },
        "acceptance": {
            "enrolled_recall_ge_80_percent": enrolled_recalls_pass,
            "zero_false_accepts": zero_false_accepts,
            "threshold_in_allowed_band": threshold_in_band,
            "face_min_size_ge_48px": min_face_size_ok,
            "zero_name_flicker": zero_flicker,
        },
        "gate_decision": {
            "gate_c_pass": gate_c_pass,
            "t4_pass": t4_pass,
        },
        "per_frame_details": per_frame_details,
    }

    with open(OUTPUT_DIR / "metrics.json", "w") as handle:
        json.dump(report, handle, indent=2)

    print(f"Gate C evaluator complete. Decision: {'PASS' if gate_c_pass else 'FAIL'}")
    print(f"  T4 zero-false-accept + zero-flicker: {'PASS' if t4_pass else 'FAIL'}")
    for person_index in range(4):
        print(
            f"  Person {person_index} ({PERSON_TARGETS[person_index]}): "
            f"recall={recall_per_person[person_index]:.2%}, "
            f"flicker={flicker_per_person[person_index]}, "
            f"avg_conf={avg_confidence[person_index]:.3f}"
        )
    print(f"  False accepts: {len(false_accept_frames)} frames")
    print(f"  Recognition FPS: {recognition_fps:.2f}")
    print(f"  Outbound connection attempts: {len(attempted_connections)}")

    return report


if __name__ == "__main__":
    report = main()
    sys.exit(0 if report["gate_decision"]["gate_c_pass"] else 1)
