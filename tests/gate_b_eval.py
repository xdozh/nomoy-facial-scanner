#!/usr/bin/env python3
"""Gate B / T3 QA evaluator — persistent tracking on the synthetic sample video.

This evaluator is owned by QA.  It runs the detector→tracker pipeline on the
actual ``samples/test.mp4`` asset, maps the resulting BoT-SORT track IDs to the
synthetic ground-truth identities produced by ``scripts/generate_synthetic_assets.py``,
and produces annotated/machine-readable evidence under ``outputs/gate_b/``.

Scope rule: only ``tests/gate_b_eval.py``, ``tests/results.md``, and
``outputs/gate_b/`` are modified by this evaluator.
"""

from __future__ import annotations

import json
import math
import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

os.environ.update({
    "CUDA_VISIBLE_DEVICES": "",
    "HF_HUB_OFFLINE": "1",
    "NO_PROXY": "*",
    "ULTRALYTICS_OFFLINE": "true",
    "YOLO_OFFLINE": "true",
})

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detect import PersonDetector
from scripts.verify_models import main as verify_models
from track import PersonTracker

SAMPLE_VIDEO = ROOT / "samples" / "test.mp4"
OUTPUT_DIR = ROOT / "outputs" / "gate_b"
FRAMES_DIR = OUTPUT_DIR / "frames"

MATCH_DISTANCE_THRESHOLD_PX = 200.0  # center-to-center, generous for 720p synthetic sprites
VIDEO_FPS = 10


# ---------------------------------------------------------------------------
# Ground truth: exact reproduction of generate_synthetic_assets.py
# ---------------------------------------------------------------------------
def _generator_positions(frame_index: int) -> tuple[list[tuple[int, int]], list[int]]:
    swap_one = 0.5 * (1 + np.tanh((frame_index - 75) / 4))
    swap_two = 0.5 * (1 + np.tanh((frame_index - 225) / 4))
    swap = swap_one - swap_two
    positions = [
        (int(40 + 480 * swap), 130),
        (int(520 - 480 * swap), 300),
        (920, 190),
    ]
    order = [0, 2, 1] if frame_index % 80 < 40 else [1, 2, 0]
    return positions, order


def intended_boxes(frame_index: int) -> list[list[float]]:
    """Return the three 300x300 synthetic sprite boxes for *frame_index*."""
    positions, _ = _generator_positions(frame_index)
    return [[float(x), float(y), float(x + 300), float(y + 300)] for x, y in positions]


def box_center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def center_distance(a: list[float], b: list[float]) -> float:
    ca = box_center(a)
    cb = box_center(b)
    return math.hypot(ca[0] - cb[0], ca[1] - cb[1])


def fully_hidden_indices(frame_index: int) -> set[int]:
    """Indices of intended boxes completely covered by a later-drawn opaque sprite."""
    positions, order = _generator_positions(frame_index)
    boxes = [[x, y, x + 300, y + 300] for x, y in positions]
    hidden: set[int] = set()
    for draw_position, person_index in enumerate(order):
        ax1, ay1, ax2, ay2 = boxes[person_index]
        for later_person in order[draw_position + 1:]:
            bx1, by1, bx2, by2 = boxes[later_person]
            if bx1 <= ax1 and by1 <= ay1 and bx2 >= ax2 and by2 >= ay2:
                hidden.add(person_index)
                break
    return hidden


# ---------------------------------------------------------------------------
# Deterministic track↔ground-truth mapping
# ---------------------------------------------------------------------------
def match_tracks_to_ground_truth(
    tracks: list[dict[str, Any]],
    ground_truth: list[list[float]],
    hidden: set[int],
) -> dict[int, int]:
    """Return a mapping ``gt_index -> track_id`` for this frame.

    Only visible GT identities participate.  Assignment is a minimum-cost
    (Euclidean center-distance) bipartite matching solved with the Hungarian
    algorithm, then filtered by ``MATCH_DISTANCE_THRESHOLD_PX``.  Unmatched
    tracks/identities are omitted from the mapping.
    """
    visible_gt = [(i, box) for i, box in enumerate(ground_truth) if i not in hidden]
    if not visible_gt or not tracks:
        return {}

    cost = np.full((len(visible_gt), len(tracks)), 1e9, dtype=np.float64)
    for gi, (gt_index, gt_box) in enumerate(visible_gt):
        for ti, track in enumerate(tracks):
            cost[gi, ti] = center_distance(gt_box, track["bbox"])

    gt_indices, track_indices = linear_sum_assignment(cost)
    mapping: dict[int, int] = {}
    for gi, ti in zip(gt_indices, track_indices):
        if cost[gi, ti] <= MATCH_DISTANCE_THRESHOLD_PX:
            mapping[visible_gt[gi][0]] = tracks[ti]["track_id"]
    return mapping


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------
def annotate_frame(
    frame: np.ndarray,
    frame_index: int,
    tracks: list[dict[str, Any]],
    ground_truth: list[list[float]],
    hidden: set[int],
    mapping: dict[int, int],
    swap_events: list[dict[str, Any]],
    unmatched_tracks: set[int],
) -> np.ndarray:
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Header background
    cv2.rectangle(annotated, (0, 0), (w, 78), (235, 235, 235), -1)
    cv2.putText(
        annotated,
        f"SYNTHETIC GATE B QA | frame {frame_index:03d} | tracks={len(tracks)} | mapped={len(mapping)}",
        (24, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (10, 10, 10),
        2,
    )
    cv2.putText(
        annotated,
        "Green/red: GT matched/missed; Blue: matched track; Orange: unmatched track",
        (24, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (10, 10, 10),
        1,
    )

    # Ground truth boxes
    mapped_gts = set(mapping.keys())
    for gt_index, box in enumerate(ground_truth):
        if gt_index in hidden:
            continue
        color = (0, 180, 0) if gt_index in mapped_gts else (0, 0, 255)
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"GT{gt_index}"
        if gt_index in mapping:
            label += f"→T{mapping[gt_index]}"
        cv2.putText(annotated, label, (x1, max(110, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Track boxes
    mapped_track_ids = set(mapping.values())
    for track in tracks:
        tid = track["track_id"]
        x1, y1, x2, y2 = map(int, track["bbox"])
        color = (255, 120, 0) if tid in unmatched_tracks else (0, 165, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        state_tag = "A" if track["state"] == "active" else "C"
        cv2.putText(annotated, f"T{tid} {state_tag}", (x1, max(110, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Swap flashes
    for swap in swap_events:
        if swap["frame"] == frame_index:
            cv2.putText(
                annotated,
                f"SWAP: GT{swap['gt_index']} {swap['old_track']}->{swap['new_track']}",
                (24, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 220),
                2,
            )

    return annotated


# ---------------------------------------------------------------------------
# Dropout verification (T3 exact 30-frame survival)
# ---------------------------------------------------------------------------
def verify_30_frame_dropout() -> dict[str, Any]:
    """Deterministic synthetic check that PersonTracker preserves ID through 30 missed frames."""
    tracker = PersonTracker(max_occlusion=30, frame_rate=VIDEO_FPS)

    # Seed 5 detections so the track is confirmed; pass frames to exercise the BoT-SORT path.
    seed_id = None
    for frame_index in range(5):
        frame = np.zeros((120, 200, 3), dtype=np.uint8)
        tracks, events = tracker.update(
            [[20 + frame_index, 30, 40 + frame_index, 70, 0.95]], frame_index, frame
        )
        seed_id = tracks[0]["track_id"]

    # Drop exactly 30 frames.
    for frame_index in range(5, 35):
        frame = np.zeros((120, 200, 3), dtype=np.uint8)
        tracks, events = tracker.update([], frame_index, frame)
        assert [track["track_id"] for track in tracks] == [seed_id], f"track lost at frame {frame_index}"
        assert events == [], f"unexpected event at frame {frame_index}: {events}"
        assert tracks[0]["state"] == "coasting", f"track should be coasting at frame {frame_index}"

    # Re-acquire on frame 35; the tracker should preserve the original ID.
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    tracks, events = tracker.update([[55, 30, 75, 70, 0.95]], 35, frame)
    assert [track["track_id"] for track in tracks] == [seed_id], "track ID changed after re-acquisition"
    assert tracks[0]["state"] == "active", "track should be active after re-detection"
    assert events == [], "unexpected lifecycle event on re-acquisition"

    return {
        "method": "Synthetic person moves right 5 frames, all detections dropped for frames 5-34, then re-detected at frame 35",
        "seed_frames": 5,
        "dropout_frames": 30,
        "seed_track_id": int(seed_id),
        "passed": True,
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def main() -> dict[str, Any]:
    if verify_models() != 0:
        raise SystemExit("Model integrity verification failed; inference refused")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    attempted_connections: list[str] = []
    original_connect = socket.socket.connect
    original_create_connection = socket.create_connection

    def blocked_connect(sock, address):
        attempted_connections.append(str(address))
        raise RuntimeError(f"Outbound network blocked during Gate B evaluation: {address}")

    def blocked_create_connection(address, *args, **kwargs):
        attempted_connections.append(str(address))
        raise RuntimeError(f"Outbound network blocked during Gate B evaluation: {address}")

    socket.socket.connect = blocked_connect
    socket.create_connection = blocked_create_connection

    try:
        detector = PersonDetector(device="cpu", weights_dir=str(ROOT / "models"))
        tracker = PersonTracker(max_occlusion=30, frame_rate=VIDEO_FPS)

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

        if len(frames) != 300 or (width, height) != (1280, 720):
            raise RuntimeError(
                f"Unexpected video properties: decoded={len(frames)}, metadata={frame_count_meta}, size={width}x{height}"
            )

        # Warm-up to avoid timing the cold model load.
        for frame in frames[:10]:
            detector.detect(frame)

        # Timed detection + tracking pipeline.
        timed_start = time.perf_counter()
        frame_records: list[dict[str, Any]] = []
        all_lifecycle_events: list[dict[str, Any]] = []
        for frame_index, frame in enumerate(frames):
            detections = detector.detect(frame)
            tracks, events = tracker.update(detections, frame_index, frame)
            all_lifecycle_events.extend(events)
            frame_records.append({
                "frame": frame_index,
                "detections": detections,
                "tracks": tracks,
                "events": events,
            })
        elapsed = time.perf_counter() - timed_start
        pipeline_fps = len(frames) / elapsed if elapsed > 0 else 0.0

        # Map tracks to GT frame-by-frame.
        previous_mapping: dict[int, int] = {}
        gt_identity_assignments: dict[int, int | None] = {0: None, 1: None, 2: None}
        identity_first_seen: dict[int, int] = {}
        identity_last_seen: dict[int, int] = {}
        swap_events: list[dict[str, Any]] = []
        unassigned_visible_frames: dict[int, int] = {0: 0, 1: 0, 2: 0}
        mapped_visible_frames: dict[int, int] = {0: 0, 1: 0, 2: 0}
        total_visible_frames: dict[int, int] = {0: 0, 1: 0, 2: 0}
        unmatched_tracks_per_frame: list[set[int]] = []

        for record in frame_records:
            frame_index = record["frame"]
            gt = intended_boxes(frame_index)
            hidden = fully_hidden_indices(frame_index)
            tracks = record["tracks"]

            mapping = match_tracks_to_ground_truth(tracks, gt, hidden)
            unmatched_tracks = {track["track_id"] for track in tracks} - set(mapping.values())
            unmatched_tracks_per_frame.append(unmatched_tracks)

            for gt_index in range(3):
                if gt_index in hidden:
                    continue
                total_visible_frames[gt_index] += 1
                if gt_index in mapping:
                    mapped_visible_frames[gt_index] += 1
                    tid = mapping[gt_index]
                    if gt_index not in identity_first_seen:
                        identity_first_seen[gt_index] = frame_index
                    identity_last_seen[gt_index] = frame_index

                    prev = previous_mapping.get(gt_index)
                    if prev is not None and prev != tid:
                        swap_events.append({
                            "frame": frame_index,
                            "gt_index": gt_index,
                            "old_track": int(prev),
                            "new_track": int(tid),
                        })
                    previous_mapping[gt_index] = tid
                else:
                    unassigned_visible_frames[gt_index] += 1

        # Fragmentation: count distinct track IDs ever assigned to each GT identity.
        identity_to_tracks: dict[int, set[int]] = {0: set(), 1: set(), 2: set()}
        for record in frame_records:
            mapping = match_tracks_to_ground_truth(
                record["tracks"],
                intended_boxes(record["frame"]),
                fully_hidden_indices(record["frame"]),
            )
            for gt_index, tid in mapping.items():
                identity_to_tracks[gt_index].add(tid)

        fragmentations = [
            {"gt_index": i, "distinct_tracks": len(identity_to_tracks[i])}
            for i in range(3)
        ]

        # Annotated evidence video: every frame at 10 FPS (matches source).
        writer = cv2.VideoWriter(
            str(OUTPUT_DIR / "annotated_tracking.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            VIDEO_FPS,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("Could not create annotated tracking evidence video")

        contact_frames: list[np.ndarray] = []
        per_frame_details: list[dict[str, Any]] = []
        for record in frame_records:
            frame_index = record["frame"]
            gt = intended_boxes(frame_index)
            hidden = fully_hidden_indices(frame_index)

            # Recompute mapping for annotation.
            mapping = match_tracks_to_ground_truth(record["tracks"], gt, hidden)
            frame_swaps = [s for s in swap_events if s["frame"] == frame_index]

            annotated = annotate_frame(
                frames[frame_index],
                frame_index,
                record["tracks"],
                gt,
                hidden,
                mapping,
                frame_swaps,
                unmatched_tracks_per_frame[frame_index],
            )
            writer.write(annotated)
            if frame_index % 10 == 0:
                cv2.imwrite(str(FRAMES_DIR / f"frame_{frame_index:03d}.jpg"), annotated)
                contact_frames.append(cv2.resize(annotated, (426, 240)))

            per_frame_details.append({
                "frame": frame_index,
                "num_detections": len(record["detections"]),
                "num_tracks": len(record["tracks"]),
                "gt_mapping": {str(k): v for k, v in mapping.items()},
                "hidden_gt": sorted(hidden),
                "events": record["events"],
            })

        writer.release()

        # Contact sheet.
        if contact_frames:
            rows = [np.hstack(contact_frames[i:i + 3]) for i in range(0, len(contact_frames), 3)]
            if rows:
                cv2.imwrite(str(OUTPUT_DIR / "contact_sheet.jpg"), np.vstack(rows))

        # Dropout verification.
        dropout_result = verify_30_frame_dropout()

        # Birth/death summary.
        births = [e for e in all_lifecycle_events if e["event"] == "track_birth"]
        deaths = [e for e in all_lifecycle_events if e["event"] == "track_death"]

        # Final verdict.
        total_swaps = len(swap_events)
        t3_no_swaps = total_swaps == 0
        t3_dropout_survives = dropout_result["passed"]

        gate_b_pass = (
            t3_no_swaps
            and t3_dropout_survives
            and len(identity_to_tracks[0]) == 1
            and len(identity_to_tracks[1]) == 1
            and len(identity_to_tracks[2]) == 1
        )

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
                "detector": "detect.PersonDetector(device='cpu', weights_dir='models')",
                "tracker": f"track.PersonTracker(max_occlusion=30, frame_rate={VIDEO_FPS})",
                "gt_mapping": "Hungarian center-distance matching between visible 300x300 synthetic sprites and active/coasting track bounding boxes; threshold 200 px",
                "swap_definition": "A GT identity is assigned a different track ID than in the previous frame where that GT was visible",
                "fragmentation_definition": "Number of distinct track IDs ever assigned to a single synthetic identity",
            },
            "performance": {
                "timed_frames": len(frames),
                "elapsed_seconds": elapsed,
                "pipeline_fps": pipeline_fps,
                "decode_excluded": True,
                "render_excluded": True,
            },
            "tracking_metrics": {
                "total_births": len(births),
                "total_deaths": len(deaths),
                "birth_events": births,
                "death_events": deaths,
                "id_swaps": swap_events,
                "total_id_swaps": total_swaps,
                "fragmentation": fragmentations,
                "identity_to_tracks": {str(k): sorted(v) for k, v in identity_to_tracks.items()},
                "mapped_visible_frames": mapped_visible_frames,
                "unassigned_visible_frames": unassigned_visible_frames,
                "total_visible_frames": total_visible_frames,
                "coverage_ratio": {
                    str(i): mapped_visible_frames[i] / total_visible_frames[i] if total_visible_frames[i] else 0.0
                    for i in range(3)
                },
                "first_seen_frame": identity_first_seen,
                "last_seen_frame": identity_last_seen,
            },
            "t3_dropout_verification": dropout_result,
            "runtime": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "device": "cpu",
                "model": "models/yolo11n.pt",
                "outbound_connections_attempted": attempted_connections,
            },
            "verdict": {
                "gate_b_pass": gate_b_pass,
                "t3_no_swaps": t3_no_swaps,
                "t3_dropout_survives": t3_dropout_survives,
                "kill_criteria": [] if gate_b_pass else ["Gate B did not pass without reproducible evidence and exact metrics"],
            },
            "per_frame_details": per_frame_details,
        }

        (OUTPUT_DIR / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return report
    finally:
        socket.socket.connect = original_connect
        socket.create_connection = original_create_connection


if __name__ == "__main__":
    main()
