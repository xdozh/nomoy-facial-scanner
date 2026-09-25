#!/usr/bin/env python3
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

os.environ.update({
    "CUDA_VISIBLE_DEVICES": "",
    "HF_HUB_OFFLINE": "1",
    "NO_PROXY": "*",
    "ULTRALYTICS_OFFLINE": "true",
})

import cv2
import numpy as np
import torch
import ultralytics

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detect import PersonDetector
from scripts.verify_models import main as verify_models

SAMPLE_INDICES = tuple(range(0, 300, 10))
IOU_THRESHOLD = 0.50


def _generator_positions(frame_index):
    """Reproduce the exact sprite positions from generate_synthetic_assets.py.

    The v1 evaluator used obsolete linear equations; Gate A v2 uses the
    revised tanh-scripted crossings in the generator.
    """
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


def intended_boxes(frame_index):
    positions, _ = _generator_positions(frame_index)
    return [[float(x), float(y), float(x + 300), float(y + 300)] for x, y in positions]


def fully_hidden_indices(frame_index):
    """Return indices of intended boxes completely covered by later-drawn opaque sprites.

    Only full opaque compositing occlusion is excluded; partial occlusions are
    intentionally retained in the denominator.
    """
    positions, order = _generator_positions(frame_index)
    boxes = [[x, y, x + 300, y + 300] for x, y in positions]
    hidden = set()
    for draw_position, person_index in enumerate(order):
        ax1, ay1, ax2, ay2 = boxes[person_index]
        for later_person in order[draw_position + 1:]:
            bx1, by1, bx2, by2 = boxes[later_person]
            if bx1 <= ax1 and by1 <= ay1 and bx2 >= ax2 and by2 >= ay2:
                hidden.add(person_index)
                break
    return hidden


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection) if area_a + area_b > intersection else 0.0


def match(detections, expected):
    candidates = sorted(
        ((iou(det[:4], gt), det_index, gt_index) for det_index, det in enumerate(detections) for gt_index, gt in enumerate(expected)),
        reverse=True,
    )
    used_detections, used_expected, matches = set(), set(), []
    for overlap, det_index, gt_index in candidates:
        if overlap < IOU_THRESHOLD:
            break
        if det_index not in used_detections and gt_index not in used_expected:
            used_detections.add(det_index)
            used_expected.add(gt_index)
            matches.append((det_index, gt_index, overlap))
    return matches


def annotate(frame, frame_index, detections, expected, matches):
    matched_detections = {item[0] for item in matches}
    matched_expected = {item[1] for item in matches}
    for gt_index, box in enumerate(expected):
        color = (0, 180, 0) if gt_index in matched_expected else (0, 0, 255)
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"GT {gt_index + 1} {'MATCH' if gt_index in matched_expected else 'MISS'}", (x1, max(70, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    for det_index, detection in enumerate(detections):
        x1, y1, x2, y2, confidence = detection
        color = (255, 120, 0) if det_index in matched_detections else (0, 165, 255)
        cv2.rectangle(frame, (round(x1), round(y1)), (round(x2), round(y2)), color, 2)
        cv2.putText(frame, f"DET {confidence:.2f}", (round(x1), max(90, round(y1) - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.rectangle(frame, (0, 0), (1280, 66), (235, 235, 235), -1)
    visible_count = len(expected)
    cv2.putText(frame, f"SYNTHETIC GATE A QA | frame {frame_index} | matched {len(matches)}/{visible_count} | IoU >= {IOU_THRESHOLD:.2f}", (24, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (10, 10, 10), 2)
    cv2.putText(frame, "Green/red: intended GT match/miss; blue/orange: matched/unmatched detection", (24, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 1)
    return frame


def main():
    if verify_models() != 0:
        raise SystemExit("Model integrity verification failed; inference refused")

    output_dir = ROOT / "outputs" / "gate_a"
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    attempted_connections = []
    original_connect = socket.socket.connect
    original_create_connection = socket.create_connection

    def blocked_connect(sock, address):
        attempted_connections.append(str(address))
        raise RuntimeError(f"Outbound network blocked during Gate A evaluation: {address}")

    def blocked_create_connection(address, *args, **kwargs):
        attempted_connections.append(str(address))
        raise RuntimeError(f"Outbound network blocked during Gate A evaluation: {address}")

    socket.socket.connect = blocked_connect
    socket.create_connection = blocked_create_connection
    try:
        detector = PersonDetector(device="cpu", weights_dir=str(ROOT / "models"))
        capture = cv2.VideoCapture(str(ROOT / "samples" / "test.mp4"))
        if not capture.isOpened():
            raise RuntimeError("Could not open samples/test.mp4")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        capture.release()
        if len(frames) != 300 or (width, height) != (1280, 720):
            raise RuntimeError(f"Unexpected video properties: decoded={len(frames)}, metadata={frame_count}, size={width}x{height}")

        for frame in frames[:10]:
            detector.detect(frame)
        start = time.perf_counter()
        detections_by_frame = [detector.detect(frame) for frame in frames]
        elapsed = time.perf_counter() - start
        steady_state_fps = len(frames) / elapsed

        writer = cv2.VideoWriter(str(output_dir / "annotated_samples.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (width, height))
        if not writer.isOpened():
            raise RuntimeError("Could not create annotated evidence video")
        sampled = []
        contact_frames = []
        matched_total = 0
        excluded_total = 0
        for frame_index in SAMPLE_INDICES:
            detections = detections_by_frame[frame_index]
            expected = intended_boxes(frame_index)
            hidden = fully_hidden_indices(frame_index)
            excluded_total += len(hidden)
            # Remove fully-hidden intended boxes from evaluation; partial
            # occlusions remain visible person-frames.
            visible_expected = [box for index, box in enumerate(expected) if index not in hidden]
            matches = match(detections, visible_expected)
            matched_total += len(matches)
            annotated = annotate(frames[frame_index].copy(), frame_index, detections, visible_expected, matches)
            if hidden:
                cv2.putText(annotated, f"EXCLUDED fully hidden: {sorted(hidden)}", (24, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 200), 1)
            cv2.imwrite(str(frames_dir / f"frame_{frame_index:03d}.jpg"), annotated)
            writer.write(annotated)
            contact_frames.append(cv2.resize(annotated, (426, 240)))
            sampled.append({
                "frame": frame_index,
                "intended": 3,
                "excluded": sorted(hidden),
                "visible": len(visible_expected),
                "detections": len(detections),
                "matched": len(matches),
                "matches": [{"detection": d, "intended": g, "iou": round(o, 6)} for d, g, o in matches],
            })
        writer.release()
        rows = [np.hstack(contact_frames[index:index + 3]) for index in range(0, len(contact_frames), 3)]
        cv2.imwrite(str(output_dir / "contact_sheet.jpg"), np.vstack(rows))

        denominator = (len(SAMPLE_INDICES) * 3) - excluded_total
        report = {
            "synthetic_stopgap": True,
            "video": {"path": "samples/test.mp4", "decoded_frames": len(frames), "metadata_frames": frame_count, "width": width, "height": height, "source_fps": source_fps},
            "sampling": {
                "protocol": "Every 10th frame starting at frame 0 and ending at frame 290",
                "indices": list(SAMPLE_INDICES),
                "sampled_frames": len(SAMPLE_INDICES),
                "fully_hidden_excluded": excluded_total,
                "person_frame_denominator": denominator,
                "iou_threshold": IOU_THRESHOLD,
                "methodology": "IoU >= 0.50 greedy one-to-one matching; only fully opaque-composited hidden sprites excluded",
            },
            "recall": {"matched_person_frames": matched_total, "person_frame_denominator": denominator, "value": matched_total / denominator if denominator else 0.0},
            "performance": {"warmup_frames": 10, "timed_frames": len(frames), "elapsed_seconds": elapsed, "steady_state_fps": steady_state_fps, "decode_excluded": True},
            "runtime": {"platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__, "ultralytics": ultralytics.__version__, "device": "cpu", "model": "models/yolo11n.pt", "outbound_connections_attempted": attempted_connections},
            "sample_results": sampled,
        }
        (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        socket.socket.connect = original_connect
        socket.create_connection = original_create_connection


if __name__ == "__main__":
    main()
