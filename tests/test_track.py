"""Deterministic offline tests for the tracking boundary."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from track import PersonTracker, _LocalColorReID, _prepare_detections


class FakeBackend:
    def __init__(self, outputs: list[list[list[float]]]) -> None:
        self.outputs = outputs
        self.inputs: list[np.ndarray] = []

    def update(self, detections: np.ndarray, frame: np.ndarray | None = None) -> np.ndarray:
        self.inputs.append(detections.copy())
        return np.asarray(self.outputs.pop(0), dtype=np.float32).reshape((-1, 8))


def test_input_adapter_and_json_output_contract() -> None:
    backend = FakeBackend([[[1, 2, 11, 22, 7, 0.9, 0, 0]]])
    tracker = PersonTracker(backend=backend)

    tracks, events = tracker.update([[1, 2, 11, 22, 0.9]], 0)

    assert backend.inputs[0].shape == (1, 6)
    assert backend.inputs[0][0].tolist() == pytest.approx([1, 2, 11, 22, 0.9, 0])
    assert tracks == [{"track_id": 7, "bbox": [1.0, 2.0, 11.0, 22.0], "age": 1, "state": "active"}]
    assert events == [{"event": "track_birth", "frame": 0, "track_id": 7}]
    json.dumps({"tracks": tracks, "events": events})


def test_lifecycle_survives_30_misses_then_dies() -> None:
    backend = FakeBackend(
        [[[0, 0, 10, 20, 3, 0.9, 0, 0]]] + [[] for _ in range(31)]
    )
    tracker = PersonTracker(backend=backend)
    tracks, events = tracker.update([[0, 0, 10, 20, 0.9]], 0)
    assert events[0]["event"] == "track_birth"

    for frame_index in range(1, 31):
        tracks, events = tracker.update([], frame_index)
        assert tracks[0]["track_id"] == 3
        assert tracks[0]["state"] == "coasting"
        assert events == []

    tracks, events = tracker.update([], 31)
    assert tracks == []
    assert events == [{"event": "track_death", "frame": 31, "track_id": 3}]


def test_real_bytetrack_reacquires_after_full_30_frame_dropout() -> None:
    tracker = PersonTracker()
    track_id = None
    for frame_index in range(5):
        tracks, _ = tracker.update([[20 + frame_index, 30, 40 + frame_index, 70, 0.95]], frame_index)
        track_id = tracks[0]["track_id"]

    for frame_index in range(5, 35):
        tracks, events = tracker.update([], frame_index)
        assert [track["track_id"] for track in tracks] == [track_id]
        assert events == []

    tracks, events = tracker.update([[55, 30, 75, 70, 0.95]], 35)
    assert [track["track_id"] for track in tracks] == [track_id]
    assert tracks[0]["state"] == "active"
    assert events == []


def test_two_scripted_crossings_have_no_id_swaps_or_fragmentation() -> None:
    tracker = PersonTracker()
    identity_ids: dict[str, int] = {}
    births = 0

    for frame_index in range(301):
        phase = frame_index if frame_index <= 150 else 300 - frame_index
        ax = 20.0 + (4.0 / 3.0) * phase
        bx = 220.0 - (4.0 / 3.0) * phase
        detections = [
            [ax, 30.0, ax + 24.0, 82.0, 0.96],
            [bx, 42.0, bx + 24.0, 94.0, 0.95],
        ]
        tracks, events = tracker.update(detections, frame_index)
        births += sum(event["event"] == "track_birth" for event in events)
        assert len(tracks) == 2

        expected = {"a": np.array([ax + 12.0, 56.0]), "b": np.array([bx + 12.0, 68.0])}
        assigned: dict[str, int] = {}
        for identity, center in expected.items():
            nearest = min(
                tracks,
                key=lambda track: np.linalg.norm(
                    np.array([(track["bbox"][0] + track["bbox"][2]) / 2, (track["bbox"][1] + track["bbox"][3]) / 2])
                    - center
                ),
            )
            assigned[identity] = nearest["track_id"]
        assert assigned["a"] != assigned["b"]
        if not identity_ids:
            identity_ids = assigned
        assert assigned == identity_ids

    assert births == 2


@pytest.mark.parametrize(
    "detections",
    [None, [[0, 0, 1, 1]], [[2, 0, 1, 1, 0.9]], [[0, 0, 1, 1, float("nan")]]],
)
def test_invalid_detections_are_rejected(detections: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _prepare_detections(detections)  # type: ignore[arg-type]


def test_frame_indices_must_be_contiguous() -> None:
    tracker = PersonTracker(backend=FakeBackend([[]]))
    tracker.update([], 4)
    with pytest.raises(ValueError, match="exactly one"):
        tracker.update([], 6)


def test_local_reid_distinguishes_appearance_without_weights() -> None:
    image = np.zeros((20, 40, 3), dtype=np.uint8)
    image[:, :20] = (255, 0, 0)
    image[:, 20:] = (0, 0, 255)
    features = _LocalColorReID().get_features(
        np.asarray([[0, 0, 20, 20], [20, 0, 40, 20]], dtype=np.float32), image
    )

    assert features.shape == (2, 513)
    assert np.isfinite(features).all()
    assert not np.array_equal(features[0], features[1])


def test_frame_selects_appearance_assisted_botsort() -> None:
    tracker = PersonTracker()
    frame = np.zeros((30, 30, 3), dtype=np.uint8)
    tracker.update([[2, 2, 20, 25, 0.95]], 0, frame)

    assert type(tracker._backend).__name__ == "BotSort"
    assert tracker._backend.use_embeddings is True
    assert isinstance(tracker._backend._reid_model, _LocalColorReID)
