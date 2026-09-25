"""Persistent person tracking with an offline, appearance-assisted BoT-SORT backend."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class TrackingBackend(Protocol):
    """Minimal interface required from a swappable tracking backend."""

    def update(self, detections: np.ndarray, frame: np.ndarray | None = None) -> np.ndarray: ...


@dataclass
class _TrackState:
    bbox: list[float]
    born_frame: int
    last_seen_frame: int


class _LocalColorReID:
    """Small deterministic ReID encoder which never loads weights or uses I/O.

    Spatial colour histograms work particularly well for clothing/sprite colour
    while retaining enough layout information to disambiguate similar colours.
    BoT-SORT normalises the returned vectors before association.
    """

    def get_features(self, boxes: np.ndarray, image: np.ndarray) -> np.ndarray:
        features: list[np.ndarray] = []
        height, width = image.shape[:2]
        for box in np.asarray(boxes):
            x1, y1, x2, y2 = box[:4]
            left, top = max(0, int(x1)), max(0, int(y1))
            right, bottom = min(width, int(math.ceil(x2))), min(height, int(math.ceil(y2)))
            crop = image[top:bottom, left:right]
            parts: list[np.ndarray] = []
            if crop.size:
                # Eight spatial cells reduce corruption when people partially overlap.
                for row in np.array_split(crop, 4, axis=0):
                    for cell in np.array_split(row, 2, axis=1):
                        hist = np.histogramdd(
                            cell.reshape(-1, 3), bins=(4, 4, 4), range=((0, 256),) * 3
                        )[0].astype(np.float32).ravel()
                        total = float(hist.sum())
                        parts.append(hist / total if total else hist)
            feature = np.concatenate(parts) if parts else np.zeros(512, dtype=np.float32)
            # A non-zero sentinel keeps even an empty/clipped crop valid for
            # BoxMOT's mandatory embedding normalisation.
            features.append(np.concatenate((feature, np.ones(1, dtype=np.float32))))
        return np.asarray(features, dtype=np.float32).reshape((len(boxes), 513))


class PersonTracker:
    """Adapt BoxMOT BoT-SORT to the project's JSON-compatible contract.

    The default backend uses a local handcrafted appearance encoder; it cannot
    download weights. ``update`` returns ``(tracks, events)``. A track remains
    ``"coasting"`` for up to ``max_occlusion`` consecutive missed frames.
    """

    def __init__(
        self,
        *,
        max_occlusion: int = 30,
        frame_rate: int = 30,
        backend: TrackingBackend | None = None,
    ) -> None:
        if not isinstance(max_occlusion, int) or isinstance(max_occlusion, bool) or max_occlusion < 0:
            raise ValueError("max_occlusion must be a non-negative integer")
        if not isinstance(frame_rate, int) or isinstance(frame_rate, bool) or frame_rate <= 0:
            raise ValueError("frame_rate must be a positive integer")
        self._uses_local_reid = backend is None
        self._backend = backend
        self.max_occlusion = max_occlusion
        self.frame_rate = frame_rate
        self._tracks: dict[int, _TrackState] = {}
        self._last_frame: int | None = None

    def update(
        self,
        detections: list[list[float]],
        frame_index: int,
        frame: np.ndarray | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Process one frame and return current tracks and lifecycle events."""
        self._validate_frame_index(frame_index)
        array = _prepare_detections(detections)
        if self._backend is None:
            self._backend = self._make_backend(use_appearance=frame is not None)
        raw = np.asarray(self._backend.update(array, frame), dtype=np.float64)
        if raw.size == 0:
            raw = np.empty((0, 8), dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] < 5:
            raise RuntimeError("tracking backend returned an invalid result shape")

        events: list[dict[str, Any]] = []
        seen: set[int] = set()
        for row in raw:
            track_id = int(row[4])
            bbox = [float(value) for value in row[:4]]
            if track_id not in self._tracks:
                self._tracks[track_id] = _TrackState(bbox, frame_index, frame_index)
                events.append({"event": "track_birth", "frame": frame_index, "track_id": track_id})
            else:
                state = self._tracks[track_id]
                state.bbox = bbox
                state.last_seen_frame = frame_index
            seen.add(track_id)

        expired = [
            track_id
            for track_id, state in self._tracks.items()
            if track_id not in seen and frame_index - state.last_seen_frame > self.max_occlusion
        ]
        for track_id in sorted(expired):
            del self._tracks[track_id]
            events.append({"event": "track_death", "frame": frame_index, "track_id": track_id})

        tracks = [
            {
                "track_id": track_id,
                "bbox": list(state.bbox),
                "age": frame_index - state.born_frame + 1,
                "state": "active" if track_id in seen else "coasting",
            }
            for track_id, state in sorted(self._tracks.items())
        ]
        self._last_frame = frame_index
        return tracks, events

    def _make_backend(self, *, use_appearance: bool) -> TrackingBackend:
        track_buffer = math.ceil(self.max_occlusion * 30 / self.frame_rate)
        if use_appearance:
            from boxmot import BotSort

            return BotSort(
                track_buffer=track_buffer,
                frame_rate=self.frame_rate,
                reid_model=_LocalColorReID(),
                use_embeddings=True,
                use_cmc=False,
                proximity_thresh=0.7,
                appearance_thresh=0.4,
            )
        from boxmot import ByteTrack

        return ByteTrack(track_buffer=track_buffer, frame_rate=self.frame_rate)

    def reset(self) -> None:
        """Clear wrapper and backend state."""
        reset = getattr(self._backend, "reset", None)
        if callable(reset):
            reset()
        if self._uses_local_reid:
            self._backend = None
        self._tracks.clear()
        self._last_frame = None

    def _validate_frame_index(self, frame_index: int) -> None:
        if not isinstance(frame_index, int) or isinstance(frame_index, bool) or frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        if self._last_frame is not None and frame_index != self._last_frame + 1:
            raise ValueError("frame_index must increase by exactly one")


Tracker = PersonTracker


def _prepare_detections(detections: list[list[float]]) -> np.ndarray:
    if not isinstance(detections, list):
        raise TypeError("detections must be a list")
    rows: list[list[float]] = []
    for detection in detections:
        if not isinstance(detection, (list, tuple)) or len(detection) != 5:
            raise ValueError("each detection must contain [x1, y1, x2, y2, conf]")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in detection):
            raise TypeError("detection values must be numeric")
        values = [float(value) for value in detection]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("detection values must be finite")
        x1, y1, x2, y2, confidence = values
        if x2 < x1 or y2 < y1:
            raise ValueError("detection coordinates must be ordered")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("detection confidence must be between 0 and 1")
        rows.append([x1, y1, x2, y2, confidence, 0.0])
    return np.asarray(rows, dtype=np.float32).reshape((-1, 6))
