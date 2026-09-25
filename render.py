"""Annotated video renderer — RENDER AGENT "Set Dresser".

Renders per-frame overlays for tracked/identified people and writes the
result to an H.264 MP4 file.  This module is intentionally decoupled from
the tracker, recogniser, and web panel: it consumes plain JSON-like dicts
and produces plain BGR frames.

Output contract
---------------
* ``annotate_frame(frame, tracks, recognitions, display_config)`` returns a
  new annotated BGR ``ndarray`` without mutating the input frame.
* ``VideoWriter(path, fps, frame_size)`` writes a sequence of annotated
  frames to ``outputs/output.mp4`` (default) while preserving the source
  resolution, frame rate and frame count.

Visual rules
------------
* Known identity: green box, label ``"DISPLAY NAME — 0.87"``.
* Unknown identity: grey box, label ``"ID: <track_id>"``.
* A track that is new this frame (``age <= 1``) flashes yellow for that one
  frame, overriding the known/unknown colour.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

# Candidate H.264 / fallback codecs in preference order.
_CODECS = ("avc1", "H264", "X264", "mp4v")

DEFAULT_OUTPUT_PATH = Path("outputs") / "output.mp4"

# BGR palette
COLOR_KNOWN = (0, 200, 0)        # green
COLOR_UNKNOWN = (128, 128, 128)  # grey
COLOR_NEW = (0, 220, 220)        # yellow flash
COLOR_TEXT = (255, 255, 255)     # white
UNKNOWN_IDENTITY = "UNKNOWN"

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.8
_FONT_THICKNESS = 2
_BOX_THICKNESS = 2
_LABEL_PAD = 4


def _resolve_display_name(
    identity: str,
    display_config: Mapping[str, Any] | None,
) -> str:
    """Return the configured display name for ``identity`` or ``identity`` itself."""
    display_config = display_config or {}
    if not isinstance(display_config, dict):
        return identity

    gallery = display_config.get("gallery")
    if isinstance(gallery, dict) and identity in gallery:
        meta = gallery[identity]
        if isinstance(meta, dict):
            name = meta.get("display_name")
            if isinstance(name, str) and name.strip():
                return name.strip()

    # Also accept a flat mapping of identity -> display name.
    direct = display_config.get(identity)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    if isinstance(direct, dict):
        name = direct.get("display_name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return identity


def _join_recognitions(
    recognitions: Sequence[Mapping[str, Any]],
) -> dict[int, Mapping[str, Any]]:
    """Index recognition results by ``track_id``; raise on duplicates."""
    joined: dict[int, Mapping[str, Any]] = {}
    for rec in recognitions:
        track_id = int(rec["track_id"])
        if track_id in joined:
            raise ValueError(f"duplicate recognition for track_id {track_id}")
        joined[track_id] = rec
    return joined


def _make_label(
    track: Mapping[str, Any],
    recognition: Mapping[str, Any] | None,
    display_config: Mapping[str, Any] | None,
) -> str:
    """Build the text label for a track."""
    if recognition is not None:
        identity = recognition.get("identity") or UNKNOWN_IDENTITY
        confidence = float(recognition.get("confidence", 0.0))
    else:
        identity = UNKNOWN_IDENTITY
        confidence = 0.0

    if identity and identity != UNKNOWN_IDENTITY:
        display_name = _resolve_display_name(identity, display_config)
        return f"{display_name} — {confidence:.2f}"

    return f"ID: {int(track.get('track_id', -1))}"


def _choose_color(
    track: Mapping[str, Any],
    recognition: Mapping[str, Any] | None,
) -> tuple[int, int, int]:
    """Pick the overlay colour for a track."""
    age = int(track.get("age", 1))
    if age <= 1:
        return COLOR_NEW

    identity = recognition.get("identity") if recognition else UNKNOWN_IDENTITY
    if identity and identity != UNKNOWN_IDENTITY:
        return COLOR_KNOWN
    return COLOR_UNKNOWN


def _clamp_box(
    bbox: Sequence[float],
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    """Clamp and validate a bounding box to image bounds."""
    x1 = int(round(float(bbox[0])))
    y1 = int(round(float(bbox[1])))
    x2 = int(round(float(bbox[2])))
    y2 = int(round(float(bbox[3])))
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def annotate_frame(
    frame: np.ndarray,
    tracks: Sequence[Mapping[str, Any]],
    recognitions: Sequence[Mapping[str, Any]],
    display_config: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Return a new annotated BGR frame.

    Parameters
    ----------
    frame : np.ndarray
        Source BGR image, shape ``(H, W, 3)``.
    tracks : sequence of mappings
        Each item must contain ``track_id``, ``bbox`` and ``age`` keys.
    recognitions : sequence of mappings
        Each item must contain ``track_id``, ``identity`` and ``confidence``.
    display_config : mapping, optional
        ``config.yaml`` contents or a flat ``identity -> display_name`` map.

    Returns
    -------
    np.ndarray
        New annotated BGR frame; the input ``frame`` is left untouched.
    """
    if not isinstance(frame, np.ndarray):
        raise TypeError("frame must be a numpy ndarray")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be a 3-channel BGR image (H, W, 3)")

    annotated = frame.copy()
    height, width = annotated.shape[:2]
    rec_map = _join_recognitions(recognitions)

    for track in tracks:
        track_id = int(track["track_id"])
        bbox = _clamp_box(track["bbox"], width, height)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        recognition = rec_map.get(track_id)
        color = _choose_color(track, recognition)
        label = _make_label(track, recognition, display_config)

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            color,
            _BOX_THICKNESS,
            cv2.LINE_AA,
        )

        (text_w, text_h), _ = cv2.getTextSize(
            label, _FONT, _FONT_SCALE, _FONT_THICKNESS
        )
        label_h = text_h + _LABEL_PAD * 2
        label_w = text_w + _LABEL_PAD * 2

        # Prefer above the box; drop below if it would clip the top edge.
        if y1 - label_h >= 0:
            bg_y1 = y1 - label_h
            bg_y2 = y1
            text_y = y1 - _LABEL_PAD
        else:
            bg_y1 = y2
            bg_y2 = y2 + label_h
            text_y = y2 + label_h - _LABEL_PAD

        bg_x1 = x1
        bg_x2 = min(x1 + label_w, width)

        cv2.rectangle(annotated, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
        cv2.putText(
            annotated,
            label,
            (bg_x1 + _LABEL_PAD, text_y),
            _FONT,
            _FONT_SCALE,
            COLOR_TEXT,
            _FONT_THICKNESS,
            cv2.LINE_AA,
        )

    return annotated


class VideoWriter:
    """OpenCV-backed MP4 writer that prefers H.264 and preserves source timing."""

    def __init__(
        self,
        path: str | os.PathLike[str] = DEFAULT_OUTPUT_PATH,
        fps: float = 30.0,
        frame_size: tuple[int, int] = (1280, 720),
        *,
        codec: str | None = None,
    ) -> None:
        """Create a video writer.

        Parameters
        ----------
        path
            Destination file.  Parent directories are created automatically.
        fps
            Frames per second; must be positive.
        frame_size
            ``(width, height)`` of every written frame.
        codec
            Optional FourCC code.  When omitted H.264 is attempted first,
            falling back to ``mp4v``.
        """
        self.path = Path(path)
        self.fps = float(fps)
        if len(frame_size) != 2:
            raise ValueError("frame_size must be a (width, height) tuple")
        self.frame_size = (int(frame_size[0]), int(frame_size[1]))

        if self.frame_size[0] <= 0 or self.frame_size[1] <= 0:
            raise ValueError("frame_size width and height must be positive")
        if self.fps <= 0.0 or not math.isfinite(self.fps):
            raise ValueError("fps must be a positive finite number")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = self._open_writer(codec)
        self._frame_count = 0

    def _open_writer(self, codec: str | None) -> cv2.VideoWriter:
        """Try codecs in order and return the first usable writer."""
        if codec and codec not in _CODECS:
            raise RuntimeError(
                f"unsupported video codec {codec!r}; supported: {_CODECS}"
            )
        candidates: list[str] = []
        if codec:
            candidates.append(codec)
        candidates.extend(c for c in _CODECS if c != codec)

        last_error: Exception | None = None
        for candidate in candidates:
            try:
                fourcc = cv2.VideoWriter_fourcc(*candidate)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
            writer = cv2.VideoWriter(
                str(self.path),
                fourcc,
                self.fps,
                self.frame_size,
            )
            if writer.isOpened():
                self._codec = candidate
                return writer
            writer.release()

        if last_error is None:
            raise RuntimeError(f"no usable video codec found; tried {candidates}")
        raise RuntimeError(
            f"no usable video codec found; tried {candidates}: {last_error}"
        )

    @property
    def codec(self) -> str:
        """FourCC string selected for this writer."""
        return self._codec

    @property
    def frame_count(self) -> int:
        """Number of frames written so far."""
        return self._frame_count

    def write_frame(self, frame: np.ndarray) -> None:
        """Write one BGR frame.

        Raises
        ------
        TypeError
            If ``frame`` is not an ndarray.
        ValueError
            If the frame shape does not match ``frame_size``.
        """
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a numpy ndarray")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a 3-channel BGR image")
        if (frame.shape[1], frame.shape[0]) != self.frame_size:
            raise ValueError(
                f"frame size {(frame.shape[1], frame.shape[0])} does not match "
                f"writer frame_size {self.frame_size}"
            )
        self._writer.write(frame)
        self._frame_count += 1

    def finalize(self) -> Path:
        """Release the writer and return the final file path."""
        self._writer.release()
        return self.path

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.finalize()
