"""Unit tests for the render module."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import cv2
import numpy as np
import pytest

import render


@pytest.fixture
def black_720p() -> np.ndarray:
    return np.zeros((720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def display_config() -> dict[str, Any]:
    return {
        "gallery": {
            "alex": {"display_name": "Alex Synthetic"},
            "blair": {"display_name": "Blair Synthetic"},
        }
    }


def _make_frame(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_annotate_returns_new_frame(black_720p: np.ndarray) -> None:
    original = black_720p.copy()
    out = render.annotate_frame(black_720p, [], [])
    assert out is not black_720p
    assert np.array_equal(black_720p, original)


def test_annotate_exact_track_id_join(display_config: dict[str, Any]) -> None:
    frame = _make_frame()
    tracks = [
        {"track_id": 1, "bbox": [50.0, 50.0, 150.0, 150.0], "age": 5},
        {"track_id": 2, "bbox": [200.0, 200.0, 300.0, 300.0], "age": 5},
    ]
    recognitions = [
        {"track_id": 1, "identity": "alex", "confidence": 0.87},
        {"track_id": 2, "identity": "UNKNOWN", "confidence": 0.0},
    ]
    captured: list[tuple[np.ndarray, str]] = []

    def fake_putText(img: np.ndarray, text: str, *_: Any, **__: Any) -> None:
        captured.append((img, text))

    with patch("cv2.putText", side_effect=fake_putText):
        render.annotate_frame(frame, tracks, recognitions, display_config)

    labels = {text for _, text in captured}
    assert "Alex Synthetic — 0.87" in labels
    assert "ID: 2" in labels


def test_annotate_duplicate_recognition_raises() -> None:
    frame = _make_frame()
    tracks = [{"track_id": 1, "bbox": [10, 10, 50, 50], "age": 2}]
    recognitions = [
        {"track_id": 1, "identity": "alex", "confidence": 0.9},
        {"track_id": 1, "identity": "blair", "confidence": 0.8},
    ]
    with pytest.raises(ValueError, match="duplicate recognition for track_id 1"):
        render.annotate_frame(frame, tracks, recognitions)


def test_annotate_missing_recognition_is_unknown() -> None:
    frame = _make_frame()
    tracks = [{"track_id": 7, "bbox": [20, 20, 80, 80], "age": 3}]
    captured: list[str] = []

    with patch("cv2.putText", side_effect=lambda img, text, *_a, **_k: captured.append(text)):
        render.annotate_frame(frame, tracks, [])

    assert captured == ["ID: 7"]


def test_annotate_colors() -> None:
    frame = _make_frame()
    tracks = [
        {"track_id": 1, "bbox": [50, 50, 150, 150], "age": 1},
        {"track_id": 2, "bbox": [200, 200, 300, 300], "age": 5},
        {"track_id": 3, "bbox": [350, 350, 450, 450], "age": 5},
    ]
    recognitions = [
        {"track_id": 2, "identity": "alex", "confidence": 0.9},
        {"track_id": 3, "identity": "UNKNOWN", "confidence": 0.0},
    ]
    out = render.annotate_frame(frame, tracks, recognitions)

    # Sample the filled label background just inside its rectangle.
    # For the first track (new/yellow), the label is above the box.
    assert np.array_equal(out[48, 52], np.array(render.COLOR_NEW))
    # Known green
    assert np.array_equal(out[186, 202], np.array(render.COLOR_KNOWN))
    # Unknown grey
    assert np.array_equal(out[336, 352], np.array(render.COLOR_UNKNOWN))


def test_annotate_invalid_frame_shape() -> None:
    with pytest.raises(ValueError, match="3-channel BGR image"):
        render.annotate_frame(np.zeros((100, 100), dtype=np.uint8), [], [])
    with pytest.raises(ValueError, match="3-channel BGR image"):
        render.annotate_frame(np.zeros((100, 100, 4), dtype=np.uint8), [], [])
    with pytest.raises(TypeError, match="numpy ndarray"):
        render.annotate_frame("not an array", [], [])  # type: ignore[arg-type]


def test_resolve_display_name(display_config: dict[str, Any]) -> None:
    assert render._resolve_display_name("alex", display_config) == "Alex Synthetic"
    assert render._resolve_display_name("casey", display_config) == "casey"
    assert render._resolve_display_name("casey", {}) == "casey"
    flat = {"alex": "Alex Flat"}
    assert render._resolve_display_name("alex", flat) == "Alex Flat"


def test_video_writer_invalid_codec(tmp_path: Path) -> None:
    out = tmp_path / "out.mp4"
    with pytest.raises(RuntimeError, match="unsupported video codec"):
        render.VideoWriter(out, fps=30.0, frame_size=(640, 480), codec="FAIL")


def test_video_writer_bad_frame_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        render.VideoWriter(frame_size=(0, 480))
    with pytest.raises(ValueError, match="positive"):
        render.VideoWriter(frame_size=(640, -1))
    with pytest.raises(ValueError, match="frame_size"):
        render.VideoWriter(frame_size=(640,))


def test_video_writer_frame_size_mismatch(tmp_path: Path) -> None:
    out = tmp_path / "out.mp4"
    writer = render.VideoWriter(out, fps=30.0, frame_size=(640, 480))
    bad = np.zeros((481, 640, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="does not match"):
        writer.write_frame(bad)
    writer.finalize()


def test_video_writer_bad_frame_channels(tmp_path: Path) -> None:
    out = tmp_path / "out.mp4"
    writer = render.VideoWriter(out, fps=30.0, frame_size=(640, 480))
    with pytest.raises(ValueError, match="3-channel BGR"):
        writer.write_frame(np.zeros((480, 640, 1), dtype=np.uint8))
    writer.finalize()


def test_video_writer_preserves_count_resolution_fps(tmp_path: Path) -> None:
    out = tmp_path / "out.mp4"
    fps = 30.0
    size = (640, 480)
    n_frames = 12

    with render.VideoWriter(out, fps=fps, frame_size=size) as writer:
        for i in range(n_frames):
            frame = np.full((size[1], size[0], 3), (i * 20, 0, 0), dtype=np.uint8)
            writer.write_frame(frame)
        assert writer.frame_count == n_frames

    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened()
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == n_frames
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(fps, abs=0.01)
    assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == size[0]
    assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == size[1]

    read_back = 0
    while cap.read()[0]:
        read_back += 1
    assert read_back == n_frames
    cap.release()


def test_output_is_h264_mp4(tmp_path: Path) -> None:
    out = tmp_path / "out.mp4"
    with render.VideoWriter(out, fps=30.0, frame_size=(640, 480)) as writer:
        for _ in range(5):
            writer.write_frame(np.zeros((480, 640, 3), dtype=np.uint8))

    # CV2 may report the canonical H264 FourCC even though we requested avc1.
    cap = cv2.VideoCapture(str(out))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()
    # OpenCV may normalise avc1 to H264 or h264 on read-back; all are H.264.
    assert fourcc in (
        cv2.VideoWriter_fourcc(*"avc1"),
        cv2.VideoWriter_fourcc(*"H264"),
        cv2.VideoWriter_fourcc(*"h264"),
    )

    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        result = subprocess.run(
            [
                ffprobe,
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name,codec_tag_string",
                "-of", "default=noprint_wrappers=1",
                str(out),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "codec_name=h264" in result.stdout
        assert "codec_tag_string=avc1" in result.stdout


def test_video_writer_context_manager(tmp_path: Path) -> None:
    out = tmp_path / "out.mp4"
    with render.VideoWriter(out, fps=30.0, frame_size=(320, 240)) as writer:
        writer.write_frame(np.zeros((240, 320, 3), dtype=np.uint8))
    assert out.exists()


def test_default_output_path() -> None:
    assert render.DEFAULT_OUTPUT_PATH == Path("outputs") / "output.mp4"
