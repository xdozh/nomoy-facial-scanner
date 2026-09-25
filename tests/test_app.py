"""Tests for app.py — no public server is launched."""

import os
import sys
from pathlib import Path

import pytest
import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app as app_module
from app import (
    CONSENT_TEXT,
    DISCLAIMER_TEXT,
    MAX_FILE_SIZE_BYTES,
    PipelineResult,
    build_app,
    render_identity_cards,
    require_consent,
    run_pipeline,
    validate_input,
)


def _fake_video(tmp_path: Path) -> str:
    """Create a tiny real video so the cv2 decode check passes."""
    import cv2
    import numpy as np

    p = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
    w.write(np.zeros((64, 64, 3), dtype=np.uint8))
    w.release()
    return str(p)


def _stub_process(video_path, progress_cb):
    progress_cb(0.5, "halfway")
    return {
        "video_path": "/out/output.mp4",
        "run_log_path": "/out/run_log.jsonl",
        "identities": [
            {
                "gallery_photo": None,
                "display_name": "ALEX",
                "confidence": 0.87,
                "first_seen": 1.5,
                "last_seen": 62.0,
                "timestamps_seen": [1.5, 30.0, 62.0],
                "screen_time_seconds": 12.5,
            }
        ],
    }


# --- consent gate -----------------------------------------------------------


@pytest.mark.parametrize("bad", [False, None, 0, "", "yes", 1])
def test_consent_cannot_be_bypassed(bad, tmp_path):
    with pytest.raises(gr.Error):
        run_pipeline(_fake_video(tmp_path), bad, _stub_process)


def test_consent_must_be_literal_true(tmp_path):
    # Only a real checked checkbox (True) opens the gate.
    with pytest.raises(gr.Error):
        require_consent("true")
    require_consent(True)  # no raise


def test_process_fn_not_called_without_consent(tmp_path):
    called = []

    def spy(path, cb):
        called.append(path)

    with pytest.raises(gr.Error):
        run_pipeline(_fake_video(tmp_path), False, spy)
    assert called == []


# --- input validation --------------------------------------------------------


def test_rejects_bad_extension(tmp_path):
    bad = tmp_path / "evil.exe"
    bad.write_bytes(b"x")
    with pytest.raises(ValueError, match="Unsupported file type"):
        validate_input(str(bad))


def test_rejects_missing_file(tmp_path):
    with pytest.raises(ValueError):
        validate_input(str(tmp_path / "nope.mp4"))


def test_rejects_oversize(tmp_path, monkeypatch):
    v = _fake_video(tmp_path)
    monkeypatch.setattr(os.path, "getsize", lambda p: MAX_FILE_SIZE_BYTES + 1)
    with pytest.raises(ValueError, match="500MB"):
        validate_input(v)


def test_accepts_valid_video(tmp_path):
    validate_input(_fake_video(tmp_path))  # no raise


# --- pipeline / output wiring -------------------------------------------------


def test_run_pipeline_returns_normalized_result(tmp_path):
    res = run_pipeline(_fake_video(tmp_path), True, _stub_process)
    assert isinstance(res, PipelineResult)
    assert res.video_path == "/out/output.mp4"
    assert res.run_log_path == "/out/run_log.jsonl"
    assert res.identities[0]["display_name"] == "ALEX"


def test_run_pipeline_accepts_tuple_result(tmp_path):
    res = run_pipeline(
        _fake_video(tmp_path), True, lambda p, cb: ("v.mp4", "l.jsonl", [])
    )
    assert (res.video_path, res.run_log_path, res.identities) == ("v.mp4", "l.jsonl", [])


def test_progress_callback_passed_to_process_fn(tmp_path):
    seen = []

    def fn(path, cb):
        cb(0.25, "q1")
        return ("v", "l", [])

    class FakeProgress:
        def __call__(self, frac, desc=""):
            seen.append((frac, desc))

    run_pipeline(_fake_video(tmp_path), True, fn, FakeProgress())
    assert seen == [(0.25, "q1")]


# --- identity cards -----------------------------------------------------------


def test_cards_render_exact_summary_values(tmp_path):
    photo = tmp_path / "alex.jpg"
    photo.write_bytes(b"\xff\xd8fakejpeg")  # content embedded as base64
    cards = render_identity_cards(
        [
            {
                "gallery_photo": str(photo),
                "display_name": "ALEX",
                "confidence": 0.87,
                "first_seen": 1.5,
                "last_seen": 62.0,
                "timestamps_seen": [1.5, 62.0],
                "screen_time_seconds": 12.5,
            },
            {
                "gallery_photo": None,
                "display_name": "UNKNOWN",
                "confidence": 0.0,
                "first_seen": 5.0,
                "last_seen": 9.0,
                "timestamps_seen": [],
                "screen_time_seconds": 4.0,
            },
        ]
    )
    assert cards.count('class="id-card"') == 2
    assert ">ALEX<" in cards and ">UNKNOWN<" in cards
    assert "0.87" in cards
    assert "00:01.5" in cards and "01:02.0" in cards
    assert "12.5s" in cards
    assert "data:image/jpeg;base64," in cards


def test_cards_empty():
    assert "No identities" in render_identity_cards([])


# --- app wiring / disclaimer ---------------------------------------------------


def test_build_app_no_server_and_disclaimer_present():
    demo = build_app(_stub_process)
    assert isinstance(demo, gr.Blocks)
    # disclaimer + consent text rendered into the Blocks config
    cfg = str(demo.get_config_file())
    for needle in ("enrolled volunteers", "non-commercial", "AGPL", CONSENT_TEXT):
        assert needle in cfg


def test_main_binds_loopback_no_share(monkeypatch):
    launched = {}

    class FakeBlocks:
        def launch(self, **kw):
            launched.update(kw)

    monkeypatch.setattr(app_module, "build_app", lambda *a, **k: FakeBlocks())
    app_module.main()
    assert launched["server_name"] == "127.0.0.1"
    assert launched["share"] is False
    assert launched["max_file_size"] == MAX_FILE_SIZE_BYTES
