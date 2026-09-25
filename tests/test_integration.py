"""Integration tests for the end-to-end demo pipeline.

These tests exercise ``demo.process_video`` and the CLI without launching a
browser. They use temporary directories so they never collide with the demo's
default ``outputs/`` files.
"""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

import cv2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app
import audit
import demo
from demo import process_video


SAMPLES = Path(__file__).resolve().parent.parent / "samples"
TEST_VIDEO = SAMPLES / "test.mp4"
RECOGNITION_VIDEO = SAMPLES / "recognition_test.mp4"


def _clip_video(src: Path, dst: Path, n_frames: int) -> None:
    """Write the first ``n_frames`` of ``src`` to ``dst`` as mp4."""
    cap = cv2.VideoCapture(str(src))
    assert cap.isOpened(), f"could not open {src}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(dst),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    assert writer.isOpened(), "could not open video writer"
    written = 0
    while written < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
    writer.release()
    cap.release()
    assert written == n_frames, f"requested {n_frames} frames, got {written}"


def _read_jsonl(path: Path) -> list[dict]:
    return [dict(audit.json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Headless wiring tests (fast)
# ---------------------------------------------------------------------------


def test_process_video_runs_offline_and_returns_pipeline_result(tmp_path: Path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    _clip_video(TEST_VIDEO, clip, n_frames=15)

    out_video = tmp_path / "out.mp4"
    out_log = tmp_path / "run_log.jsonl"

    progress = []
    logger_options: dict = {}
    real_audit_logger = audit.AuditLogger

    def capturing_audit_logger(*args, **kwargs):
        logger_options.update(kwargs)
        return real_audit_logger(*args, **kwargs)

    monkeypatch.setattr(audit, "AuditLogger", capturing_audit_logger)

    def cb(frac: float, desc: str = "") -> None:
        progress.append((frac, desc))

    with audit.enforce_offline():
        result = process_video(
            clip,
            progress_cb=cb,
            consent=True,
            output_video=out_video,
            output_log=out_log,
        )

    assert isinstance(result, app.PipelineResult)
    assert result.video_path == str(out_video)
    assert result.run_log_path == str(out_log)
    assert out_video.is_file()
    assert out_log.is_file()

    # Verify audit log integrity and contiguous frame completion.
    records = audit.verify_log(out_log)
    assert records[0]["event"] == "startup"
    assert logger_options["save"] is True
    startup_models = records[0]["model_versions"]
    assert startup_models
    assert all(set(details) >= {"version", "sha256"} for details in startup_models.values())
    complete = [r for r in records if r["event"] == "frame_complete"]
    assert len(complete) == 15
    assert [r["frame"] for r in complete] == list(range(15))

    # Output video matches source timing.
    cap = cv2.VideoCapture(str(out_video))
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 15
    cap.release()

    # Progress reached 100%.
    assert progress and progress[-1][0] == pytest.approx(1.0, abs=0.01)


def test_process_video_finds_enrolled_identities_on_test_clip(tmp_path: Path):
    clip = tmp_path / "clip.mp4"
    _clip_video(TEST_VIDEO, clip, n_frames=30)

    out_video = tmp_path / "out.mp4"
    out_log = tmp_path / "run_log.jsonl"

    result = process_video(
        clip,
        consent=True,
        output_video=out_video,
        output_log=out_log,
    )

    names = {p["display_name"] for p in result.identities}
    assert "Alex Synthetic" in names
    assert "Blair Synthetic" in names
    assert "Casey Synthetic" in names

    # Identity summaries are derived from the audit log and span the clip.
    fps = 10.0  # samples/test.mp4 metadata
    duration = 30 / fps
    for person in result.identities:
        assert person["screen_time_seconds"] >= duration * 0.9
        assert person["first_seen"] == pytest.approx(0.0, abs=0.01)
        assert person["last_seen"] == pytest.approx(duration - 1 / fps, abs=0.05)
        assert isinstance(person["timestamps_seen"], list)
        assert len(person["timestamps_seen"]) >= 1


def test_unknown_person_in_recognition_test_clip(tmp_path: Path):
    clip = tmp_path / "clip.mp4"
    _clip_video(RECOGNITION_VIDEO, clip, n_frames=30)

    out_log = tmp_path / "run_log.jsonl"
    result = process_video(
        clip,
        consent=True,
        output_video=tmp_path / "out.mp4",
        output_log=out_log,
    )

    # Only the three enrolled identities should appear as matched names.
    enrolled = {"alex", "blair", "casey"}
    matched = {p["identity"] for p in result.identities}
    assert matched == enrolled

    # Check that the non-enrolled track stayed UNKNOWN in the audit log.
    records = audit.verify_log(out_log)
    match_events = [r for r in records if r["event"] == "match"]
    assert all(r["identity"] in enrolled for r in match_events)


def test_app_run_pipeline_wires_process_video(tmp_path: Path):
    """``app.run_pipeline`` calls process_video with the exact Gradio contract."""
    clip = tmp_path / "clip.mp4"
    _clip_video(TEST_VIDEO, clip, n_frames=5)

    out_video = tmp_path / "out.mp4"
    out_log = tmp_path / "run_log.jsonl"

    progress: list[tuple[float, str]] = []

    class FakeProgress:
        def __call__(self, frac: float, desc: str = "") -> None:
            progress.append((frac, desc))

    def fn(video_path: str, cb):
        return process_video(
            video_path,
            progress_cb=cb,
            consent=True,
            output_video=out_video,
            output_log=out_log,
        )

    result = app.run_pipeline(str(clip), True, fn, progress=FakeProgress())
    assert isinstance(result, app.PipelineResult)
    assert result.video_path == str(out_video)
    assert result.run_log_path == str(out_log)
    assert out_video.is_file()
    assert out_log.is_file()
    assert progress and progress[-1][0] == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Security / contract tests
# ---------------------------------------------------------------------------


def test_consent_required_before_processing(tmp_path: Path):
    clip = tmp_path / "clip.mp4"
    _clip_video(TEST_VIDEO, clip, n_frames=5)

    with pytest.raises((demo.DemoError, audit.ConsentRequiredError)):
        process_video(
            clip,
            consent=False,
            output_video=tmp_path / "out.mp4",
            output_log=tmp_path / "run_log.jsonl",
        )


def test_invalid_video_rejected(tmp_path: Path):
    bad = tmp_path / "bad.exe"
    bad.write_bytes(b"not a video")
    with pytest.raises(demo.DemoError):
        process_video(
            bad,
            consent=True,
            output_video=tmp_path / "out.mp4",
            output_log=tmp_path / "run_log.jsonl",
        )


def test_offline_enforcement_blocks_socket_connect():
    with audit.enforce_offline():
        with pytest.raises(audit.OfflineViolationError):
            socket.create_connection(("127.0.0.1", 9))


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


def test_cli_demo_error_is_clean_and_nonzero(monkeypatch, capsys):
    monkeypatch.setattr(demo, "_model_versions", lambda: {})

    def reject_input(*args, **kwargs):
        raise demo.DemoError("Unsupported extension: .exe")

    monkeypatch.setattr(demo, "process_video", reject_input)
    code = demo.main(["--video", "bad.exe", "--no-browser"])
    captured = capsys.readouterr()

    assert code == 1
    assert captured.err == "Error: Unsupported extension: .exe\n"
    assert "Traceback" not in captured.out + captured.err
    assert 'File "' not in captured.out + captured.err


def test_cli_audit_error_is_clean_and_nonzero(monkeypatch, capsys):
    def reject_models():
        raise audit.ModelVerificationError("checksum mismatch")

    monkeypatch.setattr(demo, "_model_versions", reject_models)
    code = demo.main(["--video", "unused.mp4", "--no-browser"])
    captured = capsys.readouterr()

    assert code == 1
    assert captured.err == "Error: checksum mismatch\n"
    assert "Traceback" not in captured.out + captured.err
    assert 'File "' not in captured.out + captured.err


def test_cli_headless_runs_and_does_not_open_browser(tmp_path: Path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    _clip_video(TEST_VIDEO, clip, n_frames=10)

    out_video = tmp_path / "out.mp4"
    out_log = tmp_path / "run_log.jsonl"

    monkeypatch.setattr(demo, "build_app", lambda **k: (_ for _ in ()).throw(AssertionError("browser launch attempted")))

    code = demo.main([
        "--video", str(clip),
        "--no-browser",
        "--output-video", str(out_video),
        "--output-log", str(out_log),
    ])

    assert code == 0
    assert out_video.is_file()
    assert out_log.is_file()


def test_cli_standard_run_opens_browser_during_gradio_launch(tmp_path: Path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    _clip_video(TEST_VIDEO, clip, n_frames=5)

    launched: dict = {}

    class FakeBlocks:
        def launch(self, **kw):
            launched.update(kw)

    monkeypatch.setattr(demo, "build_app", lambda **k: FakeBlocks())

    code = demo.main([
        "--video", str(clip),
        "--output-video", str(tmp_path / "out.mp4"),
        "--output-log", str(tmp_path / "run_log.jsonl"),
    ])

    assert code == 0
    assert launched.get("server_name") == "127.0.0.1"
    assert launched.get("share") is False
    assert launched.get("inbrowser") is True


# ---------------------------------------------------------------------------
# Determinism / swappability
# ---------------------------------------------------------------------------


def test_default_output_paths_are_deterministic():
    assert demo.DEFAULT_OUTPUT_VIDEO == demo.ROOT / "outputs" / "output.mp4"
    assert demo.DEFAULT_OUTPUT_LOG == demo.ROOT / "outputs" / "run_log.jsonl"


def test_gallery_and_config_swappable_without_code_changes(tmp_path: Path):
    # Copy the real gallery into a temporary directory and use a custom config
    # with a renamed display name. No code change should be required.
    clip = tmp_path / "clip.mp4"
    _clip_video(TEST_VIDEO, clip, n_frames=30)

    custom_gallery = tmp_path / "gallery"
    custom_gallery.mkdir()
    for name in ("alex", "blair", "casey"):
        src = Path(__file__).resolve().parent.parent / "gallery" / f"{name}.jpg"
        (custom_gallery / f"{name}.jpg").write_bytes(src.read_bytes())

    custom_config = tmp_path / "config.yaml"
    custom_config.write_text(
        "gallery:\n"
        "  alex: {display_name: Custom Alex}\n"
        "  blair: {display_name: Custom Blair}\n"
        "  casey: {display_name: Custom Casey}\n"
        "match_threshold: 0.4\n"
        "face_min_size_px: 48\n"
        "embed_every_n_tracks: 1\n"
        "face_check_interval_frames: 15\n"
    )

    result = process_video(
        clip,
        consent=True,
        config_path=custom_config,
        gallery_dir=custom_gallery,
        output_video=tmp_path / "out.mp4",
        output_log=tmp_path / "run_log.jsonl",
    )

    display_names = {p["display_name"] for p in result.identities}
    assert "Custom Alex" in display_names
    assert "Custom Blair" in display_names
    assert "Custom Casey" in display_names


def test_paths_created_safely(tmp_path: Path):
    clip = tmp_path / "clip.mp4"
    _clip_video(TEST_VIDEO, clip, n_frames=5)

    nested_video = tmp_path / "a" / "b" / "out.mp4"
    nested_log = tmp_path / "c" / "d" / "run_log.jsonl"
    assert not nested_video.parent.exists()

    result = process_video(
        clip,
        consent=True,
        output_video=nested_video,
        output_log=nested_log,
    )
    assert Path(result.video_path).is_file()
    assert Path(result.run_log_path).is_file()


# ---------------------------------------------------------------------------
# Default-path collision handling
# ---------------------------------------------------------------------------


def _sha256_bytes(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _redirect_default_outputs(monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    """Point the delivered default output paths into a temporary directory."""
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    out_video = out_dir / "output.mp4"
    out_log = out_dir / "run_log.jsonl"

    monkeypatch.setattr(demo, "DEFAULT_OUTPUT_VIDEO", out_video)
    monkeypatch.setattr(demo, "DEFAULT_OUTPUT_LOG", out_log)
    # The default values of ``process_video`` are bound at import time, so
    # patch them to match for tests that omit the output arguments.
    monkeypatch.setitem(
        demo.process_video.__kwdefaults__, "output_video", out_video
    )
    monkeypatch.setitem(
        demo.process_video.__kwdefaults__, "output_log", out_log
    )
    return out_video, out_log


def test_repeated_default_run_auto_suffixes_and_preserves_evidence(
    tmp_path: Path, monkeypatch
):
    """Two default runs must keep the first evidence intact and write a second pair."""
    clip = tmp_path / "clip.mp4"
    _clip_video(TEST_VIDEO, clip, n_frames=5)

    out_video, out_log = _redirect_default_outputs(monkeypatch, tmp_path)

    result1 = process_video(clip, consent=True)
    assert result1.video_path == str(out_video)
    assert result1.run_log_path == str(out_log)
    assert out_video.is_file()
    assert out_log.is_file()

    video_hash = _sha256_bytes(out_video)
    log_hash = _sha256_bytes(out_log)

    result2 = process_video(clip, consent=True)
    out_video2 = out_video.with_name("output-2.mp4")
    out_log2 = out_log.with_name("run_log-2.jsonl")
    assert result2.video_path == str(out_video2)
    assert result2.run_log_path == str(out_log2)
    assert out_video2.is_file()
    assert out_log2.is_file()

    # Original evidence must be byte-for-byte unchanged.
    assert _sha256_bytes(out_video) == video_hash
    assert _sha256_bytes(out_log) == log_hash


def test_default_run_skips_existing_suffixed_pair(
    tmp_path: Path, monkeypatch
):
    """If ``output-2``/``run_log-2`` already exist, the next run uses ``-3``."""
    clip = tmp_path / "clip.mp4"
    _clip_video(TEST_VIDEO, clip, n_frames=5)

    out_video, out_log = _redirect_default_outputs(monkeypatch, tmp_path)

    # Pre-create a run so the suffix-2 pair exists.
    process_video(clip, consent=True)
    process_video(clip, consent=True)
    assert out_video.with_name("output-2.mp4").is_file()
    assert out_log.with_name("run_log-2.jsonl").is_file()

    result3 = process_video(clip, consent=True)
    out_video3 = out_video.with_name("output-3.mp4")
    out_log3 = out_log.with_name("run_log-3.jsonl")
    assert result3.video_path == str(out_video3)
    assert result3.run_log_path == str(out_log3)


def test_explicit_output_video_collision_fails_closed(tmp_path: Path):
    """An explicit --output-video path must refuse to overwrite an existing file."""
    clip = tmp_path / "clip.mp4"
    _clip_video(TEST_VIDEO, clip, n_frames=5)

    existing_video = tmp_path / "existing.mp4"
    existing_video.write_bytes(b"not a video, but it exists")
    out_log = tmp_path / "run_log.jsonl"

    with pytest.raises(demo.DemoError, match="Output video already exists"):
        process_video(
            clip,
            consent=True,
            output_video=existing_video,
            output_log=out_log,
        )


def test_explicit_output_log_collision_fails_closed(tmp_path: Path):
    """An explicit --output-log path must refuse to overwrite an existing file."""
    clip = tmp_path / "clip.mp4"
    _clip_video(TEST_VIDEO, clip, n_frames=5)

    out_video = tmp_path / "out.mp4"
    existing_log = tmp_path / "run_log.jsonl"
    existing_log.write_text('{"event": "startup"}\n')

    with pytest.raises(demo.DemoError, match="Audit log already exists"):
        process_video(
            clip,
            consent=True,
            output_video=out_video,
            output_log=existing_log,
        )


def test_cli_default_run_with_existing_outputs_selects_suffix_and_prints_paths(
    tmp_path: Path, monkeypatch, capsys
):
    """The bare CLI command auto-suffixes when delivered default outputs exist."""
    clip = tmp_path / "clip.mp4"
    _clip_video(TEST_VIDEO, clip, n_frames=5)

    out_video, out_log = _redirect_default_outputs(monkeypatch, tmp_path)

    # First run writes the default pair.
    code = demo.main([
        "--video", str(clip),
        "--no-browser",
    ])
    assert code == 0
    assert out_video.is_file()
    assert out_log.is_file()

    video_hash = _sha256_bytes(out_video)
    log_hash = _sha256_bytes(out_log)

    captured1 = capsys.readouterr()
    assert str(out_video) in captured1.out
    assert str(out_log) in captured1.out

    # Second bare run must pick the -2 pair and leave the originals intact.
    code = demo.main([
        "--video", str(clip),
        "--no-browser",
    ])
    assert code == 0
    out_video2 = out_video.with_name("output-2.mp4")
    out_log2 = out_log.with_name("run_log-2.jsonl")
    assert out_video2.is_file()
    assert out_log2.is_file()
    assert _sha256_bytes(out_video) == video_hash
    assert _sha256_bytes(out_log) == log_hash

    captured2 = capsys.readouterr()
    assert str(out_video2) in captured2.out
    assert str(out_log2) in captured2.out


def test_cli_explicit_output_collision_fails_closed(
    tmp_path: Path, monkeypatch, capsys
):
    """Passing --output-video/--output-log keeps the fail-closed contract."""
    clip = tmp_path / "clip.mp4"
    _clip_video(TEST_VIDEO, clip, n_frames=5)

    existing_video = tmp_path / "out.mp4"
    existing_video.write_bytes(b"collision")
    out_log = tmp_path / "run_log.jsonl"

    monkeypatch.setattr(demo, "build_app", lambda **k: (_ for _ in ()).throw(AssertionError("browser launch attempted")))

    code = demo.main([
        "--video", str(clip),
        "--no-browser",
        "--output-video", str(existing_video),
        "--output-log", str(out_log),
    ])
    captured = capsys.readouterr()
    assert code == 1
    assert "Output video already exists" in captured.err
    assert "Traceback" not in captured.out + captured.err
