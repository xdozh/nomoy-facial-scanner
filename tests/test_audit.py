"""Security and completeness tests for the append-only audit boundary."""

from __future__ import annotations

import hashlib
import json
import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audit import (
    AuditLogError,
    AuditLogger,
    ConsentRequiredError,
    InputValidationError,
    ModelVerificationError,
    OfflineViolationError,
    derive_identity_summary,
    enforce_offline,
    offline_smoke_test,
    validate_video,
    verify_log,
    verify_models,
)


def _model_fixture(tmp_path: Path) -> tuple[Path, Path]:
    artifact = tmp_path / "models" / "fixture.bin"
    artifact.parent.mkdir()
    artifact.write_bytes(b"local model")
    lock = tmp_path / "models.lock"
    lock.write_text(json.dumps({
        "schema": "models.lock/v1",
        "models": [{
            "path": "models/fixture.bin",
            "model_version": "fixture-v1",
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        }],
    }), encoding="utf-8")
    return lock, artifact


def test_models_verified_and_tamper_refused_using_copy(tmp_path: Path) -> None:
    lock, artifact = _model_fixture(tmp_path)
    assert verify_models(lock) == {
        "models/fixture.bin": {
            "version": "fixture-v1",
            "sha256": hashlib.sha256(b"local model").hexdigest(),
        }
    }
    artifact.write_bytes(b"tampered copy")
    with pytest.raises(ModelVerificationError, match="checksum mismatch") as caught:
        verify_models(lock)
    assert caught.value.__cause__ is None


def test_complete_append_only_log_and_hash_chain(tmp_path: Path) -> None:
    path = tmp_path / "run_log.jsonl"
    models = {"fixture": {"version": "v1", "sha256": "a" * 64}}
    logger = AuditLogger(path, models, consent=True, save=True)
    logger.record_frame(0, 0.0, [{"event": "track_birth", "track_id": 7}])
    logger.record_frame(1, 0.04, [{"event": "match", "track_id": 7, "identity": "Ada", "confidence": 0.875}])
    logger.record_frame(2, 0.08, [{"event": "track_death", "track_id": 7}])
    logger.close()

    records = verify_log(path)
    assert [item["frame"] for item in records if item["event"] == "frame_complete"] == [0, 1, 2]
    match = next(item for item in records if item["event"] == "match")
    assert match["identity"] == "Ada"
    assert match["confidence"] == 0.875
    assert match["model_versions"] == models
    assert all(len(item["record_hash"]) == 64 and len(item["prev_hash"]) == 64 for item in records)
    with pytest.raises(AuditLogError, match="already exists"):
        AuditLogger(path, models, consent=True)


def test_frame_gap_refused_before_write(tmp_path: Path) -> None:
    path = tmp_path / "run_log.jsonl"
    logger = AuditLogger(path, {}, consent=True)
    before = path.read_bytes()
    with pytest.raises(AuditLogError, match="expected 0"):
        logger.record_frame(1, 0.1)
    assert path.read_bytes() == before


def test_ephemeral_log_removes_identity_data_on_close(tmp_path: Path) -> None:
    path = tmp_path / "ephemeral.jsonl"
    logger = AuditLogger(path, {}, consent=True)
    logger.record_frame(0, 0.0, [{"event": "match", "track_id": 1, "identity": "Ada", "confidence": 0.9}])
    assert b'"identity":"Ada"' in path.read_bytes()
    logger.close()
    logger.close()
    assert not path.exists()


def test_ephemeral_context_manager_removes_identity_data(tmp_path: Path) -> None:
    path = tmp_path / "ephemeral.jsonl"
    with AuditLogger(path, {}, consent=True) as logger:
        logger.record_frame(0, 0.0, [{"event": "match", "track_id": 1, "identity": "Ada", "confidence": 0.9}])
    assert not path.exists()


def test_log_tampering_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "run_log.jsonl"
    logger = AuditLogger(path, {}, consent=True, save=True)
    logger.record_frame(0, 0.0)
    logger.close()
    path.write_text(path.read_text(encoding="utf-8").replace('"frame":0', '"frame":9'), encoding="utf-8")
    with pytest.raises(AuditLogError, match="integrity"):
        verify_log(path)


def test_recognition_requires_explicit_consent_without_partial_frame(tmp_path: Path) -> None:
    path = tmp_path / "run_log.jsonl"
    logger = AuditLogger(path, {}, consent=False)
    before = path.read_bytes()
    with pytest.raises(ConsentRequiredError, match="Explicit consent"):
        logger.record_frame(0, 0.0, [{"event": "match", "track_id": 1, "identity": "Ada", "confidence": 0.9}])
    assert path.read_bytes() == before


def test_video_extension_oversize_and_decode_errors_are_clean(tmp_path: Path) -> None:
    bad_extension = tmp_path / "clip.txt"
    bad_extension.write_bytes(b"data")
    with pytest.raises(InputValidationError, match="Unsupported"):
        validate_video(bad_extension)

    oversized = tmp_path / "large.mp4"
    oversized.write_bytes(b"12345")
    with pytest.raises(InputValidationError, match="500MB") as caught:
        validate_video(oversized, max_bytes=4)
    assert caught.value.__cause__ is None

    invalid = tmp_path / "invalid.mp4"
    invalid.write_bytes(b"not a video")
    with pytest.raises(InputValidationError, match="decoded") as caught:
        validate_video(invalid, check_decode=True)
    assert caught.value.__cause__ is None


def test_offline_enforcement_blocks_and_restores_socket() -> None:
    original = socket.create_connection
    with enforce_offline():
        with pytest.raises(OfflineViolationError, match="disabled"):
            socket.create_connection(("127.0.0.1", 9))
    assert socket.create_connection is original
    assert offline_smoke_test() is True


def test_identity_summary_preserves_exact_values() -> None:
    records = [
        {"event": "match", "frame": 2, "ts": 0.08, "identity": "Ada", "confidence": 0.8},
        {"event": "reidentify", "frame": 4, "ts": 0.16, "identity": "Ada", "confidence": 0.95},
        {"event": "match", "frame": 4, "ts": 0.16, "identity": "Ada", "confidence": 0.9},
    ]
    assert derive_identity_summary(records, frame_duration=0.04) == [{
        "identity": "Ada",
        "confidence": 0.95,
        "first_ts": 0.08,
        "last_ts": 0.16,
        "screen_time": 0.08,
    }]
