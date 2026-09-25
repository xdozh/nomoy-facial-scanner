"""Append-only audit and runtime security boundaries for local video runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import socket
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

ALLOWED_EVENTS = frozenset({"track_birth", "track_death", "match", "reidentify"})
ALLOWED_VIDEO_EXTENSIONS = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"})
MAX_VIDEO_BYTES = 500 * 1024 * 1024
GENESIS_HASH = "0" * 64


class AuditError(Exception):
    """Base class for errors safe to display directly to a user."""


class ModelVerificationError(AuditError):
    pass


class InputValidationError(AuditError):
    pass


class ConsentRequiredError(AuditError):
    pass


class OfflineViolationError(AuditError):
    pass


class AuditLogError(AuditError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ModelVerificationError(f"Cannot read model artifact: {path.name}") from None
    return digest.hexdigest()


def verify_models(lock_path: str | os.PathLike[str], root: str | os.PathLike[str] | None = None) -> dict[str, dict[str, str]]:
    """Verify every locked artifact and return versions and actual hashes."""
    lock = Path(lock_path)
    try:
        document = json.loads(lock.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ModelVerificationError("Model lock file is missing or invalid") from None
    if document.get("schema") != "models.lock/v1" or not isinstance(document.get("models"), list):
        raise ModelVerificationError("Unsupported model lock format")
    base = Path(root) if root is not None else lock.parent
    verified: dict[str, dict[str, str]] = {}
    for item in document["models"]:
        if not isinstance(item, dict) or not all(isinstance(item.get(key), str) for key in ("path", "model_version", "sha256")):
            raise ModelVerificationError("Model lock contains an invalid entry")
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ModelVerificationError("Model lock contains an unsafe artifact path")
        artifact = base / relative
        if not artifact.is_file():
            raise ModelVerificationError(f"Model artifact is missing: {relative.as_posix()}")
        actual = _sha256(artifact)
        expected = item["sha256"].lower()
        if len(expected) != 64 or actual != expected:
            raise ModelVerificationError(f"Model checksum mismatch: {relative.as_posix()}")
        verified[relative.as_posix()] = {"version": item["model_version"], "sha256": actual}
    return verified


def validate_video(path: str | os.PathLike[str], *, check_decode: bool = False, max_bytes: int = MAX_VIDEO_BYTES) -> Path:
    """Validate a local candidate without leaking implementation exceptions."""
    candidate = Path(path)
    if candidate.suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
        raise InputValidationError("Unsupported video format")
    try:
        if not candidate.is_file():
            raise InputValidationError("Video file does not exist")
        if candidate.stat().st_size > max_bytes:
            raise InputValidationError("Video exceeds the 500MB limit")
    except InputValidationError:
        raise
    except OSError:
        raise InputValidationError("Video file cannot be read") from None
    if check_decode:
        try:
            import cv2

            capture = cv2.VideoCapture(str(candidate))
            opened = capture.isOpened()
            decoded, _ = capture.read() if opened else (False, None)
            capture.release()
        except Exception:
            raise InputValidationError("Video could not be decoded") from None
        if not opened or not decoded:
            raise InputValidationError("Video could not be decoded")
    return candidate


def require_consent(consent: bool) -> None:
    if consent is not True:
        raise ConsentRequiredError("Explicit consent is required for identity recognition")


def _blocked(*_args: Any, **_kwargs: Any) -> Any:
    raise OfflineViolationError("Outbound network access is disabled")


@contextmanager
def enforce_offline() -> Iterator[None]:
    """Fail closed for Python socket connection and DNS entry points."""
    originals = (socket.socket.connect, socket.socket.connect_ex, socket.create_connection, socket.getaddrinfo)
    socket.socket.connect = _blocked  # type: ignore[method-assign]
    socket.socket.connect_ex = _blocked  # type: ignore[method-assign]
    socket.create_connection = _blocked
    socket.getaddrinfo = _blocked
    try:
        yield
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.create_connection, socket.getaddrinfo = originals  # type: ignore[method-assign]


def offline_smoke_test() -> bool:
    with enforce_offline():
        try:
            socket.create_connection(("127.0.0.1", 9))
        except OfflineViolationError:
            return True
    raise OfflineViolationError("Offline enforcement smoke test failed")


def _timestamp(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
        raise AuditLogError("Event timestamp must be a finite non-negative number")
    return float(value)


def _confidence(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0 <= value <= 1:
        raise AuditLogError("Event confidence must be between 0 and 1")
    return float(value)


def normalize_event(event: Mapping[str, Any], frame: int, ts: float, model_versions: Mapping[str, Any], consent: bool) -> dict[str, Any]:
    kind = event.get("event")
    if kind not in ALLOWED_EVENTS:
        raise AuditLogError("Unsupported audit event")
    if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
        raise AuditLogError("Frame index must be a non-negative integer")
    track_id = event.get("track_id")
    if isinstance(track_id, bool) or not isinstance(track_id, int) or track_id < 0:
        raise AuditLogError("Track ID must be a non-negative integer")
    identity = event.get("identity")
    confidence = _confidence(event.get("confidence"))
    if kind in {"match", "reidentify"}:
        require_consent(consent)
        if not isinstance(identity, str) or not identity.strip():
            raise AuditLogError("Recognition events require an identity")
        if confidence is None:
            raise AuditLogError("Recognition events require confidence")
    elif identity is not None:
        require_consent(consent)
    return {
        "frame": frame,
        "ts": _timestamp(ts),
        "track_id": track_id,
        "event": kind,
        "identity": identity,
        "confidence": confidence,
        "model_versions": dict(model_versions),
    }


class AuditLogger:
    """Durable JSONL writer using one append syscall per hash-linked record."""

    def __init__(self, path: str | os.PathLike[str], model_versions: Mapping[str, Any], *, consent: bool, save: bool = False) -> None:
        self.path = Path(path)
        self.model_versions = dict(model_versions)
        self.consent = consent
        self.save = save
        self._last_frame = -1
        self._previous_hash = GENESIS_HASH
        self._closed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.path.stat().st_size:
            raise AuditLogError("Audit log already exists and cannot be overwritten")
        self._append({
            "frame": None,
            "ts": datetime.now(timezone.utc).isoformat(),
            "track_id": None,
            "event": "startup",
            "identity": None,
            "confidence": None,
            "model_versions": self.model_versions,
        })

    def _append(self, record: dict[str, Any]) -> None:
        if self._closed:
            raise AuditLogError("Audit log is closed")
        chained = {**record, "prev_hash": self._previous_hash}
        canonical = json.dumps(chained, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        chained["record_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        payload = (json.dumps(chained, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        descriptor = os.open(self.path, flags, 0o600)
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._previous_hash = chained["record_hash"]

    def record_frame(self, frame: int, ts: float, events: Iterable[Mapping[str, Any]] = ()) -> None:
        if frame != self._last_frame + 1:
            raise AuditLogError(f"Frame sequence gap: expected {self._last_frame + 1}, received {frame}")
        timestamp = _timestamp(ts)
        normalized = [normalize_event(item, frame, timestamp, self.model_versions, self.consent) for item in events]
        for item in normalized:
            self._append(item)
        self._append({
            "frame": frame,
            "ts": timestamp,
            "track_id": None,
            "event": "frame_complete",
            "identity": None,
            "confidence": None,
            "model_versions": self.model_versions,
        })
        self._last_frame = frame

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self.save:
            try:
                self.path.unlink(missing_ok=True)
            except OSError:
                raise AuditLogError("Ephemeral audit log could not be removed") from None

    def __enter__(self) -> "AuditLogger":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def verify_log(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Verify hashes and contiguous frame completion markers."""
    records: list[dict[str, Any]] = []
    previous = GENESIS_HASH
    expected_frame = 0
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        for line in lines:
            record = json.loads(line)
            claimed = record.pop("record_hash")
            canonical = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            if record.get("prev_hash") != previous or hashlib.sha256(canonical.encode("utf-8")).hexdigest() != claimed:
                raise AuditLogError("Audit log integrity check failed")
            record["record_hash"] = claimed
            previous = claimed
            records.append(record)
            if record.get("event") == "frame_complete":
                if record.get("frame") != expected_frame:
                    raise AuditLogError("Audit log contains a frame sequence gap")
                expected_frame += 1
    except AuditLogError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError):
        raise AuditLogError("Audit log is missing or invalid") from None
    if not records or records[0].get("event") != "startup":
        raise AuditLogError("Audit log is missing startup metadata")
    return records


def derive_identity_summary(records: Sequence[Mapping[str, Any]], *, frame_duration: float | None = None) -> list[dict[str, Any]]:
    """Derive exact first/last timestamps, best confidence, and screen time."""
    grouped: dict[str, dict[str, Any]] = {}
    seen_frames: dict[str, set[int]] = {}
    for record in records:
        identity = record.get("identity")
        if record.get("event") not in {"match", "reidentify"} or not isinstance(identity, str):
            continue
        ts = _timestamp(record.get("ts"))
        confidence = _confidence(record.get("confidence"))
        frame = record.get("frame")
        item = grouped.setdefault(identity, {"identity": identity, "confidence": confidence, "first_ts": ts, "last_ts": ts, "screen_time": 0.0})
        item["confidence"] = max(item["confidence"], confidence)
        item["first_ts"] = min(item["first_ts"], ts)
        item["last_ts"] = max(item["last_ts"], ts)
        if isinstance(frame, int):
            seen_frames.setdefault(identity, set()).add(frame)
    for identity, item in grouped.items():
        item["screen_time"] = len(seen_frames.get(identity, set())) * frame_duration if frame_duration is not None else item["last_ts"] - item["first_ts"]
    return sorted(grouped.values(), key=lambda item: (item["first_ts"], item["identity"]))


validate_video_input = validate_video
verify_model_lock = verify_models
identity_summary = derive_identity_summary
