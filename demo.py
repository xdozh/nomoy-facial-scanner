"""End-to-end integration entry point for the "It Scans & Identifies" demo.

This module belongs to the INTEGRATION AGENT ("The Assembler"). It wires the
Gate-A/B/C-certified modules (detect, track, face, render, app, audit) into a
single offline, consent-gated, audit-traceable pipeline.

Public API
----------
``process_video(video_path, progress_cb=None, consent=True, ...)`` runs the
full decode → detect → track → recognize → audit → annotate pipeline and
returns an ``app.PipelineResult``-compatible record.

CLI
---
``python demo.py --video samples/test.mp4`` processes the video and launches
the local Gradio panel. Use ``--no-browser`` for headless/QA runs.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import cv2

# Local module boundaries (certified by upstream gates; do not edit internals).
import app
import audit
from app import PipelineResult, build_app
from detect import PersonDetector
from face import FaceRecognizer
from render import VideoWriter, annotate_frame
from track import PersonTracker


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_VIDEO = ROOT / "outputs" / "output.mp4"
DEFAULT_OUTPUT_LOG = ROOT / "outputs" / "run_log.jsonl"
DEFAULT_CONFIG = ROOT / "config.yaml"
DEFAULT_GALLERY = ROOT / "gallery"
DEFAULT_MODELS = ROOT / "models"
DEFAULT_MODEL_LOCK = ROOT / "models.lock"

# Video extension whitelist shared with app.py / audit.py
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


class DemoError(Exception):
    """User-facing error raised by the integration layer."""


def _resolve_default_outputs(
    output_video: Path, output_log: Path
) -> tuple[Path, Path]:
    """Pick a collision-free output pair under ``outputs/`` for default runs.

    When the caller did not explicitly override the default paths, the demo
    must keep delivering immutable prior evidence intact.  This helper chooses
    the lowest numeric suffix such that *both* the annotated video and the
    audit log are absent, guaranteeing the two artifacts of a single run are
    never split across different suffixes.

    ``output.mp4`` / ``run_log.jsonl`` are used on the first run; repeated runs
    produce ``output-2.mp4`` / ``run_log-2.jsonl``, ``output-3.mp4`` /
    ``run_log-3.jsonl``, etc.
    """
    if output_video != DEFAULT_OUTPUT_VIDEO or output_log != DEFAULT_OUTPUT_LOG:
        return output_video, output_log

    # Use the bare defaults only when the pair is completely absent.
    if not DEFAULT_OUTPUT_VIDEO.exists() and not DEFAULT_OUTPUT_LOG.exists():
        return DEFAULT_OUTPUT_VIDEO, DEFAULT_OUTPUT_LOG

    suffix = 2
    while True:
        candidate_video = DEFAULT_OUTPUT_VIDEO.with_name(f"output-{suffix}.mp4")
        candidate_log = DEFAULT_OUTPUT_LOG.with_name(f"run_log-{suffix}.jsonl")
        if not candidate_video.exists() and not candidate_log.exists():
            return candidate_video, candidate_log
        suffix += 1


def _model_versions() -> dict[str, Any]:
    """Return the verified model lock as the model-version map for audit."""
    try:
        verified = audit.verify_models(DEFAULT_MODEL_LOCK, root=ROOT)
    except audit.ModelVerificationError as exc:
        raise DemoError(f"Model verification failed: {exc}") from exc
    return verified


def _summarize_identities(
    log_path: Path,
    frame_duration: float,
    display_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Derive identity cards from the append-only audit log.

    The audit log only records identity at ``match``/``reidentify`` events and
    track lifecycle at ``track_birth``/``track_death``. Identity presence between
    those events is inferred by extending a matched track's lifespan, which
    gives accurate first/last seen and total screen time without weakening the
    append-only contract.
    """
    records = audit.verify_log(log_path)
    gallery_cfg = (display_config or {}).get("gallery", {}) or {}

    last_frame = max(
        (r["frame"] for r in records if r.get("event") == "frame_complete" and r.get("frame") is not None),
        default=-1,
    )
    frame_ts: dict[int, float] = {
        r["frame"]: float(r["ts"])
        for r in records
        if r.get("event") == "frame_complete" and r.get("frame") is not None
    }

    # Track lifespans from birth/death events.
    births: dict[int, int] = {}
    deaths: dict[int, int] = {}
    for r in records:
        if r.get("event") == "track_birth":
            births[int(r["track_id"])] = int(r["frame"])
        elif r.get("event") == "track_death":
            deaths[int(r["track_id"])] = int(r["frame"])

    # Identity assignment changes per track.
    assignments: dict[int, list[tuple[int, str]]] = {}
    for r in records:
        if r.get("event") not in {"match", "reidentify"}:
            continue
        track_id = int(r["track_id"])
        identity = str(r["identity"])
        frame = int(r["frame"])
        assignments.setdefault(track_id, []).append((frame, identity))

    # For each track, build [start_frame, end_frame] identity intervals.
    identity_frames: dict[str, set[int]] = {}
    for track_id, history in assignments.items():
        # Sort by assignment frame; use the latest identity for overlapping parts.
        history.sort(key=lambda x: x[0])
        birth = births.get(track_id, 0)
        death = deaths.get(track_id, last_frame)
        for idx, (start_frame, identity) in enumerate(history):
            end_frame = history[idx + 1][0] - 1 if idx + 1 < len(history) else death
            effective_start = max(start_frame, birth)
            effective_end = max(effective_start, min(end_frame, last_frame))
            identity_frames.setdefault(identity, set()).update(
                range(effective_start, effective_end + 1)
            )

    # Best confidence per identity from match/reidentify events.
    confidence: dict[str, float] = {}
    for r in records:
        if r.get("event") not in {"match", "reidentify"}:
            continue
        identity = str(r["identity"])
        conf = float(r["confidence"]) if r.get("confidence") is not None else 0.0
        confidence[identity] = max(confidence.get(identity, 0.0), conf)

    cards: list[dict[str, Any]] = []
    for identity in sorted(identity_frames):
        frames = identity_frames[identity]
        if not frames:
            continue
        timestamps = sorted({frame_ts[f] for f in frames if f in frame_ts})
        meta = gallery_cfg.get(identity, {}) if isinstance(gallery_cfg, dict) else {}
        display_name = meta.get("display_name") if isinstance(meta, dict) else None
        display_name = display_name or identity
        photo = DEFAULT_GALLERY / f"{identity}.jpg"
        if not photo.is_file():
            for suffix in (".jpeg", ".png", ".webp", ".bmp"):
                alt = DEFAULT_GALLERY / f"{identity}{suffix}"
                if alt.is_file():
                    photo = alt
                    break

        # Keep the list readable in the UI while remaining audit-derived.
        timestamps_seen = timestamps
        if len(timestamps_seen) > 12:
            step = max(1, len(timestamps_seen) // 10)
            sampled = [timestamps_seen[0]] + timestamps_seen[1:-1:step] + [timestamps_seen[-1]]
            timestamps_seen = sorted(set(round(t, 3) for t in sampled))

        cards.append({
            "identity": identity,
            "display_name": display_name,
            "gallery_photo": str(photo) if photo.is_file() else None,
            "confidence": confidence.get(identity, 0.0),
            "first_seen": timestamps[0] if timestamps else 0.0,
            "last_seen": timestamps[-1] if timestamps else 0.0,
            "screen_time_seconds": len(frames) * frame_duration,
            "timestamps_seen": timestamps_seen,
        })
    return cards


def process_video(
    video_path: str | os.PathLike[str],
    progress_cb: Optional[Callable[[float, str], None]] = None,
    *,
    consent: bool = True,
    config_path: str | os.PathLike[str] = DEFAULT_CONFIG,
    gallery_dir: str | os.PathLike[str] = DEFAULT_GALLERY,
    output_video: str | os.PathLike[str] = DEFAULT_OUTPUT_VIDEO,
    output_log: str | os.PathLike[str] = DEFAULT_OUTPUT_LOG,
    weights_dir: str | os.PathLike[str] = DEFAULT_MODELS,
    resolve_default_collisions: bool = True,
) -> PipelineResult:
    """Run the full offline pipeline on ``video_path``.

    Parameters
    ----------
    video_path
        Local video file to process.
    progress_cb
        Optional ``callback(0.0..1.0, description)`` for progress reporting.
    consent
        Whether explicit consent has been obtained. Required for identity
        recognition; when ``False`` the function raises before inference.
    config_path, gallery_dir
        Paths to ``config.yaml`` and the gallery folder. Swapping these
        changes who is recognized without any code edits.
    output_video, output_log
        Deterministic output paths. Parent directories are created.
    weights_dir
        Directory containing YOLO ``.pt`` files and the extracted ``buffalo_l``
        InsightFace bundle.
    resolve_default_collisions
        When ``True`` and the requested outputs are the delivered defaults,
        automatically pick a numeric suffix pair instead of overwriting prior
        evidence.  Explicitly overridden paths are never silently redirected.
    The append-only audit log is always durably retained because it is a
    required pipeline output and UI download.

    Returns
    -------
        ``app.PipelineResult`` with annotated video path, log path, and
        audit-derived identity summaries.
    """
    video_path = Path(video_path)
    output_video = Path(output_video)
    output_log = Path(output_log)
    config_path = Path(config_path)

    if resolve_default_collisions:
        output_video, output_log = _resolve_default_outputs(output_video, output_log)

    # Fail closed for any explicitly requested collision.  The log guard in
    # ``audit.AuditLogger`` also protects the JSONL, but we refuse the video
    # up front so both artifacts behave consistently.
    if output_video.exists():
        raise DemoError(f"Output video already exists: {output_video}")
    if output_log.exists():
        raise DemoError(f"Audit log already exists and cannot be overwritten: {output_log}")

    # ---- Consent gate ------------------------------------------------------
    try:
        audit.require_consent(consent)
    except audit.ConsentRequiredError as exc:
        raise DemoError(str(exc)) from exc

    # ---- Input validation --------------------------------------------------
    try:
        audit.validate_video(video_path, check_decode=True)
    except audit.InputValidationError as exc:
        raise DemoError(str(exc)) from exc
    if video_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise DemoError(f"Unsupported extension: {video_path.suffix}")

    # ---- Load config early for display names -------------------------------
    import yaml
    try:
        with open(config_path) as handle:
            display_config = yaml.safe_load(handle) or {}
        if not isinstance(display_config, dict):
            raise ValueError("config.yaml must be a mapping")
    except Exception as exc:
        raise DemoError(f"Cannot read config.yaml: {exc}") from exc

    def _progress(frac: float, desc: str = "") -> None:
        if progress_cb is not None:
            try:
                progress_cb(frac, desc)
            except Exception:  # noqa: BLE001
                pass

    # ---- Decode source metadata -------------------------------------------
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise DemoError("Video could not be opened for decoding")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise DemoError("Video has invalid frame dimensions")
    frame_duration = 1.0 / fps if fps > 0 else 0.0
    cap.release()

    # ---- Offline enforcement + model verification + run -------------------
    model_versions = _model_versions()

    with audit.enforce_offline():
        detector = PersonDetector(weights_dir=str(weights_dir))
        tracker = PersonTracker(max_occlusion=30, frame_rate=int(round(fps)))
        recognizer = FaceRecognizer(
            config_path,
            gallery_dir=gallery_dir,
            model_dir=Path(weights_dir) / "buffalo_l",
        )

        output_video.parent.mkdir(parents=True, exist_ok=True)
        output_log.parent.mkdir(parents=True, exist_ok=True)

        with audit.AuditLogger(
            output_log,
            model_versions=model_versions,
            consent=consent,
            save=True,
        ) as logger:
            with VideoWriter(
                output_video,
                fps=fps,
                frame_size=(width, height),
            ) as writer:
                cap = cv2.VideoCapture(str(video_path))
                frame_index = 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    ts = frame_index * frame_duration

                    detections = detector.detect(frame)
                    tracks, track_events = tracker.update(
                        detections, frame_index=frame_index, frame=frame
                    )
                    recognitions, recognition_events = recognizer.update(
                        frame, tracks, frame_index=frame_index
                    )

                    events = list(track_events) + list(recognition_events)
                    # Events already carry their own frame from upstream modules;
                    # the logger records the completion marker separately.
                    logger.record_frame(frame_index, ts, events)

                    annotated = annotate_frame(frame, tracks, recognitions, display_config)
                    writer.write_frame(annotated)

                    frame_index += 1
                    if frame_index % 10 == 0 or frame_index == frame_count:
                        _progress(
                            min(1.0, frame_index / max(1, frame_count)),
                            f"frame {frame_index}/{max(1, frame_count)}",
                        )

                cap.release()

    # ---- Derive identity summaries from the immutable audit log -------------
    identities = _summarize_identities(output_log, frame_duration, display_config)

    _progress(1.0, "done")
    return PipelineResult(
        video_path=str(output_video),
        run_log_path=str(output_log),
        identities=identities,
    )


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="It Scans & Identifies — offline face-recognition demo."
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the local video file to process (mp4/mov/avi/mkv/webm/m4v).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-open the Gradio panel after processing (headless/QA).",
    )
    parser.add_argument(
        "--output-video",
        default=None,
        help="Destination path for the annotated video (default: outputs/output.mp4).",
    )
    parser.add_argument(
        "--output-log",
        default=None,
        help="Destination path for the append-only run_log.jsonl (default: outputs/run_log.jsonl).",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to config.yaml.",
    )
    parser.add_argument(
        "--gallery",
        default=str(DEFAULT_GALLERY),
        help="Path to the gallery directory.",
    )
    parser.add_argument(
        "--models",
        default=str(DEFAULT_MODELS),
        help="Directory containing YOLO weights and the extracted buffalo_l bundle.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Gradio server port (default 7860).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    # Only when the CLI user did not override output paths do we auto-resolve
    # collisions under ``outputs/``.  Explicit overrides must fail closed.
    explicit_outputs = args.output_video is not None or args.output_log is not None
    output_video = Path(args.output_video) if args.output_video is not None else DEFAULT_OUTPUT_VIDEO
    output_log = Path(args.output_log) if args.output_log is not None else DEFAULT_OUTPUT_LOG

    try:
        print("Verifying models...")
        _model_versions()  # fail early with a clean message

        print(f"Processing {args.video}...")
        result = process_video(
            args.video,
            consent=True,
            config_path=args.config,
            gallery_dir=args.gallery,
            output_video=output_video,
            output_log=output_log,
            weights_dir=args.models,
            resolve_default_collisions=not explicit_outputs,
            progress_cb=lambda f, d: print(f"  [{f*100:5.1f}%] {d}"),
        )
        if not explicit_outputs and (
            Path(result.video_path) != DEFAULT_OUTPUT_VIDEO
            or Path(result.run_log_path) != DEFAULT_OUTPUT_LOG
        ):
            print("Prior default outputs already exist; wrote new run to:")
        print(f"Annotated video: {result.video_path}")
        print(f"Audit log:       {result.run_log_path}")
        print(f"Identities:      {len(result.identities)}")
        for person in result.identities:
            print(f"  - {person['display_name']} (conf={person['confidence']:.2f})")

        if args.no_browser:
            print("--no-browser: skipping Gradio launch.")
            return 0

        print(f"Launching local panel at http://127.0.0.1:{args.port} ...")
        demo = build_app(process_fn=process_video)
        demo.launch(
            server_name="127.0.0.1",
            server_port=args.port,
            share=False,
            inbrowser=True,
            max_file_size=app.MAX_FILE_SIZE_BYTES,
        )
        return 0
    except (DemoError, audit.AuditError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
