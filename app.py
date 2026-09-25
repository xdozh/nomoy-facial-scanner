"""Gradio front-end for the face-recognition demo (Gate D).

Local-only app: upload a video -> progress -> annotated playback +
identity cards + downloads. The processing pipeline is injected by the
integrator (``demo.py`` owns it); ``build_app(process_fn=...)`` takes any
callable with signature ``process_fn(video_path, progress_callback)`` that
returns the annotated video path, the run-log path, and identity summaries
derived from audit events.
"""

from __future__ import annotations

import base64
import html
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import gradio as gr

# ---------------------------------------------------------------------------
# Constants / validation rules
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB

CONSENT_TEXT = "I confirm everyone in this video consented."

DISCLAIMER_TEXT = (
    "DEMO BUILD — enrolled volunteers only. Only upload footage of people who "
    "have explicitly consented to be enrolled and scanned. "
    "InsightFace pretrained models are licensed for research / non-commercial "
    "use only. Ultralytics YOLO is AGPL-3.0 — internal demo use is fine, but "
    "commercial use requires a separate Ultralytics license. "
    "No sightings database is persisted beyond this run."
)


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Normalized result returned by the injected processing callable."""

    video_path: str
    run_log_path: str
    identities: list[dict] = field(default_factory=list)


def _normalize_result(result: Any) -> PipelineResult:
    """Accept dict, tuple, or object results and normalize to PipelineResult."""
    if isinstance(result, PipelineResult):
        return result
    if isinstance(result, dict):
        return PipelineResult(
            video_path=result["video_path"],
            run_log_path=result["run_log_path"],
            identities=list(result.get("identities", [])),
        )
    if isinstance(result, (tuple, list)) and len(result) == 3:
        return PipelineResult(
            video_path=result[0], run_log_path=result[1], identities=list(result[2])
        )
    # object with attributes
    return PipelineResult(
        video_path=result.video_path,
        run_log_path=result.run_log_path,
        identities=list(getattr(result, "identities", [])),
    )


def _extract_path(video: Any) -> Optional[str]:
    """Gradio Video preprocess returns a str path; be liberal anyway."""
    if video is None:
        return None
    if isinstance(video, (str, os.PathLike)):
        return str(video)
    if isinstance(video, dict):
        for key in ("path", "video", "name"):
            if video.get(key):
                return str(video[key])
    name = getattr(video, "name", None)
    return str(name) if name else None


# ---------------------------------------------------------------------------
# Deterministic helpers (unit-testable, no server)
# ---------------------------------------------------------------------------


def validate_input(video_path: str) -> None:
    """Raise ValueError unless the file passes the input-safety rules."""
    if not video_path:
        raise ValueError("No video uploaded.")
    path = Path(video_path)
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{path.suffix}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        )
    if not path.is_file():
        raise ValueError("Uploaded file is not accessible.")
    if os.path.getsize(video_path) > MAX_FILE_SIZE_BYTES:
        raise ValueError("File exceeds the 500MB maximum.")
    try:  # decode check: clean failure, never a traceback at demo time
        import cv2

        cap = cv2.VideoCapture(video_path)
        ok = cap.isOpened()
        if ok:
            ok, _ = cap.read()
        cap.release()
        if not ok:
            raise ValueError("Video could not be decoded — file may be corrupt.")
    except ImportError:
        pass


def require_consent(consent_checked: Any) -> None:
    """Server-side consent gate. Calling the callback directly cannot bypass it."""
    if consent_checked is not True:
        raise gr.Error(
            "Consent required: you must confirm everyone in this video consented."
        )


def run_pipeline(
    video: Any,
    consent_checked: Any,
    process_fn: Callable[..., Any],
    progress: Optional[gr.Progress] = None,
) -> PipelineResult:
    """Consent gate -> validation -> injected processing callable.

    This is the single entry point wired to the Run button; consent and
    input validation are enforced here server-side.
    """
    require_consent(consent_checked)
    video_path = _extract_path(video)
    try:
        validate_input(video_path)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    if process_fn is None:
        raise gr.Error("Processing pipeline is not wired into this build.")

    def _cb(frac: float, desc: str = "") -> None:
        if progress is not None:
            progress(frac, desc=desc)

    return _normalize_result(process_fn(video_path, _cb))


def _photo_data_uri(photo_path: Any) -> Optional[str]:
    """Embed the gallery photo as a base64 data URI (offline-safe)."""
    if not photo_path:
        return None
    p = Path(str(photo_path))
    if not p.is_file():
        return None
    mime = {
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(p.suffix.lower(), "image/jpeg")
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def _fmt_ts(ts: Any) -> str:
    """Format a timestamp (seconds float or 'mm:ss' string) as mm:ss."""
    if isinstance(ts, (int, float)):
        m, s = divmod(float(ts), 60)
        return f"{int(m):02d}:{s:04.1f}"
    return str(ts)


def render_identity_cards(identities: Iterable[dict]) -> str:
    """Render the identity-card grid as HTML exactly from supplied summaries.

    Values come verbatim from the audit-derived identity records — the cards
    are only as correct as the supplied data (kill criterion: card/audit
    mismatch, so nothing is recomputed or inferred here).
    """
    records = list(identities or [])
    if not records:
        return '<div class="id-empty">No identities recognized in this run.</div>'

    cards = []
    for rec in records:
        name = html.escape(str(rec.get("display_name", "UNKNOWN")))
        conf = rec.get("confidence", 0.0)
        conf_txt = (
            f"{float(conf):.2f}"
            if isinstance(conf, (int, float))
            else html.escape(str(conf))
        )
        first_seen = html.escape(_fmt_ts(rec.get("first_seen", "")))
        last_seen = html.escape(_fmt_ts(rec.get("last_seen", "")))
        screen_time = rec.get("screen_time_seconds", 0.0)
        screen_txt = (
            f"{float(screen_time):.1f}s"
            if isinstance(screen_time, (int, float))
            else html.escape(str(screen_time))
        )
        ts_seen = rec.get("timestamps_seen") or []
        ts_txt = html.escape(
            ", ".join(_fmt_ts(t) for t in ts_seen)
            if isinstance(ts_seen, (list, tuple))
            else str(ts_seen)
        )
        uri = _photo_data_uri(rec.get("gallery_photo"))
        photo = (
            f'<img class="id-photo" src="{uri}" alt="{name}"/>'
            if uri
            else '<div class="id-photo id-photo-missing">?</div>'
        )
        cards.append(
            '<div class="id-card">'
            f"{photo}"
            f'<div class="id-name">{name}</div>'
            f'<div class="id-row"><span>confidence</span><b>{conf_txt}</b></div>'
            f'<div class="id-row"><span>first seen</span><b>{first_seen}</b></div>'
            f'<div class="id-row"><span>last seen</span><b>{last_seen}</b></div>'
            f'<div class="id-row"><span>screen time</span><b>{screen_txt}</b></div>'
            f'<div class="id-ts">seen at: {ts_txt}</div>'
            "</div>"
        )
    return '<div class="id-grid">' + "".join(cards) + "</div>"


CARD_CSS = """
.id-grid { display: flex; flex-wrap: wrap; gap: 12px; }
.id-card { border: 1px solid #444; border-radius: 10px; padding: 10px;
           width: 190px; background: #1b1b1f; color: #eee;
           font-family: system-ui, sans-serif; }
.id-photo { width: 100%; height: 140px; object-fit: cover; border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            background: #333; font-size: 48px; }
.id-name { font-weight: 700; font-size: 18px; margin: 8px 0 4px; }
.id-row { display: flex; justify-content: space-between; font-size: 13px; }
.id-ts { font-size: 11px; color: #aaa; margin-top: 6px;
         max-height: 48px; overflow-y: auto; }
.id-empty { color: #aaa; font-style: italic; }
.disclaimer { border: 1px solid #a80; background: #221a08; color: #fd8;
              padding: 10px 14px; border-radius: 8px; font-size: 13px; }
"""


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------


def _default_process_fn(video_path: str, progress_cb: Callable) -> PipelineResult:
    raise gr.Error("Processing pipeline not connected — run via demo.py integration.")


def build_app(process_fn: Optional[Callable[..., Any]] = None) -> gr.Blocks:
    """Build the Blocks app. ``process_fn`` is injected by the integrator."""
    fn = process_fn or _default_process_fn

    with gr.Blocks(title="Face Recognition Demo") as demo:
        gr.Markdown("# Face Recognition Demo")
        # CSS lives in this HTML block (Gradio 6 moved css= to launch()).
        gr.HTML(
            f"<style>{CARD_CSS}</style>"
            f'<div class="disclaimer">{html.escape(DISCLAIMER_TEXT)}</div>'
        )

        with gr.Row():
            with gr.Column(scale=1):
                video_in = gr.Video(
                    label="Upload video (max 500MB)",
                    sources=["upload"],
                )
                consent = gr.Checkbox(label=CONSENT_TEXT, value=False)
                run_btn = gr.Button("Run", variant="primary", interactive=False)
                consent.change(
                    lambda c: gr.update(interactive=bool(c)),
                    inputs=consent,
                    outputs=run_btn,
                )

            with gr.Column(scale=2, visible=False) as results_panel:
                video_out = gr.Video(label="Annotated output", autoplay=False)
                cards_out = gr.HTML(label="Identities")
                with gr.Row():
                    file_video = gr.File(label="output.mp4")
                    file_log = gr.File(label="run_log.jsonl")

        def _run(video, consent_checked, progress=gr.Progress()):
            result = run_pipeline(video, consent_checked, fn, progress)
            return (
                result.video_path,
                render_identity_cards(result.identities),
                result.video_path,
                result.run_log_path,
                gr.update(visible=True),  # auto-open results panel
            )

        run_btn.click(
            _run,
            inputs=[video_in, consent],
            outputs=[video_out, cards_out, file_video, file_log, results_panel],
            show_progress="full",
        )

        gr.Markdown(
            "_Demo build: enrolled volunteers only. InsightFace pretrained models: "
            "research / non-commercial license. Ultralytics YOLO: AGPL-3.0 — "
            "commercial use requires a license._"
        )

    return demo


def main() -> None:  # pragma: no cover - manual launch path
    demo = build_app()
    # Local-only: bound to loopback, no share tunnel; auto-open browser panel.
    demo.launch(
        server_name="127.0.0.1",
        share=False,
        inbrowser=True,
        max_file_size=MAX_FILE_SIZE_BYTES,
        css=CARD_CSS,
    )


if __name__ == "__main__":
    main()
