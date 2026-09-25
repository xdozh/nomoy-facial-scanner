#!/usr/bin/env python3
"""Reproducible Gate D / T5 evaluator for the complete local Gradio flow."""
from __future__ import annotations

import contextlib
import html
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

os.environ.update({
    "CUDA_VISIBLE_DEVICES": "",
    "HF_HUB_OFFLINE": "1",
    "ULTRALYTICS_OFFLINE": "true",
    "YOLO_OFFLINE": "true",
    "NO_PROXY": "*",
})

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import yaml

import app
import audit
import demo

EVIDENCE = ROOT / "outputs" / "gate_d"
VIDEO_OUT = EVIDENCE / "output.mp4"
LOG_OUT = EVIDENCE / "run_log.jsonl"
METRICS = EVIDENCE / "metrics.json"
PANEL = EVIDENCE / "panel_capture.html"
SAMPLE = ROOT / "samples" / "test.mp4"


def check(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def video_info(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    opened = cap.isOpened()
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC)) if opened else 0
    info = {
        "path": str(path.relative_to(ROOT)),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else 0,
        "opened": opened,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0,
        "fps": float(cap.get(cv2.CAP_PROP_FPS)) if opened else 0.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0,
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0,
        "fourcc": "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)),
    }
    decoded = 0
    while opened and cap.read()[0]:
        decoded += 1
    cap.release()
    info["decoded_frames"] = decoded
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=codec_name,codec_tag_string,nb_frames", "-of", "json", str(path)],
            capture_output=True, text=True, check=False,
        )
        info["ffprobe_returncode"] = proc.returncode
        info["ffprobe"] = json.loads(proc.stdout) if proc.returncode == 0 else proc.stderr
    return info


def independent_summary(log_path: Path, frame_duration: float, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    records = audit.verify_log(log_path)
    completions = {int(r["frame"]): float(r["ts"]) for r in records if r.get("event") == "frame_complete"}
    last_frame = max(completions, default=-1)
    births: dict[int, int] = {}
    deaths: dict[int, int] = {}
    histories: dict[int, list[tuple[int, str]]] = {}
    confidence: dict[str, float] = {}
    for record in records:
        event = record.get("event")
        track = record.get("track_id")
        if event == "track_birth":
            births[int(track)] = int(record["frame"])
        elif event == "track_death":
            deaths[int(track)] = int(record["frame"])
        elif event in {"match", "reidentify"}:
            identity = str(record["identity"])
            histories.setdefault(int(track), []).append((int(record["frame"]), identity))
            confidence[identity] = max(confidence.get(identity, 0.0), float(record["confidence"]))
    frames_by_identity: dict[str, set[int]] = {}
    for track, history in histories.items():
        history.sort()
        for i, (start, identity) in enumerate(history):
            end = history[i + 1][0] - 1 if i + 1 < len(history) else deaths.get(track, last_frame)
            start = max(start, births.get(track, 0))
            end = max(start, min(end, last_frame))
            frames_by_identity.setdefault(identity, set()).update(range(start, end + 1))
    cards = []
    gallery_cfg = cfg.get("gallery", {})
    for identity in sorted(frames_by_identity):
        frames = frames_by_identity[identity]
        timestamps = sorted({completions[f] for f in frames if f in completions})
        seen = timestamps
        if len(seen) > 12:
            step = max(1, len(seen) // 10)
            seen = sorted(set(round(t, 3) for t in [seen[0], *seen[1:-1:step], seen[-1]]))
        photo = next((ROOT / "gallery" / f"{identity}{suffix}" for suffix in (".jpg", ".jpeg", ".png", ".webp", ".bmp") if (ROOT / "gallery" / f"{identity}{suffix}").is_file()), None)
        cards.append({
            "identity": identity,
            "display_name": gallery_cfg.get(identity, {}).get("display_name") or identity,
            "gallery_photo": str(photo) if photo else None,
            "confidence": confidence.get(identity, 0.0),
            "first_seen": timestamps[0] if timestamps else 0.0,
            "last_seen": timestamps[-1] if timestamps else 0.0,
            "screen_time_seconds": len(frames) * frame_duration,
            "timestamps_seen": seen,
        })
    return cards


def component_and_dependency_checks(blocks: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = blocks.get_config_file()
    text = json.dumps(cfg)
    labels = [component.get("props", {}).get("label") for component in cfg.get("components", [])]
    deps = cfg.get("dependencies", [])
    run_deps = [d for d in deps if d.get("trigger_mode") or "click" in json.dumps(d).lower()]
    output_lengths = [len(d.get("outputs", [])) for d in deps]
    checks = [
        check("disclaimer content", all(x in text for x in ("enrolled volunteers", "non-commercial", "AGPL-3.0", "consent")), "mandatory warning terms present in built config"),
        check("required UI outputs", all(x in labels for x in ("Annotated output", "Identities", "output.mp4", "run_log.jsonl")), labels),
        check("run event returns complete panel", 5 in output_lengths, {"dependency_output_counts": output_lengths}),
        check("two-click interaction model", len(deps) == 2 and any(len(d.get("inputs", [])) == 1 for d in deps) and any(len(d.get("inputs", [])) == 2 for d in deps), {"dependency_count": len(deps), "dependencies": deps}),
    ]
    return checks, cfg


def launch_and_probe(blocks: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    server = None
    probes: dict[str, Any] = {}
    launched: Any = None
    reachable = downloads = False
    error: str | None = None
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    try:
        launched = blocks.launch(server_name="127.0.0.1", server_port=port, share=False, inbrowser=False, prevent_thread_lock=True, max_file_size=app.MAX_FILE_SIZE_BYTES, quiet=True)
        server = blocks
        base = f"http://127.0.0.1:{port}"
        time.sleep(1)
        for name, url in {"root": base + "/", "config": base + "/config"}.items():
            with urllib.request.urlopen(url, timeout=10) as response:
                body = response.read()
                probes[name] = {"status": response.status, "bytes": len(body), "content_type": response.headers.get("content-type")}
        reachable = all(probes[name]["status"] == 200 for name in ("root", "config"))

        # Invoke the actual built `_run` event over Gradio's API. This registers
        # returned File values and then downloads them through the same URLs a
        # browser uses, rather than bypassing Gradio's file-access allow-list.
        from gradio_client import Client, handle_file
        api_result = Client(base, verbose=False).predict(handle_file(str(SAMPLE)), True, api_name="/_run")
        probes["api_result"] = [str(item) for item in api_result]
        returned_video = Path(api_result[0]["video"] if isinstance(api_result[0], dict) else api_result[0])
        returned_video_file = Path(api_result[2])
        returned_log_file = Path(api_result[3])
        probes["downloads"] = {
            "playback_bytes": returned_video.stat().st_size,
            "video_file_bytes": returned_video_file.stat().st_size,
            "log_file_bytes": returned_log_file.stat().st_size,
        }
        downloads = returned_video.read_bytes() == VIDEO_OUT.read_bytes() and returned_video_file.read_bytes() == VIDEO_OUT.read_bytes() and returned_log_file.read_bytes() == LOG_OUT.read_bytes()
    except Exception as exc:
        error = repr(exc)
        probes["error"] = error
    finally:
        if server is not None:
            server.close()
    launch_detail = {"server_name": "127.0.0.1", "share": False, "launch_result": str(launched), "error": error}
    checks = [
        check("loopback server reachable", reachable, probes),
        check("download URLs resolve", downloads, probes),
        check("local-only launch", launched is not None and reachable, launch_detail),
    ]
    return checks, probes


def main() -> int:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    for path in (VIDEO_OUT, LOG_OUT, METRICS, PANEL):
        path.unlink(missing_ok=True)

    checks: list[dict[str, Any]] = []
    verified = audit.verify_models(ROOT / "models.lock", root=ROOT)
    checks.append(check("model checksums verified first", len(verified) == 3, verified))

    bypass_results = {}
    for value in (False, None, 0, "", "true", 1):
        called = []
        try:
            app.run_pipeline(SAMPLE, value, lambda *_: called.append(True))
            blocked = False
        except Exception as exc:
            blocked = not called
            bypass_results[repr(value)] = type(exc).__name__
        if not blocked:
            bypass_results[repr(value)] = "BYPASSED"
    checks.append(check("server-side consent bypass blocked", "BYPASSED" not in bypass_results.values(), bypass_results))

    progress: list[dict[str, Any]] = []
    outbound: list[str] = []
    original_connect = socket.socket.connect
    original_create = socket.create_connection
    def guarded_connect(sock: socket.socket, address: Any):
        host = address[0] if isinstance(address, tuple) else str(address)
        if host not in {"127.0.0.1", "localhost", "::1"}:
            outbound.append(repr(address))
            raise RuntimeError("outbound network attempt blocked")
        return original_connect(sock, address)
    def guarded_create(address: Any, *args: Any, **kwargs: Any):
        host = address[0]
        if host not in {"127.0.0.1", "localhost", "::1"}:
            outbound.append(repr(address))
            raise RuntimeError("outbound network attempt blocked")
        return original_create(address, *args, **kwargs)
    socket.socket.connect = guarded_connect
    socket.create_connection = guarded_create
    start = time.perf_counter()
    try:
        def integrated(path: str, cb: Any):
            return demo.process_video(path, cb, consent=True, output_video=VIDEO_OUT, output_log=LOG_OUT)
        result = app.run_pipeline(str(SAMPLE), True, integrated, progress=lambda frac, desc="": progress.append({"fraction": frac, "description": desc}))
    finally:
        socket.socket.connect = original_connect
        socket.create_connection = original_create
    elapsed = time.perf_counter() - start
    checks.append(check("real integrated callback", isinstance(result, app.PipelineResult) and VIDEO_OUT.is_file() and LOG_OUT.is_file(), {"elapsed_seconds": elapsed, "progress_updates": len(progress), "last_progress": progress[-1] if progress else None}))
    checks.append(check("no outbound non-loopback attempts", not outbound, outbound))
    checks.append(check("progress reaches completion", bool(progress) and progress[-1]["fraction"] == 1.0, progress[-5:]))

    src_info, out_info = video_info(SAMPLE), video_info(VIDEO_OUT)
    codec = (out_info.get("ffprobe", {}).get("streams") or [{}])[0].get("codec_name") if isinstance(out_info.get("ffprobe"), dict) else None
    checks.append(check("video codec/frame count", codec == "h264" and out_info["decoded_frames"] == src_info["decoded_frames"] and out_info["frames"] == src_info["frames"], {"source": src_info, "output": out_info}))
    records = audit.verify_log(LOG_OUT)
    checks.append(check("download artifacts valid", VIDEO_OUT.stat().st_size > 0 and len(records) > 1, {"video_bytes": VIDEO_OUT.stat().st_size, "log_bytes": LOG_OUT.stat().st_size, "audit_records": len(records)}))

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    expected = independent_summary(LOG_OUT, 1.0 / src_info["fps"], cfg)
    checks.append(check("identity cards equal independent audit summary", result.identities == expected, {"callback": result.identities, "independent": expected}))
    cards_html = app.render_identity_cards(result.identities)
    expected_fragments = []
    for card in expected:
        expected_fragments.extend([html.escape(card["display_name"]), f'{card["confidence"]:.2f}', app._fmt_ts(card["first_seen"]), app._fmt_ts(card["last_seen"]), f'{card["screen_time_seconds"]:.1f}s'])
    checks.append(check("rendered card content exact", cards_html.count('class="id-card"') == len(expected) and all(fragment in cards_html for fragment in expected_fragments), expected_fragments))

    # The shipped integrated entry point must open the panel while launch starts.
    # Gradio launch blocks by default, so a later webbrowser.open call is unreachable
    # until shutdown and does not satisfy auto-open.
    demo_source = (ROOT / "demo.py").read_text(encoding="utf-8")
    launch_pos = demo_source.find("demo.launch(", demo_source.find("def main("))
    open_pos = demo_source.find("_open_browser(", launch_pos)
    launch_block = demo_source[launch_pos:open_pos if open_pos >= 0 else None]
    auto_open_ok = "inbrowser=True" in launch_block or (0 <= open_pos < launch_pos)
    checks.append(check("integrated panel auto-open", auto_open_ok, {"launch_uses_inbrowser_true": "inbrowser=True" in launch_block, "browser_open_after_blocking_launch": open_pos > launch_pos, "source": "demo.py:415-424"}))

    blocks = app.build_app(process_fn=demo.process_video)
    wiring_checks, gradio_cfg = component_and_dependency_checks(blocks)
    checks.extend(wiring_checks)
    # Use the already-produced real result for the transport probe so the
    # append-only audit output is not overwritten by a second inference run.
    transport_blocks = app.build_app(process_fn=lambda _video, _cb: result)
    server_checks, probes = launch_and_probe(transport_blocks)
    checks.extend(server_checks)

    panel_html = "<!doctype html><meta charset='utf-8'><title>Gate D representative panel</title>" + f"<style>{app.CARD_CSS}</style><h1>Face Recognition Demo — Gate D evidence</h1><div class='disclaimer'>{html.escape(app.DISCLAIMER_TEXT)}</div><h2>Annotated output</h2><video controls width='800' src='output.mp4'></video><h2>Identities</h2>{cards_html}<h2>Downloads</h2><a href='output.mp4' download>output.mp4</a> | <a href='run_log.jsonl' download>run_log.jsonl</a>"
    PANEL.write_text(panel_html, encoding="utf-8")

    passed = all(item["passed"] for item in checks)
    metrics = {
        "gate": "D/T5", "verdict": "PASS" if passed else "FAIL", "generated_at_epoch": time.time(),
        "environment": {"python": sys.version, "platform": platform.platform(), "opencv": cv2.__version__},
        "commands": [".venv/bin/python scripts/verify_models.py", ".venv/bin/python -m pytest -q tests/test_app.py tests/test_render.py tests/test_integration.py", ".venv/bin/python tests/gate_d_eval.py"],
        "checks": checks, "progress": progress, "outbound_non_loopback_attempts": outbound,
        "video": {"source": src_info, "output": out_info}, "identities": result.identities,
        "server_probes": probes, "gradio_config": gradio_cfg,
        "evidence": [str(p.relative_to(ROOT)) for p in (METRICS, PANEL, VIDEO_OUT, LOG_OUT)],
    }
    METRICS.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"verdict": metrics["verdict"], "checks": [{"name": c["name"], "passed": c["passed"]} for c in checks], "metrics": str(METRICS.relative_to(ROOT))}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
