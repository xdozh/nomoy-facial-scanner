#!/usr/bin/env python3
"""Reproducible Gate F fresh-machine install / integration / §11 evaluator.

Gate F certifies:
  * T1: a fresh Python 3.10/3.11 venv installs ``requirements.txt`` cleanly,
        models verify, and the README demo command runs start-to-finish;
  * T2–T7: upstream Gate A–E evidence remains present and certified;
  * §11 deliverables are complete (repo, README, samples/gallery, run_log,
    postmortem, consolidated results).

QA-owned: tests/gate_f_eval.py and outputs/gate_f/.  No product modules,
README, models, locks, assets, or existing gate evidence are modified.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
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
EVIDENCE = ROOT / "outputs" / "gate_f"
DEFAULT_VENV = ROOT / ".venv_qa_gatef"
DEFAULT_OUTPUT_VIDEO = ROOT / "outputs" / "output.mp4"
DEFAULT_OUTPUT_LOG = ROOT / "outputs" / "run_log.jsonl"
PYTHON_BIN = shutil.which("python3.11") or shutil.which("python3.10") or shutil.which("python3")


def check(name: str, passed: bool, detail: Any, criterion: str) -> dict[str, Any]:
    return {"name": name, "criterion": criterion, "passed": bool(passed), "detail": detail}


def run(cmd: list[str | Path], *, cwd: Path | None = None, env: dict[str, str] | None = None,
        timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    merged = {**os.environ, **(env or {})}
    return subprocess.run(
        [str(c) for c in cmd],
        cwd=cwd or ROOT,
        env=merged,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def ensure_fresh_venv(venv: Path, checks: list[dict[str, Any]],
                      reuse_if_present: bool) -> Path:
    """Create a fresh venv and install requirements, recording timing."""
    python = venv / "bin" / "python"
    pip = venv / "bin" / "pip"

    if venv.exists():
        if reuse_if_present:
            checks.append(check(
                "fresh venv present and reusable",
                python.is_file(),
                {"venv": str(venv), "python": str(python)},
                "T1",
            ))
            return python
        # Remove and recreate to prove a truly fresh install path.
        shutil.rmtree(venv)

    if not PYTHON_BIN:
        checks.append(check("python 3.10/3.11 interpreter available", False, None, "T1"))
        raise SystemExit(1)

    version = run([PYTHON_BIN, "--version"]).stdout.strip()
    checks.append(check("python interpreter available", bool(version), version, "T1"))

    create = run([PYTHON_BIN, "-m", "venv", str(venv)])
    checks.append(check("fresh venv created", create.returncode == 0, {
        "command": [PYTHON_BIN, "-m", "venv", str(venv)],
        "stderr": create.stderr[-500:],
    }, "T1"))

    upgrade = run([str(python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    checks.append(check("pip tooling upgrade", upgrade.returncode == 0, {
        "stdout": upgrade.stdout[-500:],
        "stderr": upgrade.stderr[-500:],
    }, "T1"))

    started = time.perf_counter()
    install = run([str(pip), "install", "-r", str(ROOT / "requirements.txt")], timeout=600)
    elapsed = time.perf_counter() - started
    checks.append(check(
        "requirements install in fresh venv",
        install.returncode == 0,
        {
            "elapsed_seconds": elapsed,
            "pip_stdout_tail": install.stdout[-1000:],
            "pip_stderr_tail": install.stderr[-1000:],
            "resolver_successful": install.returncode == 0,
        },
        "T1",
    ))

    # Persist a human-readable install transcript for the report.
    transcript = EVIDENCE / "install_log.txt"
    transcript.write_text(
        f"# Fresh venv install transcript\n"
        f"python: {PYTHON_BIN}\n"
        f"venv: {venv}\n"
        f"elapsed_seconds: {elapsed:.2f}\n"
        f"exit_code: {install.returncode}\n\n"
        f"STDOUT:\n{install.stdout}\n\nSTDERR:\n{install.stderr}",
        encoding="utf-8",
    )
    return python


def verify_environment(python: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    """Record an env_report from the fresh venv and verify key imports."""
    env_script = ROOT / "scripts" / "setup_env_report.py"
    if env_script.is_file():
        proc = run([str(python), str(env_script), str(ROOT)])
    else:
        proc = run([str(python), "-c", ENV_REPORT_SNIPPET, str(ROOT)])
    report = json.loads((EVIDENCE / "env_report_fresh.json").read_text(encoding="utf-8")) if (EVIDENCE / "env_report_fresh.json").is_file() else {}

    imports = run([str(python), "-c",
                   "import cv2, numpy, torch, onnxruntime, gradio, ultralytics, boxmot, insightface; print('imports OK')"])
    checks.append(check("fresh venv key imports", imports.stdout.strip() == "imports OK", {
        "stdout": imports.stdout[-500:],
        "stderr": imports.stderr[-500:],
    }, "T1"))
    return report


ENV_REPORT_SNIPPET = r'''
import json, os, platform, sys
import onnxruntime as ort
try:
    import psutil
    cpu = f"{psutil.cpu_count(logical=False)} cores / {psutil.cpu_count(logical=True)} threads"
except Exception:
    cpu = "unknown"
providers = ort.get_available_providers()
if any("CUDA" in p for p in providers):
    gpu = "NVIDIA"
elif "CoreMLExecutionProvider" in providers:
    gpu = "Apple GPU (CoreML)"
else:
    gpu = "None"
report = {
    "os": platform.platform(),
    "machine": platform.machine(),
    "processor": platform.processor(),
    "python": sys.version,
    "python_executable": sys.executable,
    "cpu": cpu,
    "gpu": gpu,
    "onnxruntime_providers": providers,
    "onnxruntime_version": ort.__version__,
}
root = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
out = os.path.join(root, "outputs", "gate_f", "env_report_fresh.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
'''


def verify_models(python: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    proc = run([str(python), str(ROOT / "scripts" / "verify_models.py")])
    passed = proc.returncode == 0 and "PASSED" in proc.stdout
    checks.append(check("model integrity verified", passed, {
        "stdout": proc.stdout[-500:],
        "stderr": proc.stderr[-500:],
    }, "T1/T7"))
    return {"stdout": proc.stdout, "stderr": proc.stderr}


def run_pytest(python: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    proc = run([str(python), "-m", "pytest", "-q", "tests/"], timeout=300)
    m = re.search(r"(\d+) passed", proc.stdout + proc.stderr)
    passed_count = int(m.group(1)) if m else 0
    passed = proc.returncode == 0 and passed_count > 0
    checks.append(check("full test suite", passed, {
        "passed": passed_count,
        "stdout_tail": proc.stdout[-1000:],
        "stderr_tail": proc.stderr[-1000:],
    }, "T1/T2-T7"))
    return {"passed": passed_count, "stdout": proc.stdout, "stderr": proc.stderr}


def run_headless_demo(python: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    out_video = EVIDENCE / "output.mp4"
    out_log = EVIDENCE / "run_log.jsonl"
    for path in (out_video, out_log):
        path.unlink(missing_ok=True)
    started = time.perf_counter()
    proc = run([
        str(python), "demo.py",
        "--video", str(ROOT / "samples" / "test.mp4"),
        "--no-browser",
        "--output-video", str(out_video),
        "--output-log", str(out_log),
    ], timeout=300)
    elapsed = time.perf_counter() - started
    checks.append(check("headless demo completed", proc.returncode == 0, {
        "elapsed_seconds": elapsed,
        "stdout_tail": proc.stdout[-1000:],
        "stderr_tail": proc.stderr[-1000:],
    }, "T1/T5"))

    # Validate output video.
    cap_info = run([str(python), "-c", VIDEO_VALIDATE_SNIPPET, str(out_video)])
    video_meta = json.loads(cap_info.stdout.splitlines()[-1]) if cap_info.returncode == 0 else {}
    checks.append(check("output video playable and correct", bool(
        video_meta.get("open") and video_meta.get("frames") == 300
        and video_meta.get("width") == 1280 and video_meta.get("height") == 720
        and abs(video_meta.get("fps", 0) - 10.0) < 0.1
    ), video_meta, "T1/T5"))

    # Validate audit log.
    log_info = run([str(python), "-c", LOG_VALIDATE_SNIPPET, str(out_log)])
    log_meta = json.loads(log_info.stdout.splitlines()[-1]) if log_info.returncode == 0 else {}
    checks.append(check("audit log verified", bool(
        log_meta.get("verified") and log_meta.get("records") == 307
        and log_meta.get("frame_completes") == 300
    ), log_meta, "T6"))

    return {
        "elapsed_seconds": elapsed,
        "video_meta": video_meta,
        "log_meta": log_meta,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


VIDEO_VALIDATE_SNIPPET = r'''
import cv2, json, sys
p = sys.argv[1]
cap = cv2.VideoCapture(p)
meta = {
    "open": cap.isOpened(),
    "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0,
    "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap.isOpened() else 0,
    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap.isOpened() else 0,
    "fps": cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0.0,
}
cap.release()
print(json.dumps(meta))
'''

LOG_VALIDATE_SNIPPET = r'''
import audit, json, sys
p = sys.argv[1]
try:
    records = audit.verify_log(p)
    fc = [r for r in records if r.get("event") == "frame_complete"]
    print(json.dumps({"verified": True, "records": len(records), "frame_completes": len(fc)}))
except Exception as e:
    print(json.dumps({"verified": False, "error": str(e)}))
'''


def run_exact_default_command(python: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    """Run the exact README default command ``python demo.py --video samples/test.mp4``.

    The process is terminated as soon as it reports the resolved annotated
    video and audit-log paths, so the blocking Gradio server does not remain
    alive.  This validates the default CLI path-resolution behaviour and proves
    the command produces real, verifiable artifacts without overwriting
    immutable prior evidence.
    """
    default_video = DEFAULT_OUTPUT_VIDEO
    default_log = DEFAULT_OUTPUT_LOG
    prior_video_hash = sha256_file(default_video) if default_video.is_file() else None
    prior_log_hash = sha256_file(default_log) if default_log.is_file() else None

    cmd = [
        str(python), "demo.py",
        "--video", str(ROOT / "samples" / "test.mp4"),
    ]
    merged = {**os.environ}
    proc = subprocess.Popen(
        [str(c) for c in cmd],
        cwd=ROOT,
        env=merged,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    q: queue.Queue[str] = queue.Queue()
    lines: list[str] = []

    def _reader() -> None:
        try:
            for line in proc.stdout:
                q.put(line)
        except Exception:
            pass

    reader = threading.Thread(target=_reader)
    reader.start()

    video_line: str | None = None
    log_line: str | None = None
    deadline = time.time() + 300
    try:
        while time.time() < deadline:
            try:
                line = q.get(timeout=0.5)
            except queue.Empty:
                if proc.poll() is not None:
                    break
                continue
            lines.append(line)
            if video_line is None and "Annotated video:" in line:
                video_line = line
            if log_line is None and "Audit log:" in line:
                log_line = line
            if video_line is not None and log_line is not None:
                # Give the process a moment to flush the audit log, then
                # terminate it before the blocking Gradio server starts.
                time.sleep(2)
                break
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        reader.join(timeout=2)
        # Drain any remaining output.
        while not q.empty():
            try:
                lines.append(q.get_nowait())
            except queue.Empty:
                break

    output = "".join(lines)
    video_match = re.search(r"Annotated video:\s*(.+?)(?:\n|\r)", output)
    log_match = re.search(r"Audit log:\s*(.+?)(?:\n|\r)", output)
    produced_video = Path(video_match.group(1).strip()) if video_match else None
    produced_log = Path(log_match.group(1).strip()) if log_match else None

    checks.append(check(
        "exact default command printed annotated video path",
        bool(video_match) and produced_video is not None and produced_video.is_file(),
        {"line": video_line, "path": str(produced_video) if produced_video else None},
        "T1",
    ))
    checks.append(check(
        "exact default command printed audit log path",
        bool(log_match) and produced_log is not None and produced_log.is_file(),
        {"line": log_line, "path": str(produced_log) if produced_log else None},
        "T1",
    ))

    video_meta, log_meta = {}, {}
    if produced_video is not None and produced_video.is_file():
        cap_info = run([str(python), "-c", VIDEO_VALIDATE_SNIPPET, str(produced_video)])
        try:
            video_meta = json.loads(cap_info.stdout.splitlines()[-1]) if cap_info.returncode == 0 else {}
        except Exception:
            video_meta = {}
        checks.append(check("default-command output video playable and correct", bool(
            video_meta.get("open") and video_meta.get("frames") == 300
            and video_meta.get("width") == 1280 and video_meta.get("height") == 720
            and abs(video_meta.get("fps", 0) - 10.0) < 0.1
        ), video_meta, "T1/T5"))

    if produced_log is not None and produced_log.is_file():
        log_info = run([str(python), "-c", LOG_VALIDATE_SNIPPET, str(produced_log)])
        try:
            log_meta = json.loads(log_info.stdout.splitlines()[-1]) if log_info.returncode == 0 else {}
        except Exception:
            log_meta = {}
        checks.append(check("default-command audit log verified", bool(
            log_meta.get("verified") and log_meta.get("records") == 307
            and log_meta.get("frame_completes") == 300
        ), log_meta, "T6"))

    # Confirm immutable prior default evidence is untouched.
    post_video_hash = sha256_file(default_video) if default_video.is_file() else None
    post_log_hash = sha256_file(default_log) if default_log.is_file() else None
    checks.append(check(
        "pre-existing default output.mp4 hash unchanged",
        prior_video_hash is None or prior_video_hash == post_video_hash,
        {"prior": prior_video_hash, "post": post_video_hash},
        "T1",
    ))
    checks.append(check(
        "pre-existing default run_log.jsonl hash unchanged",
        prior_log_hash is None or prior_log_hash == post_log_hash,
        {"prior": prior_log_hash, "post": post_log_hash},
        "T1",
    ))

    return {
        "output": output,
        "produced_video": str(produced_video) if produced_video else None,
        "produced_log": str(produced_log) if produced_log else None,
        "video_meta": video_meta,
        "log_meta": log_meta,
        "prior_video_hash": prior_video_hash,
        "prior_log_hash": prior_log_hash,
    }


def test_explicit_collision_refusal(python: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    """Explicit --output-video/--output-log paths that already exist must fail
    closed before any evidence is overwritten.
    """
    proc = run([
        str(python), "demo.py",
        "--video", str(ROOT / "samples" / "test.mp4"),
        "--no-browser",
        "--output-video", str(DEFAULT_OUTPUT_VIDEO),
        "--output-log", str(DEFAULT_OUTPUT_LOG),
    ], timeout=120)
    stdout_stderr = proc.stdout + proc.stderr
    refused = (
        proc.returncode != 0
        and (
            "Output video already exists" in stdout_stderr
            or "Audit log already exists" in stdout_stderr
        )
        and "Traceback (most recent call last)" not in stdout_stderr
    )
    checks.append(check("explicit output collision refused cleanly", refused, {
        "returncode": proc.returncode,
        "contains_video_guard": "Output video already exists" in stdout_stderr,
        "contains_audit_guard": "Audit log already exists" in stdout_stderr,
        "no_traceback": "Traceback (most recent call last)" not in stdout_stderr,
        "message_tail": stdout_stderr[-500:],
    }, "T1"))
    return {"returncode": proc.returncode, "output": stdout_stderr}


def audit_readme() -> dict[str, Any]:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    numbered_steps = re.findall(r"^\d+\.\s+", readme, flags=re.MULTILINE)
    return {
        "numbered_steps": len(numbered_steps),
        "steps_le_10": len(numbered_steps) <= 10,
        "mentions_license": any(k in readme.lower() for k in ("license", "agpl", "non-commercial")),
        "mentions_consent": "consent" in readme.lower(),
        "mentions_offline": "offline" in readme.lower(),
        "mentions_roadmap": "roadmap" in readme.lower(),
        "has_demo_command": "python demo.py --video samples/test.mp4" in readme,
        "has_pytest_command": "python -m pytest -q tests/" in readme,
    }


def audit_section11() -> dict[str, Any]:
    required = {
        "detect.py": ROOT / "detect.py",
        "track.py": ROOT / "track.py",
        "face.py": ROOT / "face.py",
        "render.py": ROOT / "render.py",
        "app.py": ROOT / "app.py",
        "audit.py": ROOT / "audit.py",
        "demo.py": ROOT / "demo.py",
        "README.md": ROOT / "README.md",
        "samples/test.mp4": ROOT / "samples" / "test.mp4",
        "gallery/alex.jpg": ROOT / "gallery" / "alex.jpg",
        "gallery/blair.jpg": ROOT / "gallery" / "blair.jpg",
        "gallery/casey.jpg": ROOT / "gallery" / "casey.jpg",
        "run_log.jsonl": ROOT / "outputs" / "run_log.jsonl",
        "postmortem.md": ROOT / "postmortem.md",
        "tests/results.md": ROOT / "tests" / "results.md",
    }
    return {name: path.is_file() for name, path in required.items()}


def reconfirm_gate_evidence(checks: list[dict[str, Any]]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    # Gate A: recall >= 90 %, FPS >= 2, offline.
    gate_a = ROOT / "outputs" / "gate_a" / "metrics.json"
    if gate_a.is_file():
        data = json.loads(gate_a.read_text(encoding="utf-8"))
        recall = data.get("recall", {}).get("value", 0.0)
        fps = data.get("performance", {}).get("steady_state_fps", 0.0)
        offline = data.get("runtime", {}).get("outbound_connections_attempted", ["err"]) == []
        results["gate_a"] = {"recall": recall, "fps": fps, "offline": offline, "pass": recall >= 0.9 and fps >= 2 and offline}
    else:
        results["gate_a"] = {"pass": False, "missing": str(gate_a)}

    # Gate B: no swaps, dropout survives.
    gate_b = ROOT / "outputs" / "gate_b" / "metrics.json"
    if gate_b.is_file():
        data = json.loads(gate_b.read_text(encoding="utf-8"))
        v = data.get("verdict", {})
        results["gate_b"] = {"pass": v.get("gate_b_pass", False) and v.get("t3_no_swaps", False)}
    else:
        results["gate_b"] = {"pass": False, "missing": str(gate_b)}

    # Gate C: recognition pass (uses 'gate_decision' key in current metrics).
    gate_c = ROOT / "outputs" / "gate_c" / "metrics.json"
    if gate_c.is_file():
        data = json.loads(gate_c.read_text(encoding="utf-8"))
        v = data.get("gate_decision") or data.get("verdict", {})
        acc = data.get("acceptance", {})
        results["gate_c"] = {
            "pass": v.get("gate_c_pass", False) and v.get("t4_pass", False),
            "acceptance": acc,
        }
    else:
        results["gate_c"] = {"pass": False, "missing": str(gate_c)}

    # Gate D/E: PASS verdict, no failed checks.
    for label in ("gate_d", "gate_e"):
        path = ROOT / "outputs" / label / "metrics.json"
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            results[label] = {"pass": data.get("verdict") == "PASS" and not data.get("failed_checks")}
        else:
            results[label] = {"pass": False, "missing": str(path)}

    for label, data in results.items():
        checks.append(check(f"certified {label} evidence", data["pass"], data, label.split("_")[1].upper()))
    return results


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gate F fresh-install evaluator")
    parser.add_argument("--venv", default=str(DEFAULT_VENV), help="Fresh venv directory")
    parser.add_argument("--reuse-venv", action="store_true", help="Reuse existing venv if present")
    parser.add_argument("--skip-install", action="store_true", help="Skip venv creation/install")
    parser.add_argument("--skip-demo", action="store_true", help="Skip the expensive headless demo")
    args = parser.parse_args(argv)

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    for old in (EVIDENCE / "metrics.json", EVIDENCE / "pytest.log", EVIDENCE / "demo_run.log",
                EVIDENCE / "default_command_collision.log", EVIDENCE / "default_command_exact.log",
                EVIDENCE / "explicit_collision_refusal.log", EVIDENCE / "verify_models.log"):
        old.unlink(missing_ok=True)

    checks: list[dict[str, Any]] = []
    python = Path(args.venv) / "bin" / "python"

    if not args.skip_install:
        python = ensure_fresh_venv(Path(args.venv), checks, args.reuse_venv)

    env_report = verify_environment(python, checks)

    # Save environment report if the helper didn't already.
    env_path = EVIDENCE / "env_report_fresh.json"
    if not env_path.is_file():
        env_path.write_text(json.dumps(env_report, indent=2), encoding="utf-8")

    verify_models(python, checks)

    pytest_result = run_pytest(python, checks)
    (EVIDENCE / "pytest.log").write_text(
        f"exit_code: {0 if any(c['name'] == 'full test suite' and c['passed'] for c in checks) else 1}\n\n"
        f"STDOUT:\n{pytest_result['stdout']}\n\nSTDERR:\n{pytest_result['stderr']}",
        encoding="utf-8",
    )

    if not args.skip_demo:
        exact = run_exact_default_command(python, checks)
        (EVIDENCE / "default_command_exact.log").write_text(exact["output"], encoding="utf-8")

        collision = test_explicit_collision_refusal(python, checks)
        (EVIDENCE / "explicit_collision_refusal.log").write_text(
            f"returncode: {collision['returncode']}\n\n{collision['output']}",
            encoding="utf-8",
        )

        demo_result = run_headless_demo(python, checks)
        (EVIDENCE / "demo_run.log").write_text(
            f"STDOUT:\n{demo_result['stdout']}\n\nSTDERR:\n{demo_result['stderr']}",
            encoding="utf-8",
        )

    readme_audit = audit_readme()
    checks.append(check("README setup ≤10 numbered steps", readme_audit["steps_le_10"], readme_audit, "§11"))
    checks.append(check("README license/consent/offline/roadmap", all([
        readme_audit["mentions_license"], readme_audit["mentions_consent"],
        readme_audit["mentions_offline"], readme_audit["mentions_roadmap"],
    ]), readme_audit, "§11"))

    s11 = audit_section11()
    checks.append(check("§11 deliverables present", all(s11.values()), s11, "§11"))

    gate_evidence = reconfirm_gate_evidence(checks)

    # Compute hashes of the key Gate F artifacts.
    artifact_hashes: dict[str, str] = {}
    for name, path in (
        ("output.mp4", EVIDENCE / "output.mp4"),
        ("run_log.jsonl", EVIDENCE / "run_log.jsonl"),
    ):
        if path.is_file():
            artifact_hashes[name] = sha256_file(path)

    if not args.skip_demo:
        produced_video_path = Path(exact["produced_video"]) if exact.get("produced_video") else None
        produced_log_path = Path(exact["produced_log"]) if exact.get("produced_log") else None
        if produced_video_path and produced_video_path.is_file():
            artifact_hashes[produced_video_path.name] = sha256_file(produced_video_path)
        if produced_log_path and produced_log_path.is_file():
            artifact_hashes[produced_log_path.name] = sha256_file(produced_log_path)

    passed = all(c["passed"] for c in checks)
    metrics = {
        "gate": "F",
        "verdict": "PASS" if passed else "FAIL",
        "generated_at_epoch": time.time(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "fresh_venv_python": str(python),
            "env_report": str(env_path.relative_to(ROOT)),
        },
        "commands": [
            f"{python} -m pip install -r requirements.txt",
            f"{python} scripts/verify_models.py",
            f"{python} -m pytest -q tests/",
            f"{python} demo.py --video samples/test.mp4",
            f"{python} demo.py --video samples/test.mp4 --no-browser --output-video outputs/gate_f/output.mp4 --output-log outputs/gate_f/run_log.jsonl",
        ],
        "checks": checks,
        "readme_audit": readme_audit,
        "section11": s11,
        "upstream_gate_evidence": gate_evidence,
        "artifact_sha256": artifact_hashes,
        "default_command_produced": {
            "video_path": exact.get("produced_video") if not args.skip_demo else None,
            "log_path": exact.get("produced_log") if not args.skip_demo else None,
        },
        "failed_checks": [c["name"] for c in checks if not c["passed"]],
        "evidence": [str(p.relative_to(ROOT)) for p in (
            EVIDENCE / "metrics.json", EVIDENCE / "output.mp4", EVIDENCE / "run_log.jsonl"
        ) if p.is_file()] + [
            exact.get("produced_video"), exact.get("produced_log")
        ] if not args.skip_demo else [],
    }
    (EVIDENCE / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"verdict": metrics["verdict"], "failed_checks": metrics["failed_checks"],
                      "metrics": str((EVIDENCE / "metrics.json").relative_to(ROOT))}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
