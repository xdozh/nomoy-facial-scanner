#!/usr/bin/env bash
# Reproducible environment bootstrap for the "It Scans & Identifies" demo.
# Run from anywhere; all paths are relative to this script's parent directory.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[$PYTHON_BIN] not found. Attempting Homebrew installation of python@3.11..." >&2
    if ! command -v brew >/dev/null 2>&1; then
        echo "ERROR: Homebrew is not available and no Python 3.10/3.11 interpreter was found." >&2
        echo "Please install Python 3.10 or 3.11 and re-run with PYTHON_BIN=/path/to/python3.11" >&2
        exit 1
    fi
    HOMEBREW_NO_AUTO_UPDATE=1 brew install python@3.11
    # Homebrew may install to /opt/homebrew/bin even when it is not first on PATH.
    PYTHON_BIN="$(command -v python3.11 2>/dev/null || echo /opt/homebrew/bin/python3.11)"
fi

PY_MAJOR=$("$PYTHON_BIN" -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$("$PYTHON_BIN" -c 'import sys; print(sys.version_info.minor)')
if [[ "$PY_MAJOR" -ne 3 || "$PY_MINOR" -lt 10 || "$PY_MINOR" -gt 11 ]]; then
    echo "ERROR: Python 3.10 or 3.11 is required, found $PY_MAJOR.$PY_MINOR" >&2
    exit 1
fi

echo "Bootstrapping venv with $PYTHON_BIN ..."
"$PYTHON_BIN" -m venv "$ROOT/.venv"

VENV_PYTHON="$ROOT/.venv/bin/python"
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel
"$VENV_PYTHON" -m pip install -r "$ROOT/requirements.txt"

# First-run model download / integrity check.
"$VENV_PYTHON" "$ROOT/scripts/download_models.py"
"$VENV_PYTHON" "$ROOT/scripts/verify_models.py"

# Generate env_report.json from real local detection.
"$VENV_PYTHON" - "$ROOT" <<'PY'
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
with open(os.path.join(root, "env_report.json"), "w") as f:
    json.dump(report, f, indent=2)
print("env_report.json written.")
PY

echo "Environment ready at $ROOT/.venv"
echo "Activate with: source $ROOT/.venv/bin/activate"
