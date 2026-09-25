# It Scans & Identifies — Local Face-Recognition Demo

A single-machine, offline pipeline: upload a video → detect people → track them
persistently → recognize enrolled identities → get an annotated video + audit
log + identity panel.

> **Demo / research build only.** All biometric recognition is done locally on
the machine you run this on. No video or identity data leaves your laptop.

---

## 1. Quick start

1. **Prerequisites:** macOS or Linux, Python 3.10 or 3.11, ~4 GB free disk space.
2. **Clone / unzip** this repository and `cd` into it.
3. **Run the automated setup** (creates `.venv`, installs pinned packages,
   downloads/verifies models once):
   ```bash
   bash scripts/setup.sh
   ```
4. **Activate the virtual environment**:
   ```bash
   source .venv/bin/activate
   ```
5. **Run the demo** (auto-opens your browser after processing):
   ```bash
   python demo.py --video samples/test.mp4
   ```
   In the UI, upload a video, check the consent box, and click **Run**.
6. **Check the outputs** created in `outputs/`:
   - `output.mp4` — annotated video with bounding boxes + names/confidences
   - `run_log.jsonl` — append-only audit log with every recognition event
   - Repeated runs automatically use numeric suffixes (`output-2.mp4`,
     `run_log-2.jsonl`, …) so prior evidence is never overwritten.
7. **Headless / QA run** (no browser):
   ```bash
   python demo.py --video samples/test.mp4 --no-browser
   ```
8. **Swap the gallery** by dropping a new `<identity>.jpg` into `gallery/` and
   adding one entry under `gallery:` in `config.yaml`. Re-run; no code changes
   are needed.
9. **Inspect the audit trail**:
   ```bash
   head outputs/run_log.jsonl
   ```
10. **Run tests** before declaring the build ready:
    ```bash
    python -m pytest -q tests/
    ```

---

## 2. Important warnings

- **Consent required.** Only process video of people who have explicitly
  consented to biometric enrollment and scanning. A checkbox in the UI enforces
  this; the CLI assumes you have obtained consent.
- **Research / non-commercial models.** InsightFace `buffalo_l` pretrained
  models are licensed for research and non-commercial use only.
- **AGPL-3.0 dependency.** Ultralytics YOLO (used for person detection) is
  AGPL-3.0. Internal/demo use is fine; commercial use requires a separate
  Ultralytics license.
- **Synthetic test assets.** `samples/test.mp4` and
  `samples/recognition_test.mp4` are procedurally generated stopgaps (astronaut
  sprites + LFW-subset face tiles). Recognition accuracy on these assets is
  **provisional** and does not imply real-world performance.
- **No network at runtime.** After setup, the pipeline is forced offline. Any
  outbound connection attempt during processing is treated as a fatal error.

---

## 3. Runtime vs. setup

| Phase | Goes online? | What it does |
|---|---|---|
| `scripts/setup.sh` | **Yes** (once) | Installs packages, downloads YOLO and InsightFace weights, records `env_report.json` and verifies checksums against `models.lock`. |
| `python demo.py ...` | **No** | Loads only local weights, processes the video offline, writes `outputs/output.mp4` + `outputs/run_log.jsonl`, and opens the local panel. |

---

## 4. CLI flags

```bash
python demo.py --video PATH [--no-browser] \
               [--output-video outputs/output.mp4] \
               [--output-log outputs/run_log.jsonl] \
               [--config config.yaml] \
               [--gallery gallery/] \
               [--models models/] \
               [--port 7860]
```

- `--video` — required input path (`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`, `.m4v`).
- `--no-browser` — headless/QA mode; skips the Gradio launch after processing.
- Default output paths are `outputs/output.mp4` and `outputs/run_log.jsonl`.
  Repeated default runs auto-select `output-2.mp4` / `run_log-2.jsonl`, etc.,
  to keep prior evidence intact.
- Explicit `--output-video` / `--output-log` arguments refuse to overwrite an
  existing file (fail closed).

---

## 5. Pipeline architecture

```
VIDEO FILE ──► decode ──► PersonDetector ──► PersonTracker ──► FaceRecognizer
                                                          │
                                                          ▼
                                          audit events / frame_complete
                                                          │
                                                          ▼
                                              annotate + write output.mp4
                                                          │
                                                          ▼
                                              derive identity summaries
                                                          │
                                                          ▼
                                                 Gradio playback panel
```

---

## 6. License summary

| Component | License | Note |
|---|---|---|
| InsightFace `buffalo_l` | Research / non-commercial | Demo and internal R&D only. |
| Ultralytics YOLO | AGPL-3.0 | Commercial use requires a separate license. |
| BoxMOT | AGPL-3.0 | Tracking backend; same commercial caveat. |
| Gradio | Apache-2.0 | Local-only UI, no share tunnel enabled. |
| Project integration code | MIT-style (see repository) | Written for this demo. |

For exact package versions see `requirements.txt` and `models.lock`.

---

## 7. Phase-2 roadmap (out of scope for this demo)

- Multi-camera Re-ID across RTSP/ONVIF streams
- NVIDIA DeepStream / Metropolis runtime
- Persistent sightings ledger with tamper-evident hashing
- eKYC integration and opt-in enrollment flow
- Jetson / edge deployment target

---

## 8. Asset provenance

- `gallery/alex.jpg`, `gallery/blair.jpg`, `gallery/casey.jpg` — synthetic
  composite faces generated by `scripts/generate_synthetic_assets.py` for
  reproducible local testing. Replace with real, consenting volunteer photos
  for a production-style demo.
- `samples/test.mp4` — synthetic 300-frame 720p video with three people and
  two scripted crossings, generated by the same script.
- `samples/recognition_test.mp4` — synthetic 120-frame 720p video used for the
  Gate C recognition test; includes one non-enrolled person.

---

## 9. Troubleshooting

- **"Audit log already exists"** — you passed an explicit `--output-log` path
  that already exists; choose a new path or omit the flag to let the demo pick
  the next numeric suffix automatically.
- **"Model checksum mismatch"** — re-run `python scripts/verify_models.py`; if
  it fails, delete the model file and re-run `bash scripts/setup.sh`.
- **No H.264 codec** — the writer falls back through `avc1`, `H264`, `X264`,
  and `mp4v`; if all fail, install an OpenCV build with MP4 support.
