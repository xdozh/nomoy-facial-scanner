# BUILD SPEC — "It Scans & Identifies" Demo
## Instruction file for Claude (or any implementing agent). Non-negotiable. If it doesn't work, it dies.

**Owner:** Shourya Maithani, PSSTEC — September 25, 2026
**Scope:** Demo only. A video file goes in → the system scans every person, tracks them, recognizes them against a pre-loaded gallery, and shows "who is this person." No platform build. No production claims.

---

## 0. Definition of Done (read first — the build fails without ALL of these)

1. `python demo.py --video samples/test.mp4` runs start-to-finish with zero manual intervention on a fresh machine following the README.
2. Output 1: annotated video — every person gets a bounding box + persistent track ID that survives occlusion crossings and camera pans; recognized persons show NAME + confidence, unrecognized show `UNKNOWN`.
3. Output 2: identity panel (web page, auto-opened) — per recognized person: photo, name, confidence, timestamps seen, first/last sighting, total screen time.
4. Output 3: `run_log.jsonl` audit file — every recognition event logged with frame number, track ID, matched identity, confidence, model versions.
5. Gallery swap works: adding `gallery/newperson.jpg` + one line in `config.yaml` makes that person recognized on the next run with no code changes.
6. Runs on CPU-only laptop (slow acceptable: ≥2 FPS processing of 720p video) AND uses GPU automatically if present.
7. No network calls after setup except model downloads on first run. No telemetry. Runs offline in demo mode.
8. All 7 acceptance tests in §9 pass. Any fail = the build dies and gets rebuilt per §10.

---

## 1. Architecture (the whole thing, five stages)

```
VIDEO FILE (mp4) ──► 1. DECODE ──► 2. DETECT persons ──► 3. TRACK (persistent IDs)
                                              │                    │
                                              └──► 4. FACE: detect → embed → match gallery
                                                                   │
                                              5. OUTPUT: annotated video + identity panel + audit log
```

Single-process Python pipeline is acceptable. Clean module boundaries are mandatory (each stage = one module, swappable).

## 2. Tech stack — pinned choices with fallbacks

| Stage | PRIMARY (use this) | Fallback A | Fallback B | Notes |
|---|---|---|---|---|
| Person detection | **Ultralytics YOLO (YOLO11n/s)** — https://github.com/ultralytics/ultralytics | YOLOv8n (same repo) | torchvision Faster R-CNN | `yolo11n.pt` auto-downloads; person class filter (`class=0`) |
| Tracking | **BoxMOT (ByteTrack default; BoT-SORT for ReID-assisted)** — https://github.com/mikel-brostrom/boxmot | `norfair` — https://github.com/tryolabs/norfair | `abewley/sort` — https://github.com/abewley/sort | BoxMOT ships ByteTrack/BoT-SORT/DeepOCSORT pluggable with one API; ~8.3k stars, active 2026 |
| Face detect + recognize | **InsightFace `buffalo_l` pack** — https://github.com/deepinsight/insightface | `facenet-pytorch` — https://github.com/timesler/facenet-pytorch | `ageitgey/face_recognition` — https://github.com/ageitgey/face_recognition | ⚠️ InsightFace pretrained models are RESEARCH/NON-COMMERCIAL licensed — demo/internal use only; flag this in README and in the demo itself |
| Gallery matching | cosine similarity on 512-D embeddings, threshold 0.4 (tune in config) | FAISS index — https://github.com/facebookresearch/faiss | — | Gallery = folder of photos; embed once, cache to `gallery_cache.pkl` |
| Annotation/UI | **OpenCV draw + Gradio app** — https://github.com/gradio-app/gradio | Static HTML panel (no app) | Streamlit | Gradio: upload video → processing bar → playback + identity cards |
| Runtime | Python 3.10–3.11, `onnxruntime` (CPU) / `onnxruntime-gpu` (GPU) | PyTorch CUDA | — | InsightFace runs on onnxruntime — keep GPU path via onnxruntime-gpu, NOT custom CUDA code |
| Reference implementations (READ, don't ship) | multi-cam demo: https://github.com/AarambhDevHub/multi-cam-face-tracker · MTMC ReID logic: https://github.com/samihormi/Multi-Camera-Person-Tracking-and-Re-Identification | ODTrack (flagged-track future): https://github.com/GXNU-ZhongLab/ODTrack · paper: https://arxiv.org/abs/2401.01686 | TorchReID: https://github.com/KaiyangZhou/deep-person-reid | These inform design; do NOT import them into the demo |

Hard out-of-scope for this build: multi-camera ReID, RTSP live streams, eKYC integration, NVIDIA DeepStream/Metropolis, databases. Those are phase-2 per `surveillance-stack-repo-shortlist.md`. Mention them in README as "roadmap" only.

## 3. Gallery = "who's who" (pre-loaded identities)

```
gallery/
  shourya.jpg
  ahmed.jpg
  fatima.jpg
config.yaml:
  gallery:
    shourya: { display_name: "Shourya Maithani", role: "Founder" }
    ahmed:   { display_name: "Ahmed K.",         role: "Engineer" }
    fatima:  { display_name: "Fatima R.",        role: "Analyst" }
  match_threshold: 0.4
  face_min_size_px: 48        # skip faces smaller than this
  embed_every_n_tracks: 1     # run face on every track
  face_check_interval_frames: 15   # re-embed an UNKNOWN track every N frames
```

Rules: one clear frontal photo per person; embeddings cached; UNKNOWN tracks re-attempted periodically as pose improves; a track's identity is decided by majority vote across its matched frames (prevents single-frame flicker).

## 4. Module-by-module spec

### `detect.py`
- YOLO person detection, conf ≥ 0.35, returns `[x1,y1,x2,y2,conf]` per frame.
- CPU: YOLO11n. GPU present: YOLO11s.

### `track.py`
- BoxMOT ByteTrack wrapper. Input: detections → output: track list with persistent integer IDs.
- Track lifecycle: keep track alive 30 frames after last detection (occlusion survival). Log track birth/death to audit log.

### `face.py`
- InsightFace `FaceAnalysis` with `buffalo_l`. On each active track: crop person bbox → detect faces in crop → if face ≥ `face_min_size_px`, embed → cosine match vs gallery cache.
- Majority-vote identity per track over rolling window of last 20 attempts.
- Confidence shown = max similarity. Below threshold → `UNKNOWN`.

### `render.py`
- Draw bbox + `ID: <n>` (unknown) or `NAME — 0.87` (known). Known = green box, unknown = gray, new-this-frame = yellow flash.
- Side strip rendered into video OR separate panel (panel preferred: Gradio).

### `app.py` (Gradio)
- Upload video → progress bar → embedded annotated video player + identity cards grid (photo from gallery, name, confidence, first/last seen timestamps, screen-time seconds) + download links for `output.mp4` and `run_log.jsonl`.
- "Demo disclaimer" banner: demo build, research-licensed models, enrolled volunteers only.

### `audit.py`
- Append-only JSONL: `{frame, ts, track_id, event: track_birth|track_death|match|reidentify, identity, confidence, model_versions}`.
- Model versions + file hashes recorded at startup. This is the seed of the governance ledger from the company blueprint — treat it as immutable.

## 5. Security layer (demo-grade, mandatory)

1. **Local-only runtime:** no outbound network in run mode. Verify with a smoke test that monkeypatches `socket` to fail after model download phase.
2. **Model integrity:** SHA-256 checksums pinned in `models.lock` for `buffalo_l` zip, `yolo11n.pt`; verify on every startup; refuse to run on mismatch.
3. **Input safety:** video decode errors → clean failure with message, never a crash traceback at demo time; max file size 500MB; allowed extensions whitelist.
4. **Audit log:** append-only, no deletes, no edits; it's the demo's governance story — "every recognition is accountable."
5. **Consent gate in UX:** gallery folder README + Gradio banner state "enroll only consenting people." Checkbox required before processing: "I confirm everyone in this video consented."
6. **License flags:** README + UI footnote — InsightFace models = research license; YOLO (Ultralytics) = AGPL-3.0 for the lib — fine for internal demo, flag for commercial use; BoxMOT/Gradio licenses listed.
7. **No identity persistence beyond run:** gallery cache regenerated from photos each run; no database of sightings persists after demo unless `--save` explicitly passed.

## 6. Physical layer

| Item | Required? | Spec |
|---|---|---|
| Demo laptop | Yes | Any 2020+ laptop; CPU-only works (YOLO11n + onnxruntime). With NVIDIA GPU: auto-faster, nothing to configure. |
| Test video | Yes | 30–90s phone video, 720p+, enrolled volunteers moving around, some mutual occlusions, one person who is NOT in the gallery (must show as UNKNOWN) |
| Optional: phone-as-camera | No | `IP Webcam` (Android) RTSP app — ONLY for wow-factor in person; demo core stays video-file based |
| Optional: Jetson Orin Nano | No | Phase-2 hardware story; note in README only |

## 7. Repo / link manifest (everything needed)

| Purpose | URL |
|---|---|
| Person detection | https://github.com/ultralytics/ultralytics |
| Tracking (primary) | https://github.com/mikel-brostrom/boxmot |
| Tracking (fallback) | https://github.com/tryolabs/norfair |
| Face stack (primary) | https://github.com/deepinsight/insightface |
| Face stack (fallback A) | https://github.com/timesler/facenet-pytorch |
| Face stack (fallback B) | https://github.com/ageitgey/face_recognition |
| UI app | https://github.com/gradio-app/gradio |
| Matching index (optional) | https://github.com/facebookresearch/faiss |
| No-code face service (reference/fallback path) | https://github.com/exadel-inc/CompreFace |
| End-to-end demo reference | https://github.com/AarambhDevHub/multi-cam-face-tracker |
| MTMC ReID reference | https://github.com/samihormi/Multi-Camera-Person-Tracking-and-Re-Identification |
| ODTrack (phase-2 tracking upgrade) | https://github.com/GXNU-ZhongLab/ODTrack |
| Metropolis/DeepStream (phase-2 base) | https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps |
| Frigate (phase-2 NVR ingest) | https://github.com/blakeblackshear/frigate |

## 8. Build order (strict)

1. Env bootstrap (`requirements.txt`, pinned versions, `models.lock` checksum script).
2. `detect.py` + annotated render on sample video. **Gate A:** persons boxed correctly.
3. `track.py` integration. **Gate B:** IDs persist through one occlusion in test video.
4. `face.py` + gallery. **Gate C:** 3 enrolled people recognized ≥80% of their visible frames; non-enrolled stays UNKNOWN.
5. `app.py` Gradio + identity panel. **Gate D:** full flow, one click.
6. `audit.py` + security checks (§5). **Gate E:** audit log complete; offline smoke test passes.
7. Polish: README, demo script, screenshots. **Gate F:** fresh-machine install test.

Do not proceed past a gate until it passes. Gates map to subagent kill criteria (see SUBAGENTS file).

## 9. Acceptance tests (all must pass)

| # | Test | Pass condition |
|---|---|---|
| T1 | Fresh install | New venv → `pip install -r requirements.txt` → demo runs, no errors |
| T2 | Detection | ≥90% of person-frames detected in test video |
| T3 | Tracking | No ID swaps in 2 scripted occlusions; ID survives 30-frame dropout |
| T4 | Recognition | Each enrolled person: ≥80% visible frames recognized; unknown person: 0 false accepts |
| T5 | Panel | Identity cards correct: names, confidences, timestamps, screen time |
| T6 | Audit | JSONL complete, hashes present, no gaps in frame sequence |
| T7 | Security | Offline run succeeds; checksum tamper → refusal; oversize file → clean rejection |

## 10. Failure protocol ("if it doesn't work, it dies")

- Any Gate or Test fail: halt the affected module, log root cause to `postmortem.md` (cause, evidence, fix), rebuild the module using Fallback A, then B.
- If all fallbacks fail: replace the module with the simplest thing that passes the test (e.g., recognition via `face_recognition` lib at lower accuracy is acceptable if InsightFace env breaks) — **the demo shipping beats the demo being perfect.**
- Every death + fix recorded in `postmortem.md` with a lesson line — this feeds the subagent learning loop.

## 11. Deliverables checklist

- [ ] Working repo with the structure in §4
- [ ] `README.md`: setup ≤10 steps, demo script, license/license-warning section, roadmap note
- [ ] `samples/test.mp4` + gallery photos (volunteers)
- [ ] `run_log.jsonl` from a full passing run
- [ ] `postmortem.md` (even if it says "no deaths — passed first try," which would be suspicious; expect at least env deaths)
- [ ] All T1–T7 test outputs in `tests/results.md`
