# Build Ledger and Postmortem

## Planning log — 2026-09-25

### Gate sequence and workforce contracts

| Stage | Gate | Agents | Input contract | Output contract |
|---|---|---|---|---|
| Environment bootstrap + detection | A | ENV "Groundskeeper" v1 and DETECT "Eyes" v1 in parallel; QA "Judge" | Build-spec stack/hardware; decoded BGR frames | Reproducible pinned environment, verified model lock/downloader/env report; JSON-serializable person boxes; detection evidence and recall/FPS report |
| Tracking | B | TRACK "Shadow"; QA "Judge" | Detection lists `[x1,y1,x2,y2,conf]` | Track dicts `{track_id,bbox,age}`, lifecycle events, occlusion/ID-swap evidence |
| Recognition | C | FACE "The Identifier"; AUDIT/SECURITY "The Ledger" review; QA "Judge" | Frames, tracks, gallery photos, config | Identity results `{track_id,identity|UNKNOWN,confidence}`, match events, recognition evidence |
| Presentation | D | RENDER "Set Dresser" and APP/UI "Front of House" in parallel after Gate C; QA "Judge" | Frames, tracks, identities, audit-derived summaries | Playable annotated MP4; consent-gated local UI, identity cards, downloads |
| Governance/security | E | AUDIT/SECURITY "The Ledger"; QA "Judge" | Pipeline events, model lock, validated video input | Append-only JSONL, integrity/offline/input-safety evidence, T6/T7 results |
| Integration/polish | F | INTEGRATION "Assembler"; QA "Judge" | All gate-certified modules and evidence | Demo entry point, README ≤10 setup steps, screenshots/assets, consolidated checklist and T1–T7 results |

### Hard-gate policy

No downstream stage starts until QA certifies the predecessor gate with reproducible evidence. A kill-criterion failure retires that incarnation immediately, records `agent | incarnation | cause | evidence | fix applied | lesson`, creates or updates `lessons/<agent>.md`, and spawns the next incarnation with accumulated lessons. Three consecutive deaths switch the module to its specified fallback.

### Initial risk register

| Risk | Likelihood / impact | Expected first failure and mitigation |
|---|---|---|
| Python/package incompatibility on macOS and Python version drift | High / High | ENV is expected to die first because InsightFace, BoxMOT, Torch, and ONNX Runtime pins may conflict. Test a clean venv, preserve resolver logs, then repin or use the specified runtime fallback. |
| Model downloads/checksums unavailable or mutable | High / High | ENV may fail before hashes can be pinned. Separate setup-time download from offline runtime, hash exact artifacts, and refuse mismatches. |
| No real test video or consenting gallery photos | Certain / High | Workspace has neither asset. Generate deterministic synthetic stopgaps immediately; mark recognition tuning and real-world Gate C/T4 evidence provisional until human assets exist. |
| Detector weights require network during tests | Medium / High | Download only in explicit setup phase; tests inject/fake model outputs where appropriate; runtime socket test blocks outbound calls. |
| Tracker API/version churn and 30-frame coast semantics | High / High | TRACK likely fails at integration. Use contract tests around deterministic scripted occlusions and move to norfair/SORT after the mandated death threshold. |
| Synthetic faces are not representative of InsightFace | High / High | FACE accuracy on synthetic assets may fail or be meaningless. Keep zero-false-accept invariant and clearly flag provisional T4 while completing pipeline smoke tests. |
| H.264 encoder unavailable in local OpenCV build | Medium / Medium | RENDER may produce an unplayable file. Probe output with ffprobe/OpenCV and use a supported MP4 codec/fallback transcode. |
| CPU throughput below 2 FPS | Medium / High | Benchmark each stage; use YOLO11n on CPU, interval face inference, and record hardware-specific evidence without weakening correctness gates. |

## Death log

`DETECT | detect-agent-v1 | Gate A recall 58.89% (<90%) and Ultralytics attempted outbound telemetry | tests/results.md; outputs/gate_a/metrics.json | reincarnate v2 with miss analysis, telemetry suppression, and offline regression coverage while preserving conf≥0.35 and CPU YOLO11n | synthetic ground truth must distinguish visible from fully occluded person-frames, and offline behavior must be tested around model initialization`

`DETECT | detect-agent-v2 | Gate A candidate recall 66.67% (<90%) after imgsz tuning | detect-agent-v2 report; unchanged Gate A matching protocol | reincarnate v3 after correcting the stopgap asset's prolonged, non-separable overlaps while retaining telemetry suppression and integrity checks | larger inference resolution cannot detect multiple people when the synthetic compositor erases their visual separability`

`TRACK | track-agent-v1 | Gate B found 3 ID swaps and fragmentation through two scripted crossings | tests/results.md; outputs/gate_b/metrics.json | reincarnate v2 with BoxMOT BoT-SORT/appearance-assisted association while preserving 30-frame coast and lifecycle contracts | ByteTrack motion/IoU association alone cannot preserve identity through full synthetic crossings`

`INTEGRATION | integration-agent-v1 | Gate D panel failed to auto-open because browser opening occurred only after blocking Gradio launch returned | tests/results.md; outputs/gate_d/metrics.json; demo.py launch flow | reincarnate v2 with focused launch-order/inbrowser fix and unchanged QA rerun | browser-open behavior must be scheduled before or by the blocking server launch, while --no-browser remains headless`

`AUDIT/SECURITY | audit-agent-v1 | Gate E found missing startup hashes, retained identities under save=False, and raw CLI traceback exposure | tests/results.md; outputs/gate_e/metrics.json | reincarnate v2 for retention semantics and coordinate full hash metadata/clean CLI wiring with Integration v2 | security acceptance must be verified at public entry points, not only domain-helper boundaries`

`QA/GATE | qa-gate-f-v1 | Gate F was passed despite the exact required default command failing on an existing immutable audit log | tests/results.md Gate F v1; outputs/gate_f/default_command_collision.log | retire verdict, reincarnate QA v2 after Integration v3 makes repeated default runs collision-safe | a workaround command is not evidence that the binding default command runs start-to-finish`

`INTEGRATION | integration-agent-v2 | exact default demo command failed instead of processing because outputs/run_log.jsonl already existed | outputs/gate_f/default_command_collision.log | reincarnate v3 with collision-free immutable run path selection and repeated-default-run tests | mandatory retained evidence and append-only logs require automatic non-destructive output versioning`

---

## Integration Agent — "The Assembler" v1

### Status

Gate-D integration artifacts assembled. The pipeline `python demo.py --video samples/test.mp4 --no-browser` runs start-to-finish offline, verifies checksums, produces `outputs/output.mp4` + `outputs/run_log.jsonl`, and derives audit-backed identity summaries for the Gradio panel. **Final completion (and §11 full sign-off) is pending QA certification of Gates D–F.**

### Integration verification — 2026-09-25

Run from repository root:

```sh
.venv/bin/python scripts/verify_models.py
.venv/bin/python -m pytest -q tests/
.venv/bin/python demo.py --video samples/test.mp4 --no-browser
```

Results:

| Check | Result |
|---|---|
| Model integrity | PASS — `models.lock` verified for YOLO and `buffalo_l` artifacts |
| Full test suite | PASS — 139 passed (incl. 12 new integration tests) |
| Headless end-to-end run | PASS — 300 frames processed offline, 3 identities recognized |
| Annotated output | `outputs/output.mp4` — 300 frames, 1280×720, 10 FPS, playable |
| Audit log | `outputs/run_log.jsonl` — 307 lines, contiguous `frame_complete` markers 0–299, hash chain intact |
| Identity summaries | Alex Synthetic (0.85), Blair Synthetic (0.91), Casey Synthetic (0.86) |

### §11 Deliverables checklist

| Item | Status | Evidence |
|---|---|---|
| Working repo with structure in §4 | ✅ | `detect.py`, `track.py`, `face.py`, `render.py`, `app.py`, `audit.py`, `demo.py` present |
| `README.md`: ≤10 setup/run steps, license warnings, roadmap | ✅ | `README.md` — 10 numbered steps, license table, phase-2 roadmap, synthetic-asset warnings |
| `samples/test.mp4` + gallery photos | ✅ | `samples/test.mp4`, `samples/recognition_test.mp4`, `gallery/{alex,blair,casey}.jpg` |
| `run_log.jsonl` from a full passing run | ✅ | `outputs/run_log.jsonl` produced by `demo.py` |
| `postmortem.md` updated | ✅ | This section appended; prior death log preserved |
| All T1–T7 test outputs in `tests/results.md` | ⏳ | T1/T5–T7 Gate-D evidence pending QA Judge; T2–T4 already certified in `tests/results.md` |

### Known limitations / pending QA gates

- **T4 is provisional** on synthetic LFW-subset face tiles; real-world recognition accuracy remains unvalidated.
- **Gate D (Presentation)** and **Gate E/F (Governance + Integration)** require QA Judge sign-off with reproducible evidence.
- **Rerun caveat:** `outputs/run_log.jsonl` is append-only; delete it or change `--output-log` before re-running.

### Integration lessons (v1 → v2 if needed)

- Keep the audit log as the single source of truth for identity cards; derive first/last seen and screen time from the matched track's lifespan (birth → death/end), not only from `match` event frames, so the panel reports accurate total presence time.
- Passing `config.yaml` and `gallery_dir` through `process_video` keeps gallery swaps code-free.
- Running full InsightFace inference on synthetic sprites inside pytest requires short clips; the integration tests trim the bundled video to 10–30 frame clips to stay under ~1 minute per test while still exercising real face matching.

## Final certification — 2026-09-25

QA Gate F v2 supersedes and invalidates the historical Gate F v1 verdict. The exact default command completed from the delivered workspace without overwriting prior evidence, producing `outputs/output-3.mp4` and `outputs/run_log-3.jsonl`. Explicit output collisions still fail closed. The fresh Python 3.11 environment installed the pinned requirements, verified all locked models, and passed 149 tests.

### Final gate and acceptance status

| Gate / test | Status | Evidence |
|---|---|---|
| Gate A / T2 detection | PASS | `outputs/gate_a/metrics.json` |
| Gate B / T3 tracking | PASS | `outputs/gate_b/metrics.json` |
| Gate C / T4 recognition | PASS, synthetic/provisional | `outputs/gate_c/metrics.json` |
| Gate D / T5 panel | PASS | `outputs/gate_d/metrics.json` |
| Gate E / T6–T7 audit/security | PASS | `outputs/gate_e/metrics.json` |
| Gate F / T1 fresh install | PASS | `outputs/gate_f/metrics.json` |

### Final §11 deliverables

- [x] Working repo with the §4 modules
- [x] README with 10 setup/run steps, license warnings, consent, offline behavior, and roadmap
- [x] `samples/test.mp4`, `samples/recognition_test.mp4`, and gallery photos
- [x] `outputs/run_log.jsonl` from a complete passing run
- [x] Consolidated postmortem and per-agent lessons
- [x] T1–T7 evidence in `tests/results.md`

### Stack deltas

- Detection remained Ultralytics YOLO11n on CPU.
- Tracking moved from ByteTrack to BoxMOT BoT-SORT with a deterministic local color-histogram appearance encoder after ByteTrack ID swaps.
- Recognition remained InsightFace Buffalo-L at threshold 0.4.
- Gradio, OpenCV H.264 rendering, ONNX Runtime, and the hash-linked JSONL audit layer remained on the specified primary path.
