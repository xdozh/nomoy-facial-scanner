# SUBAGENTS — Demo Build Workforce
## Each subagent owns exactly one component. Performs, or dies. Every death teaches the next incarnation.

**Owner:** Shourya Maithani, PSSTEC — September 25, 2026
**Pairs with:** `build-spec-scan-demo.md` (the spec these agents execute)
**Operating rule:** this roster is the org chart for the build. The orchestrator (Claude lead / "JARVIS") assigns work, reviews output against kill criteria, and retires-and-rebuilds failing agents. All agents log to `postmortem.md`.

---

## 0. Org structure

```
FOUNDER (Shourya) ──► ORCHESTRATOR (Claude lead agent)
                          │
        ┌────────┬────────┼────────┬────────┬────────┬────────┐
        ▼        ▼        ▼        ▼        ▼        ▼        ▼
      ENV      DETECT   TRACK    FACE     RENDER   APP/UI   AUDIT/
      AGENT    AGENT    AGENT    AGENT    AGENT    AGENT    SECURITY
                                                             AGENT
        └────────┴────────┴────┬────┴────────┴────────┴────────┘
                               ▼
                        QA / GATE AGENT  (reviews everyone, holds the keys)
                               ▼
                        INTEGRATION AGENT (final assembly + README)
```

Agents do not start work until their input contract's producer has passed its gate. QA Agent has veto over everything. Audit/Security Agent has veto over the run environment.

---

## 1. Shared contract (every subagent)

Each subagent has:
- **Mission** — one component, one responsibility.
- **Inputs / Outputs** — typed, file-based contracts. No agent touches another's internals.
- **KPIs** — measurable.
- **Kill criteria** — objective fail conditions → agent is retired: its work is discarded or salvaged, root cause logged, a rebuilt incarnation (new prompt incorporating the lesson) takes over.
- **Learning loop** — every incarnation appends to its `lessons/<agent>.md`; incarnations are numbered (`track-agent-v1`, `-v2`…). The roster is the product's self-improvement engine.

---

## 2. Roster (9 agents)

### 2.1 ENV AGENT — "Groundskeeper"
- **Mission:** reproducible environment. `requirements.txt` (pinned), `models.lock` checksums, setup script, first-run model downloader.
- **Inputs:** §2 stack pins, §6 hardware. **Outputs:** installable env; `env_report.json` (OS, python, CPU/GPU, onnxruntime provider).
- **KPIs:** fresh-venv install ≤15 min on target laptop; zero manual steps.
- **Kill criteria:** install fails on clean machine; version drift between two installs; model checksum file missing any pinned model.
- **Repos:** ultralytics, insightface, boxmot, gradio, onnxruntime (see build-spec §7).

### 2.2 DETECT AGENT — "Eyes"
- **Mission:** `detect.py`. Person detection per frame.
- **Inputs:** decoded frames. **Outputs:** `[x1,y1,x2,y2,conf]` list per frame (JSON-serializable).
- **KPIs:** ≥90% recall on person-frames in test video; ≥2 FPS CPU / ≥15 FPS GPU.
- **Kill criteria:** recall below bar on test video; crashes on any sample frame; returns malformed boxes.
- **Primary:** Ultralytics YOLO11n/s. **Fallback order:** YOLOv8n → torchvision Faster R-CNN.

### 2.3 TRACK AGENT — "Shadow"
- **Mission:** `track.py`. Persistent IDs, occlusion survival (30-frame coast).
- **Inputs:** detection lists. **Outputs:** track dicts `{track_id, bbox, age}` + birth/death events to AUDIT agent.
- **KPIs:** zero ID swaps in the 2 scripted occlusions; IDs survive full dropout ≤30 frames.
- **Kill criteria:** any ID swap in test occlusions; track fragmentation (one person = 2+ IDs) >1 occurrence per video.
- **Primary:** BoxMOT ByteTrack → BoT-SORT. **Fallback:** norfair → SORT.

### 2.4 FACE AGENT — "The Identifier"
- **Mission:** `face.py`. Crop tracks → InsightFace embed → gallery match; majority-vote identity per track; re-embed UNKNOWNs on interval.
- **Inputs:** frames + tracks + `gallery/` + `config.yaml`. **Outputs:** `{track_id, identity|UNKNOWN, confidence}` + match events to AUDIT agent.
- **KPIs:** enrolled recognized ≥80% of visible frames; **zero false accepts** on the non-enrolled person; name flicker <5% of frames.
- **Kill criteria:** any false accept; recall below bar after threshold tuning within 0.35–0.50 band.
- **Primary:** InsightFace `buffalo_l`. **Fallback:** facenet-pytorch → face_recognition (accept accuracy drop; never accept false accepts).

### 2.5 RENDER AGENT — "Set Dresser"
- **Mission:** `render.py`. Annotated video: boxes, IDs, names+confidence, color code (green=known, gray=unknown, yellow flash=new). Clean, readable, demo-pretty.
- **KPIs:** output plays in stock players (H.264 mp4); text legible at 720p; no frame drops vs input.
- **Kill criteria:** unplayable output; illegible overlays; wrong identity rendered (cross-wiring bug = instant death).

### 2.6 APP/UI AGENT — "Front of House"
- **Mission:** `app.py` Gradio: upload → progress → playback + identity cards + downloads; consent checkbox gate; disclaimer banner.
- **KPIs:** full flow works in 2 clicks post-upload; identity cards match audit log exactly; auto-opens panel on run end.
- **Kill criteria:** card/audit mismatch; bypassable consent checkbox; broken on Chrome.
- **Primary:** Gradio. **Fallback:** static HTML panel generated per run.

### 2.7 AUDIT/SECURITY AGENT — "The Ledger" (proto-CGSO)
- **Mission:** `audit.py` + §5 security layer. Append-only `run_log.jsonl`; `models.lock` verification at startup; offline-run enforcement; input validation; consent UX enforcement.
- **KPIs:** log has zero gaps; tampered checksum → startup refusal; offline smoke test passes.
- **Kill criteria:** any recognition event absent from log; any outbound call during run mode; crash-traceback surfaced to user.
- **Note:** this agent is the demo-incarnation of the CGSO from the AI Company blueprint. Its log format must stay compatible with the future hash-chained ledger.

### 2.8 QA / GATE AGENT — "Judge"
- **Mission:** owns §9 tests T1–T7 + gates A–F (build-spec §8). Runs them, publishes `tests/results.md`, blocks progression on any fail.
- **KPIs:** every gate tested with evidence (video/GIF/log excerpt); results file reproducible.
- **Kill criteria:** a shipped defect that its test suite should have caught; a gate passed without evidence.
- **Special:** QA cannot fix code — only reject with precise failure reports. Fixes route back to the owning agent's next incarnation.

### 2.9 INTEGRATION AGENT — "The Assembler"
- **Mission:** final repo assembly, `README.md`, demo script, screenshots, `postmortem.md` consolidation, deliverables checklist (build-spec §11).
- **KPIs:** README setup ≤10 numbered steps; demo script runnable by Shourya with zero context.
- **Kill criteria:** any §11 checkbox unticked; README step that fails verbatim.

---

## 3. Failure & learning protocol (the "die and improve" loop)

1. **Death event:** kill criterion hit → ORCHESTRATOR retires the agent incarnation immediately. Mid-build deaths are expected; env and face agents statistically die first (env hell, threshold tuning).
2. **Postmortem entry (mandatory format):** `agent | incarnation | cause (one line) | evidence (file/log) | fix applied in next incarnation | lesson`.
3. **Reincarnation:** new agent instance starts with the spec + ALL prior `lessons/<agent>.md`. Never resurrect blind — the lesson file is the memory.
4. **Escalation:** 3 consecutive deaths of the same agent → ORCHESTRATOR switches that module to its spec'd fallback stack (build-spec §2) and logs the stack change.
5. **Global rule:** a fallback that passes is ALWAYS preferred over a primary that's dying — the demo shipping on time outranks the demo being SOTA.
6. **Completion:** build is DONE only when QA certifies T1–T7 and INTEGRATION ticks every §11 box. Then the roster files one final consolidated report: deaths, lessons, stack deltas vs spec.

## 4. Handoff to the bigger system

- This roster is the seed workforce. When the demo graduates to pilot (per `surveillance-stack-repo-shortlist.md` week-1 plan), the same agents get re-specced against the Metropolis/BoxMOT + InsightFace Server stack — same contracts, heavier machinery.
- AUDIT agent's ledger format and QA's gate protocol port directly into the AI Company governance layer (CGSO spec card, `ai-company-governance-security-blueprint.md`).

---

*Roster ends. The build starts the moment the ORCHESTRATOR reads `build-spec-scan-demo.md` + this file. First assignments: ENV AGENT and DETECT AGENT, in parallel.*
