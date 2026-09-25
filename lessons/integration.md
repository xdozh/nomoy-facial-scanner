# INTEGRATION agent lessons

- v1 died at Gate D despite T5 passing: `demo.py` launched Gradio with `inbrowser=False`, then called the browser opener only after blocking `launch()` returned, so the running panel never auto-opened.
- Browser opening must occur through Gradio's `inbrowser=True` at launch or before/block-concurrently with server launch. `--no-browser` must still disable opening for QA/headless runs.
- Preserve the passing integrated pipeline, exact card/audit equality, loopback-only server, H.264 output, downloads, and consent enforcement. Re-run the unchanged Gate D evaluator after the focused fix.
- v2 died at Gate F because the exact required default command failed when the delivered append-only `outputs/run_log.jsonl` already existed. The CLI must preserve immutable prior evidence while automatically selecting a collision-free run output and still complete with no extra flags; print the actual output paths clearly and test repeated default runs.
