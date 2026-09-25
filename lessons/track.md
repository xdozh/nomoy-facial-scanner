# TRACK agent lessons

- v1 died at Gate B: actual detector→ByteTrack evaluation found 3 ID swaps (frames 77, 226, 227) and fragmentation for synthetic identities 0 and 1, despite 28.56 FPS, zero outbound attempts, and correct 30-frame dropout survival.
- Pure motion/IoU ByteTrack association is insufficient at the two scripted crossings. Preserve the typed output, lifecycle/coasting behavior, and deterministic tests, but add appearance-assisted association using the roster's next primary option, BoT-SORT, without runtime downloads.
- QA evidence is in `outputs/gate_b/metrics.json`; the unchanged evaluator must decide the next gate result.
