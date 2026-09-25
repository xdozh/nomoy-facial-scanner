# DETECT agent lessons

- v1 died at Gate A: YOLO11n at confidence 0.35 achieved only 53/90 = 58.89% recall under QA's deterministic IoU≥0.50 synthetic protocol. Investigate whether misses are true visible-person misses versus invalid ground-truth treatment of heavily occluded sprites, but do not weaken the required metric without QA evidence.
- Ultralytics attempted `www.google-analytics.com:443` while the socket guard was active. Configure Ultralytics telemetry/settings off before model construction and add a socket-blocked regression test.
- The project virtualenv lacks pytest, so DETECT tests were not runnable there. Keep tests valid and report this environment contract dependency; do not edit ENV-owned requirements.
- Steady-state CPU throughput passed at 56.29 FPS on 1280×720 synthetic video; preserve that margin.
- v2 died at Gate A candidate evaluation: increasing CPU inference size to 960 improved recall only to 60/90 = 66.67%, while throughput remained 29.07 FPS and guarded initialization made zero outbound attempts. Resolution alone cannot recover synthetic people that are not visually separable during prolonged overlap.
- v2 integrity verification and telemetry suppression are worth preserving. Gate evidence still must come from QA, and the project venv still needs an ENV-owned pytest pin.
