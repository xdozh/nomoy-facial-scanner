# AUDIT/SECURITY agent lessons

- v1 died at Gate E: startup metadata contained model version strings but not SHA-256 values, `save=False` retained identity-bearing JSONL, and the public CLI exposed raw tracebacks.
- Preserve the passing hash chain, contiguous frame markers, exact event coverage, offline enforcement, tamper refusal, input validation, and clean domain exceptions.
- `AuditLogger(save=False)` must leave no retained identity after close. A pipeline that explicitly requests a durable `output_log` may use `save=True` because run_log.jsonl is a mandatory deliverable; integration owns that call-site policy.
- Integration must pass the full `{version, sha256}` verified model map and catch public CLI errors without a traceback.
