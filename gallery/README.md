# Verified faces (enrollment gallery)

One image per enrolled person. This is the folder to add verified faces to.

1. Drop in `<id>.jpg` — a single, front-facing, well-lit face (>=48 px face size).
2. Add a matching entry to `gallery:` in `../config.yaml`:

   ```yaml
   gallery:
     <id>:
       display_name: Full Name
       role: Enrolled volunteer
   ```
3. Re-run `python demo.py --video <file>`. No code changes needed.

Only add people who have explicitly consented to biometric enrollment.
