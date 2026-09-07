---
name: test
description: Runs the fast local simulation and inference smoke test with a dummy model and data, then diagnoses setup failures. Use only when explicitly invoked with /test.
disable-model-invocation: true
---

# Test the local workflow

1. From the repository root, run:

   ```bash
   python scripts/test.py
   ```

2. Report each completed stage and the total runtime. On success, explain that
   the core simulation and inference workflow works locally; this does not
   validate a participant's model, observed dataset, or summary selection.
3. On failure, identify the first failed stage and use the script's `Problem`
   and `Next step` lines to give one concrete correction. If more evidence is
   needed, rerun:

   ```bash
   python scripts/test.py --verbose
   ```

4. Ask before installing packages, changing environments, or editing
   configuration. Do not change model behavior, observed data, summary
   implementations, or `.config/summary.ini` to make this smoke test pass.
5. After an accepted correction, rerun `python scripts/test.py` and report
   whether every stage passes. If it still fails, diagnose the new first
   failure rather than repeating the same correction.

The test is intentionally independent of workshop datasets and local summary
configuration. It uses fixed seeds, one CPU, temporary artifacts, and a tiny
in-memory model so it should finish quickly.
