---
name: test
description: Probes SSH to the shared workshop instance. Use only when explicitly invoked with /test.
disable-model-invocation: true
---

# Test SSH communication

1. From the repository root, run:

   ```bash
   python scripts/test.py
   ```

2. Report whether SSH reached the shared instance and the script's `PASS` or
   `FAIL` line. On success, quote the SSH detail. This does not validate a
   participant's model,
   observed dataset, or summary selection.
3. Do not start, stop, create, or destroy the instance from `/test`. A
   stopped instance is a failed test; tell the user they can run
   `python scripts/aws/instance.py start` themselves if they want the box up.
4. On failure, use the script's `Problem` and `Next step` lines to give one
   concrete correction. If more evidence is needed, rerun:

   ```bash
   python scripts/test.py --verbose
   ```

5. Ask before changing AWS resources, installing packages, or editing
   configuration.
6. After an accepted correction, rerun `python scripts/test.py` and report
   whether SSH now passes. If it still fails, diagnose the new failure
   rather than repeating the same correction.
