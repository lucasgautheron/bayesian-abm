---
name: test
description: Checks SSH key access, instance state, and SSH connectivity. Use only when explicitly invoked with /test.
disable-model-invocation: true
---

# Test workshop SSH access

1. From the repository root, run:

   ```bash
   python scripts/test.py
   ```

2. Report the three checks **in this order**: SSH key access (`ssh-key`),
   whether the shared instance is running (`instance`), and SSH
   connectivity (`connection`). Quote each `PASS`, `FAIL`, or `SKIP` line
   and any `Problem` / `Next step` lines. On success, quote the check
   detail. This does not validate a participant's model, observed dataset,
   or summary selection.
3. Do not start, stop, create, or destroy the instance from `/test`. A
   stopped instance fails the instance check and skips the connection
   check; tell the user they can run
   `python scripts/aws/instance.py start` themselves if they want the box
   up.
4. On failure, use the script's `Problem` and `Next step` lines to give one
   concrete correction. If more evidence is needed, rerun:

   ```bash
   python scripts/test.py --verbose
   ```

5. Ask before changing AWS resources, installing packages, or editing
   configuration.
6. After an accepted correction, rerun `python scripts/test.py` and report
   the three checks again in order. If a check still fails, diagnose the
   new failure rather than repeating the same correction.
