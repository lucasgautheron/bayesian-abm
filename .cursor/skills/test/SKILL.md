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
   connectivity (`connection`). The connection check uses the same remote
   Python/conda probe as `python scripts/aws/remote.py check`, so a box that
   accepts SSH but lacks the workshop environment fails here too. Quote each
   `PASS`, `FAIL`, or `SKIP` line and any `Problem` / `Next step` lines. On
   success, quote the check detail. This does not validate a participant's
   model, observed dataset, or summary selection.
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

5. If a check fails because `boto3` is missing, install only the
   remote-helper dependency without waiting for confirmation:

   ```bash
   python -m pip install -r requirements-remote.txt
   ```

   Never install `requirements.txt` for this command. Ask before changing
   AWS resources, installing any other package, or editing configuration.
6. After an accepted correction, rerun `python scripts/test.py` and report
   the three checks again in order. If a check still fails, diagnose the
   new failure rather than repeating the same correction.
