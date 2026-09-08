---
name: simulate
description: Runs the prior-predictive simulation workflow for a registered model and reports the generated summary-statistic figure. Use only when explicitly invoked with /simulate.
disable-model-invocation: true
---

# Run simulation

1. Obtain a registered model name if the user did not provide one.
2. Resolve it through `models.MODEL_REGISTRY` and identify its dataset.
3. Check that `.config/summary.ini` contains a valid explicit selection for
   that dataset. If not, stop and invite the participant to run
   `/configure-summary-stats`; never select statistics silently.
4. Run the command on the shared instance, falling back to the local machine
   if SSH is unavailable:

   ```bash
   python scripts/aws/remote.py run --fallback-local -- python scripts/simulate.py <model>
   ```

   Append any user-provided `--runs`, `--seed`, `--cpus`, `--output`, or
   `--show` options. Do not add `--cpus` unless the user asked for one; the
   remote helper injects `--cpus 16` only when SSH works, and the local
   script defaults to `--cpus 4` on fallback. Run the command in a visible
   terminal with stdout and stderr attached so progress bars stream while it
   runs. Do not redirect, capture, or suppress its output; keep monitoring
   the command until it exits.
5. Report whether the run was remote or local, the saved figure path, and the
   enabled summary names. If execution fails, diagnose the concrete error
   without changing model behavior or summary implementations unless asked.

The default figure is `output/<model>/simulations.png`.
