---
name: simulate
description: Runs the prior-predictive simulation workflow for a registered model and reports the generated summary-statistic figure. Use only when explicitly invoked with /simulate.
disable-model-invocation: true
---

# Run simulation

Do not import or execute the project's scientific Python stack on the local
machine. Read source and generated files instead of running `models`,
`base.observations`, or `scripts.simulate` here. The remote helper still
runs `scripts/simulate.py` on the instance, or locally if SSH is unavailable.

1. Obtain a registered model name if the user did not provide one.
2. Resolve it by reading `models/__init__.py` and the model's module; identify
   its dataset from that source. Do not `import models` locally.
3. Read `.config/summary.ini` and confirm that dataset section lists at least
   one enabled registered name. If not, stop and invite the participant to
   run `/configure-summary-stats`; never select statistics silently. Do not
   run local Python to validate the file.
4. Run the command on the shared instance, falling back to the local machine
   if SSH is unavailable:

   ```bash
   python scripts/aws/remote.py run --fallback-local -- python scripts/simulate.py <model>
   ```

   Append any user-provided `--runs`, `--seed`, `--cpus`, or `--output`
   options. Use repository-relative output paths only (for example
   `output/<model>/simulations.png`); the helper rewrites absolute paths that
   stay inside the repo. Do not add `--show` or other display options. Do not
   add `--cpus` unless the user asked for one; the remote helper injects
   `--cpus 16` only when SSH works, and the local script defaults to
   `--cpus 4` on fallback.

   Run `remote.py` in the normal shell with stdout and stderr attached. Wait
   until it exits. Do not start a second run while the first is still going;
   a retry would `rsync --delete` the in-progress remote tree.
5. Report whether the run was remote or local, the saved figure path, and the
   enabled summary names. If execution fails, diagnose the concrete error
   without changing model behavior or summary implementations unless asked.

The default figure is `output/<model>/simulations.png`.
