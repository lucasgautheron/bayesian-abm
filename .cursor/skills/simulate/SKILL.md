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
4. Run `python scripts/simulate.py <model>` with any user-provided `--runs`,
   `--seed`, `--cpus`, `--output`, or `--show` options. Use script defaults
   for options the user did not specify. Run the command in a visible terminal
   with stdout and stderr attached so progress bars stream while it runs. Do
   not redirect, capture, or suppress its output; keep monitoring the command
   until it exits.
5. Report the saved figure path and the enabled summary names. If execution
   fails, diagnose the concrete error without changing model behavior or
   summary implementations unless asked.

The default figure is `output/<model>/simulations.png`.
