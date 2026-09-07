---
name: inference
description: Runs posterior inference and predictive diagnostics for a registered model using explicitly configured summary statistics. Use only when explicitly invoked with /inference.
disable-model-invocation: true
---

# Run inference

1. Obtain a registered model name if the user did not provide one.
2. Resolve it through `models.MODEL_REGISTRY` and identify its dataset.
3. Check that `.config/summary.ini` contains a valid explicit selection for
   that dataset. If not, stop and invite the participant to run
   `/configure-summary-stats`; never select statistics silently.
4. Run `python scripts/inference.py <model>` with any user-provided training,
   posterior, predictive, diagnostic, seed, CPU, output, or display options.
   Use script defaults for options the user did not specify. Run the command
   in a visible terminal with stdout and stderr attached so progress bars
   stream while it runs. Do not redirect, capture, or suppress its output;
   keep monitoring the command until it exits.
5. Report the posterior, posterior-predictive, and diagnostics paths that were
   actually generated, together with the enabled summary names.
6. If execution fails, diagnose the concrete error. Do not change priors,
   model behavior, or summary implementations unless explicitly requested.

Inference can be computationally expensive. Before overriding defaults with
larger values, state the requested workload clearly.
