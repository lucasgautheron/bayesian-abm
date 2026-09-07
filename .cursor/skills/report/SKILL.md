---
name: report
description: Runs the combined simulation and inference pipeline for one registered model and saves every figure under reports/<model>. Use only when explicitly invoked with /report.
disable-model-invocation: true
---

# Generate a model report

1. Obtain a registered model name if the user did not provide one.
2. Resolve it through `models.MODEL_REGISTRY` and identify its dataset.
3. Check that `.config/summary.ini` contains a valid explicit selection for
   that dataset. If not, stop and invite the participant to run
   `/configure-summary-stats`; never select statistics silently.
4. Run `python scripts/report.py <model>` with any user-provided simulation,
   inference, seed, CPU, report-directory, or display options. Use script
   defaults for options the user did not specify.
   The prior-predictive summary pairplot reuses the inference network's
   training simulations; do not run `scripts/simulate.py` separately.
5. Verify `report.md` and every linked figure, then report the report path,
   enabled summary names, and generated figure paths.
6. If execution fails after producing some figures, identify the completed and
   failed stage without deleting partial results.

The default report directory is `reports/<model>/`. Its `report.md` contains
the model description; a parameter/prior table with natural-scale mean, sigma,
and units; prior-predictive, prior/posterior, and posterior-predictive figures;
and dynamic inference diagnostics. Figures are stored as `simulations.png`,
`posterior.png`, an optional `posterior_predictive.png`, and optional files
under `diagnostics/`.
