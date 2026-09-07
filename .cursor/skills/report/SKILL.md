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
4. Run `python scripts/report.py <model> --cpus 4` with any user-provided
   simulation, inference, seed, report-directory, or display options. If the
   user provides a CPU count, use it instead of `4`. Use script defaults for
   all other options the user did not specify. Run the command in a visible
   terminal with stdout and stderr attached so progress bars stream while it
   runs. Do not redirect, capture, or suppress its output; keep monitoring the
   command until it exits.
   The prior-predictive summary pairplot reuses the inference network's
   training simulations; do not run `scripts/simulate.py` separately.
5. Use [`REPORT_TEMPLATE.md`](REPORT_TEMPLATE.md) as the canonical structure
   for the generated `report.md`. Preserve its section order, headings, fixed
   explanatory prose, and figure link labels instead of drafting them again.
   Replace every `{{PLACEHOLDER}}` with the generated model-specific content
   and leave no placeholder syntax behind. Omit a section only when its
   optional figure was not generated; never leave a broken figure link.
6. Rewrite only the `## Model description` paragraph in the generated
   `report.md` after inspecting the selected model's simulator and any helpers
   that define its process. In two to four concrete sentences, explain in
   intuitive terms what entities and relationships the model represents, how
   interactions or events arise over time, and how latent structure or
   individual differences affect them. Convey each important idea in plain
   language first, then give its precise statistical name in parentheses when
   useful. Ground every claim in the implementation. Do not merely reuse the
   model's class docstring or add generic boilerplate such as saying that the
   program maps assumptions to synthetic data.
7. Verify `report.md` and every linked figure, then report the report path,
   enabled summary names, and generated figure paths.
8. If execution fails after producing some figures, identify the completed and
   failed stage without deleting partial results.

The default report directory is `reports/<model>/`. Its `report.md` contains
the model description; a parameter/prior table with natural-scale mean, sigma,
and units; prior-predictive, prior/posterior, and posterior-predictive figures;
and dynamic inference diagnostics. Figures are stored as `simulations.png`,
`posterior.png`, an optional `posterior_predictive.png`, and optional files
under `diagnostics/`.
