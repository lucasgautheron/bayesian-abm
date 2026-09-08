# Model report: {{MODEL_TITLE}}

## Model description

{{MODEL_DESCRIPTION}}

## Summary statistics

Summary statistics reduce the observed data to the scalar features used for simulation-based inference. The values below are computed from the observed dataset.

| Statistic | Description | Value |
| --- | --- | ---: |
{{SUMMARY_STATISTIC_ROWS}}

## Parameters and prior distributions

Parameters describe individual or population traits, strategies, environmental features, or latent social structure. Their priors are sampled anew for each simulation; the mean and sigma below are the implied moments on the parameter's natural scale.

| Parameter | Prior | Mean | Sigma | Unit |
| --- | --- | ---: | ---: | --- |
{{PARAMETER_ROWS}}

## Prior-predictive summary statistics

The pairplot shows the joint distribution of the selected summary statistics under repeated prior simulations ({{SUMMARY_NAMES}}), with the observed summaries overlaid. Marginal or pairwise incompatibility indicates model misspecification.

![Prior-predictive summary-statistic pairplot](simulations.png)

## Prior and posterior parameter distributions

Simulation-based inference uses the observed summary statistics $S$ to update the prior $P(θ)$ to the posterior $P(θ \mid S)$. Comparing the two distributions shows the direction and extent of parameter learning supplied by the data.

![Prior and posterior parameter distributions](posterior.png)

## Posterior-predictive summary statistics

Posterior parameter draws are propagated through the simulator and summarization pipeline. Comparing their marginal and pairwise distributions with the observed summaries tests whether the inferred model reproduces the macro-level patterns it was intended to explain.

![Posterior-predictive summary-statistic pairplot](posterior_predictive.png)

## Inference and identification diagnostics

These diagnostics assess whether simulation-based inference is reliable and whether the parameters are properly identified. They help distinguish incompatibility with the data (misspecification) from compatibility without unique parameter recovery (missidentification), which should be resolved before substantive interpretation.

### Losses

![losses diagnostic](diagnostics/losses.png)

### Recovery

![recovery diagnostic](diagnostics/recovery.png)

### Calibration ECDF

![calibration ECDF diagnostic](diagnostics/calibration_ecdf.png)

### Coverage

![coverage diagnostic](diagnostics/coverage.png)

### Z-score contraction

![z-score contraction diagnostic](diagnostics/z_score_contraction.png)
