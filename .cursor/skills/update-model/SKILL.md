---
name: update-model
description: Updates an existing registered probabilistic model while preserving its stable contract unless changes are explicitly confirmed. Use only when explicitly invoked with /update-model.
disable-model-invocation: true
---

# Update a model

For any change to priors, inference variables, or stochastic behavior, read
and follow `SKILLS/add-new-model/SKILL.md` in full. Its specification,
implementation, integration, and verification requirements remain
authoritative; adapt only dataset-specific contracts for non-contact models.

## Establish the change

1. Ask for the registered model name and the desired behavioral change.
2. Resolve the model through `models.MODEL_REGISTRY`, then inspect its dataset
   base, schema, implementation, registry entry, and tests.
3. Determine whether the request changes priors, inference variables,
   stochastic decisions, context, output schema, dataset, stable name, only
   implementation details, or only performance.
4. Preserve the stable name, dataset, public API, output schema, seeded
   behavior, and inference targets unless the user explicitly changes them.

For a localized bug fix or behavior-preserving optimization, ask only the
questions needed for that change. For any altered stochastic behavior, elicit
the affected step-by-step generative program, every changed distribution and
parameterization, and the resulting inference targets. Explain prior choices
pedagogically and obtain confirmation before editing.

## Implement and verify

- Make the smallest coherent change in the dataset model package.
- Keep simulation randomness on the supplied `numpy.random.Generator`.
- Keep scripts free of model-specific logic.
- Keep the report-ready class docstring and `parameter_units` synchronized with
  changes to inference variables or their interpretation.
- After changing a prior, run
  `base.reporting.summarize_priors(model, observations.context)` and verify the
  human-readable prior, natural-scale mean, sigma, and unit. Extend the shared
  reporting helper when adding a distribution family; do not duplicate its
  extraction formulas in this skill or a model.
- Update existing regression tests and add focused coverage for the changed
  behavior, including deterministic seeding and schema validation.
- If the stable name or dataset changes, treat it as an explicit migration and
  update all registry and documentation references.
- Run focused tests and then `python -m unittest discover -s tests`.
