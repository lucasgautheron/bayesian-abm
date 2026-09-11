---
name: add-model
description: Adds and registers a dataset-specific probabilistic simulation model after eliciting and confirming its complete generative program. Use only when explicitly invoked with /add-model.
disable-model-invocation: true
---

# Add a model

This is a workshop workflow. Be pedagogical and do not implement an
underspecified probabilistic program.

## Authoritative requirements

Before asking questions or editing files, read and follow
`SKILLS/add-new-model/SKILL.md` in full. Do not replace, abbreviate, or weaken
its specification, implementation, integration, or verification requirements.
If this command conflicts with that workflow, the authoritative workflow wins.

For contact models, every requirement applies exactly. For another dataset,
retain every cross-dataset requirement and replace only contact-specific base
class, context, and output-schema details with that dataset's existing
contracts.

## Specify before editing

1. Assume the model belongs to `contacts` (first tutorial session) unless
   the participant names another dataset: `story_daily` or
   `scientist_conventions`.
2. Ask for a unique stable model name. If it is already registered, direct the
   user to `/update-model`.
3. Read the dataset schema, base model, registry, and one nearby model.
4. Elicit a verbal, step-by-step generative program.
5. Identify every random variable and stochastic decision, including its
   distribution and parameterization.
6. Identify the prior variables that BayesFlow should infer and list them in
   `inference_variables`.
7. Resolve missing or contradictory details with focused questions. Explain
   unfamiliar prior choices in plain language. For dimensional priors, state
   the implied mean and standard deviation in natural units.
8. Present the complete specification and obtain confirmation before editing.

Prefer weakly informative priors centered on the correct order of magnitude,
not a precise expected value.

## Implement

- Add the model under `models/<dataset-family>/`.
- Subclass the matching dataset base class.
- Add a concise class docstring suitable for the generated model report.
- Declare `parameter_units` for inferred parameters with meaningful natural
  units; omit dimensionless parameters.
- Implement `build_prior(**context)` and return a prior-only `pymc.Model`.
- Implement `simulate(parameters, rng, **context)` using only the supplied
  `numpy.random.Generator`.
- Read simple parameters and context values directly where used. Extract
  helpers only for substantive or reused model logic.
- Keep the code visibly aligned with the confirmed generative program.
- Preserve each dataset's native output schema:
  - contacts: one-dimensional `int32` arrays `t`, `i`, and `j`;
  - story daily: a `float32` `mentions` matrix;
  - scientist conventions: an `int8` `preference` vector containing `-1/+1`.
- Register and export the model in its dataset package without removing
  existing models.
- Keep model-specific logic out of workshop scripts.

## Verify

Add `tests/test_<model_name>.py` covering deterministic helpers, invalid
inputs, seeded reproducibility, output schema, and registry resolution. Explore
performance improvements that preserve the confirmed behavior. Run
`base.reporting.summarize_priors(model, observations.context)` and confirm its
prior, natural-scale mean, sigma, and unit columns against `build_prior`.
Extend the shared reporting helper for a new distribution family rather than
duplicating extraction formulas here. Run focused tests and then
`python -m unittest discover -s tests`.
