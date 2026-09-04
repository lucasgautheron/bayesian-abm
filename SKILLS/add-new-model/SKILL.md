---
name: add-new-model
description: Plans, adds, and registers contact simulation models from probabilistic-program descriptions. Use when creating, replacing, or substantially changing a model in models/.
---

# Add a new model

Follow the interfaces in `base/abm.py` and the concrete example in
`models/reputation.py`.

## Required specification and response

1. Immediately switch to Plan mode when the user requests a model
   implementation. Do not edit model code before planning is complete.
2. Require a verbal, step-by-step description of the model as a probabilistic
   program.
3. Require the user to identify which random variables are prior parameters.
   These variables are the inference targets and must be listed in
   `inference_variables`.
4. Require an explicit probability distribution and parameterization for
   every random variable or stochastic decision.
5. In Plan mode, resolve every vague, missing, or contradictory part of the
   specification by asking focused questions. Do not infer unspecified
   distributions or stochastic behavior.
6. Proceed to implementation only after the probabilistic program,
   distributions, and inference variables are fully specified and confirmed
   by the user.

## Requirements

1. Add the implementation in a focused `models/<model_name>.py` module.
2. Subclass `base.abm.Model`.
3. Give the class a unique, stable `name`.
4. Set `inference_variables` to the prior variables inferred by BayesFlow. Use
   `None` only when every free prior variable is an inference target.
5. Implement `build_prior(**context)` and return a prior-only `pymc.Model`.
   Context-dependent shapes must come from explicit context values such as
   `n_agents`.
6. Implement `simulate(parameters, rng, **context)`. Use only the provided
   `numpy.random.Generator` for randomness so seeded runs are reproducible.
7. Return exactly the contact columns `t`, `i`, and `j`. Each must be a
   one-dimensional `numpy.ndarray` with dtype `np.int32`; all three arrays must
   have equal lengths. Express `t` as interval-end times compatible with
   `base.abm.INTERVAL_SECONDS`.
8. Validate required context and reject invalid values with clear exceptions.
9. Export the model class from its module with `__all__`.
10. Register the class in `models/__init__.py`. Import it, add it to
    `MODEL_CLASSES`, and expose it by its stable `name` in `MODEL_REGISTRY`:

```python
from base.abm import Model
from .new_model import NewModel

MODEL_CLASSES: tuple[type[Model], ...] = (
    NewModel,
)
MODEL_REGISTRY = {model.name: model for model in MODEL_CLASSES}
```

Preserve all existing classes when extending `MODEL_CLASSES`. Export the new
class and registry through `__all__`.

## Integration

- Import `MODEL_REGISTRY` from `models` wherever a CLI resolves a model.
- Do not maintain separate registries in individual scripts.
- Keep model-specific simulation logic out of the scripts.

## Verification

- Add focused tests in `tests/test_<model_name>.py`.
- Test deterministic helper functions and invalid inputs.
- Test that identical seeds produce identical parameters and contacts.
- Test that `models.MODEL_REGISTRY[NewModel.name]` resolves to the new class.
- Run the new tests plus the existing model and ABM tests.
