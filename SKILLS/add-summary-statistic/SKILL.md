---
name: add-summary-statistic
description: Adds and integrates summary statistics for temporal contact data. Use when creating, replacing, or substantially changing a summary function in base/summaries.py or the summaries used by simulation and inference.
---

# Add a summary statistic

Use the contact schema and validation in `base/model.py`, and follow the
interfaces and examples in `base/summaries.py`.

## Requirements

1. Implement reusable statistics in `base/summaries.py`.
2. Accept contact mappings with exactly the one-dimensional `t`, `i`, and `j`
   arrays. Treat `t` as interval-end times measured in
   `INTERVAL_SECONDS`.
3. Return exactly one finite numeric scalar. The tutorial deliberately uses
   direct scalar conditioning rather than vectors or learned summary networks.
   `compute_summaries` converts results to one-element `float32` arrays.
4. Use a factory returning `SummaryFunction` when output shape or validation
   depends on context such as `n_agents`, `n_steps`, agent IDs, or time bounds.
5. Make statistics invariant to contact-row order. Statistics describing an
   agent distribution must also be invariant to agent IDs by reducing the
   distribution to a scalar.
6. Include empty intervals and isolated agents when they are part of the
   statistic's domain. Do not infer the required output shape only from
   observed contacts.
7. Validate factory arguments immediately and reject contacts outside the
   configured domain with clear `ValueError` messages.
8. Export public functions through `base/summaries.py`'s `__all__`.
9. Add a context builder to `SUMMARY_BUILDERS` in `base/summaries.py`.
   Builders accept `n_agents` and `n_steps` and return a configured
   `SummaryFunction`.
10. Use the shared `make_summaries` function from `base/summaries.py` in
    inference, simulation, and model comparison. Do not create script-local
    registries.

## Pair-plot behavior

`scripts/simulate.py` plots every scalar statistic directly.

## Verification

Add focused tests in `tests/test_summaries.py` covering:

- a hand-computed result,
- empty contacts and zero-valued bins or agents,
- invalid factory arguments and out-of-domain contacts,
- contact-row-order invariance,
- agent-relabeling invariance where applicable, and
- output shape and dtype.

Run:

```sh
python3 -m unittest discover -s tests
```
