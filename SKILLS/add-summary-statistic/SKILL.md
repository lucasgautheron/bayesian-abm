---
name: add-summary-statistic
description: Adds and integrates summary statistics for temporal contact data. Use when creating, replacing, or substantially changing a summary function or the summaries used by simulation and inference.
---

# Add a summary statistic

Generic scalar-summary machinery lives in `base/summaries.py`. Dataset-specific
summaries and registries belong in `datasets/<dataset>/summaries.py`.

For contacts, use the schema and validation in `base/model.py`, and follow the
interfaces and examples in `datasets/contacts/summaries.py`.

## Requirements

1. Implement contact statistics in `datasets/contacts/summaries.py`.
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
8. Export public functions through
   `datasets/contacts/summaries.py`'s `__all__`.
9. Add a context builder to `SUMMARY_BUILDERS` in
   `datasets/contacts/summaries.py`.
   Builders accept `n_agents` and `n_steps` and return a configured
   `SummaryFunction`.
10. Use the shared contact `make_summaries` function from
    `datasets/contacts/summaries.py` in inference, simulation, and model
    comparison. Do not create script-local registries. Registration in the
    Python registry makes a statistic available. `/add-summary-stat` also
    writes the named statistics into the local `.config/summary.ini`
    enabled list.

## Local selection

Workshop scripts load enabled names from the ignored
`.config/summary.ini`. `/add-summary-stat` registers named or all-available
statistics in that file. `/configure-summary-stats` is the interactive
chooser. When editing the file: preserve other dataset sections, use exact
registry names, and keep at least one name enabled. Do not change
implementations or registries unless the user explicitly asks to change the
available statistics.

## Pair-plot behavior

`scripts/simulate.py` plots every selected scalar statistic directly.

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
