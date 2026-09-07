---
name: add-summary-stat
description: Adds a finite scalar summary statistic to a dataset registry with explicit mathematical, domain, invariance, and test requirements. Use only when explicitly invoked with /add-summary-stat.
disable-model-invocation: true
---

# Add a summary statistic

Registration makes a statistic available; it must never silently enable the
statistic in `.config/summary.ini`.

## Authoritative requirements

Before asking questions or editing files, read and follow
`SKILLS/add-summary-statistic/SKILL.md` in full. Do not replace, abbreviate, or
weaken its implementation, invariance, validation, integration, or testing
requirements. If this command conflicts with that workflow, the authoritative
workflow wins.

For contact statistics, every requirement applies exactly. For another
dataset, retain every cross-dataset requirement and replace only
contact-specific schema, context, and registry details with that dataset's
existing contracts.

## Specify before editing

1. Ask for the dataset and a stable snake-case registry name.
2. Ask for the mathematical definition, intended interpretation, units or
   range, and the model behavior or parameter it is meant to reveal.
3. Establish required context, behavior for empty or degenerate data, and
   validation for out-of-domain observations.
4. Establish required invariances. Row order must not matter. Statistics over
   agent distributions must also be invariant to agent labels.
5. Confirm that the result is exactly one finite numeric scalar.
6. Resolve ambiguities and confirm the definition before editing.

## Implement

- Put generic contracts in `base/summaries.py` and dataset implementations in
  `datasets/<dataset>/summaries.py`.
- Use the dataset's existing factory and registry pattern; do not create a
  script-local registry.
- For contacts, accept exactly the one-dimensional `t`, `i`, and `j` arrays,
  treat `t` as interval-end times in `INTERVAL_SECONDS`, and include empty
  intervals and isolated agents whenever they belong to the statistic's
  domain.
- Use a factory when output or validation depends on context such as agent
  IDs, `n_agents`, `n_steps`, or time bounds.
- Reuse existing validation and per-call caches where appropriate.
- Export the public implementation and add it to the dataset registry.
- Add the exact name to the available-options comments in
  `.config/summary.example.ini`. Do not add it to a recommended set without a
  separate explicit request.

## Verify

Add focused tests covering a hand-computed result, empty/degenerate data,
invalid factory arguments, out-of-domain data, required invariances, and scalar
finite output shape and dtype. Run focused tests and then
`python -m unittest discover -s tests`.

After registration, tell the participant they may use `/configure-summary-stats`
to consider enabling the new statistic.
