---
name: add-summary-stat
description: Adds a finite scalar summary statistic to a dataset registry and registers the named statistics in `.config/summary.ini`. Use only when explicitly invoked with /add-summary-stat.
disable-model-invocation: true
---

# Add a summary statistic

`/add-summary-stat` registers named statistics in `.config/summary.ini`.
If they are already in the dataset registry, write the INI only. If they
are new, implement them first, then write the same names into the INI.

## Authoritative requirements

Before asking questions or editing files, read and follow
`SKILLS/add-summary-statistic/SKILL.md` in full. Do not replace, abbreviate, or
weaken its implementation, invariance, validation, integration, or testing
requirements. Local INI registration is this command's extension.

For contact statistics, every requirement applies exactly. For another
dataset, retain every cross-dataset requirement and replace only
contact-specific schema, context, and registry details with that dataset's
existing contracts.

## Already registered names

When the participant names registered statistics, or asks for all available
statistics of a dataset:

1. Resolve the dataset (default `contacts`) and the exact registry names.
2. Do not re-implement them.
3. Register those names in `.config/summary.ini` as specified below.

## Specify before editing

Required only for a statistic that is not yet in the dataset registry.

1. Assume the `contacts` dataset (first tutorial session) unless the
   participant names another dataset, then ask for a stable snake-case
   registry name.
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

## Register in `.config/summary.ini`

- Create `.config/summary.ini` from `.config/summary.example.ini` if it does
  not exist.
- Update only the selected dataset's multiline `enabled` value.
- Append requested names that are not already enabled. Keep existing enabled
  names unless the participant asked to replace the list.
- When the participant asks for all available statistics, enable every
  registered name for that dataset in registry order.
- Preserve every other dataset section and its ordering.
- Use exact registry names and require at least one enabled statistic.
- Validate the edited file with `base.summary_config.load_summary_names`.
- Report the final enabled names.

## Verify

Add focused tests covering a hand-computed result, empty/degenerate data,
invalid factory arguments, out-of-domain data, required invariances, and scalar
finite output shape and dtype. Run focused tests and then
`python -m unittest discover -s tests`.
