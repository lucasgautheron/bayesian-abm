---
name: configure-summary-stats
description: Guides a participant through explicitly choosing registered summary statistics, proposing dataset defaults and preserving local selections. Use only when explicitly invoked with /configure-summary-stats.
disable-model-invocation: true
---

# Configure summary statistics

Do not choose silently. The participant must explicitly confirm the final
ordered list before `.config/summary.ini` is created or changed.

## Guide the choice

1. Ask for a registered model name or one of the datasets `contacts`,
   `story_daily`, or `scientist_conventions`. Resolve model names through
   `models.MODEL_REGISTRY`.
2. Read `AVAILABLE_SUMMARY_NAMES` and `RECOMMENDED_SUMMARY_NAMES` from
   `base/summary_config.py`, plus the selected dataset's registry and statistic
   docstrings.
3. Present the recommended starting set with a short explanation of what each
   statistic captures, and explicitly recommend accepting that set.
4. Ask which new summary statistic the participant would like to implement.
   They may answer that they do not want to implement one, but this should not be encouraged. 
5. Do not proactively list alternative registered statistics, ask which
   registered statistics to append, or suggest enabling every statistic.
   Honor a participant's direct request for a specific registered statistic.
   Enable every registered statistic only if the participant explicitly asks
   to use all of them.
6. If the participant wants a new implementation, invite them to run
   `/add-summary-stat`. Do not invent an unregistered INI name.
7. Show the final ordered list and obtain explicit confirmation.

## Write and verify

- Create `.config/summary.ini` from `.config/summary.example.ini` if it does
  not exist.
- Update only the selected dataset's multiline `enabled` value.
- Preserve every other dataset section and its ordering.
- Use exact registry names and require at least one enabled statistic.
- Validate the edited file with `base.summary_config.load_summary_names`.
- Report the final enabled names and remind the participant that they can run
  this command again to revise them.

Recommended sets are suggestions, not mandatory minima, but this workflow
should recommend them rather than encourage larger selections. Never modify a
summary implementation or registry during configuration.
