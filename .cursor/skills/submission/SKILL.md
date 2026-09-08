---
name: submission
description: Commits one registered model and its generated report, then pushes them to submissions/<model-name>. Use only when explicitly invoked with /submission.
disable-model-invocation: true
---

# Submit a model

1. Obtain a registered model name if the user did not provide one. Resolve it
   through `models.MODEL_REGISTRY`; never guess a model or submit multiple
   models.
2. Check `reports/<model>/report.md` and verify that every local figure linked
   by the report exists. Treat a missing or incomplete report as unavailable.
3. If no report is available, ask exactly one question before doing any Git
   work: "No report is available for `<model>`. Would you like me to generate
   one first? It will probably take about 5 minutes." If the user agrees,
   read and follow `.cursor/skills/report/SKILL.md`, then recheck the report.
   If they decline, stop without creating a branch or commit.
4. Identify the submission files:
   - the module defining the selected model;
   - registration and export changes required for that model;
   - model-specific tests, if present;
   - every custom summary-statistic implementation used by the report,
     including its required builders, registry and export changes, shared
     dependencies, example-configuration entry, and focused tests; and
   - every file under `reports/<model>/`.
   Include model- and summary-only dependencies when required for the
   submitted model and its report to run. Exclude `.config/summary.ini`, other
   models, unrelated summary statistics, unrelated working tree changes, and
   generated files outside the selected report.
5. Use the `origin` remote when it exists. If it does not, use the only
   configured remote; ask the user when the choice is ambiguous. Fetch the
   target branch before updating an existing submission.
6. Inspect the working tree before changing branches. Preserve all unrelated
   changes: do not discard or stash them. Create or switch to
   `submissions/<model>` without resetting files. If existing local or remote
   branch state cannot be reused safely, stop and explain the conflict rather
   than overwriting or force-updating it.
7. Stage only the submission files and only model- or submitted-summary-related
   hunks in shared registry and export files. Reports are ignored by Git, so
   force-add `reports/<model>/`. Review the staged file list and diff before
   committing. Confirm that no custom statistic named in the report has
   uncommitted implementation, integration, dependency, or test changes left
   outside the staged diff. Unstage anything unrelated.
8. Verify that the selected model still resolves from `models.MODEL_REGISTRY`
   and that every summary statistic used by the report resolves from its
   dataset registry and can summarize the observed data. Verify that the report
   remains complete. Do not create an empty commit. Commit with the message
   `Submit <model> model and report`.
9. Push the commit and set the upstream to
   `<remote>/submissions/<model>`. Never force-push. Report the branch, commit,
   remote, and submitted paths.
