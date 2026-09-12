---
name: report
description: Runs the combined simulation and inference pipeline for one registered model and saves every figure under reports/<model>. Use only when explicitly invoked with /report.
disable-model-invocation: true
---

# Generate a model report

Do not import or execute the project's scientific Python stack on the local
machine. Distant `/report` must succeed when only the remote helper and AWS
or SSH tools are available locally. Read source and generated files instead
of running `models`, `base.observations`, or `scripts.report` here. If a
helper command fails because `boto3` is missing, install only the
remote-helper dependency and retry:

```bash
python -m pip install -r requirements-remote.txt
```

Never install `requirements.txt` or the scientific stack to make a remote
command work.

1. Obtain a registered model name if the user did not provide one.
2. Resolve it by reading `models/__init__.py` and the model's module; identify
   its dataset from that source. Do not `import models` locally.
3. Read `.config/summary.ini` and confirm that dataset section lists at least
   one enabled registered name. If not, stop and invite the participant to
   run `/configure-summary-stats`; never select statistics silently. Do not
   run local Python to validate the file.
4. Run the command on the shared instance. Do not fall back locally if SSH
   is unavailable; stop with the helper error instead of running
   `scripts/report.py` here:

   ```bash
   python scripts/aws/remote.py run -- python scripts/report.py <model>
   ```

   Append any user-provided simulation, inference, seed, or `--report-dir`
   options. Use a repository-relative report directory only; the helper
   rewrites absolute paths that stay inside the repo. Do not add `--show` or
   other display options. Do not add `--cpus` unless the user asked for one;
   the remote helper injects `--cpus 16` only when SSH works.

   Run `remote.py` in the normal shell with stdout and stderr attached. Wait
   until it exits. Do not start a second run while the first is still going;
   a retry would `rsync --delete` the in-progress remote tree.

   The prior-predictive summary pairplot reuses the inference network's
   training simulations; do not run `scripts/simulate.py` separately.
   `scripts/report.py` writes a complete `report.md` on the instance,
   including observed summary-statistic rows taken from the same loaded
   conditions used for inference.
5. After the command exits, read the generated `reports/<model>/report.md`
   and figure files. Use [`REPORT_TEMPLATE.md`](REPORT_TEMPLATE.md) only as
   the checklist for section order, headings, fixed explanatory prose, and
   figure link labels. Do not recompute observed summary values, prior
   moments, or other table cells with local Python. If a required table or
   figure is missing, report that the run was incomplete; do not try to
   finish those calculations locally.
6. Rewrite only the `## Model description` paragraph in the generated
   `report.md` after inspecting the selected model's simulator and any helpers
   that define its process. In two to four concrete sentences, explain in
   intuitive terms what entities and relationships the model represents, how
   interactions or events arise over time, and how latent structure or
   individual differences affect them. Convey each important idea in plain
   language first, then give its precise statistical name in parentheses when
   useful. Ground every claim in the implementation. Do not merely reuse the
   model's class docstring or add generic boilerplate such as saying that the
   program maps assumptions to synthetic data.
7. Verify that `report.md` contains one complete summary-statistic row for
   every enabled name and verify every linked figure file exists. Then report
   the report path, enabled summary names, and generated figure paths.
8. If execution fails after producing some figures, identify the completed and
   failed stage without deleting partial results.

The default report directory is `reports/<model>/`. Its `report.md` contains
the model description; a summary-statistics table with names, descriptions,
and observed values; a parameter/prior table with natural-scale mean, sigma,
and units; prior-predictive, prior/posterior, and posterior-predictive figures;
and dynamic inference diagnostics. Figures are stored as `simulations.png`,
`posterior.png`, an optional `posterior_predictive.png`, and optional files
under `diagnostics/`.
