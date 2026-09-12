---
name: model-comparison
description: Runs BayesFlow model comparison for two or more registered models that share an observed dataset, using explicitly configured summary statistics. Use only when explicitly invoked with /model-comparison.
disable-model-invocation: true
---

# Compare models

Do not import or execute the project's scientific Python stack on the local
machine. Read source and generated files instead of running `models`,
`base.observations`, or `scripts.model-comparison` here. The remote helper still
runs `scripts/model-comparison.py` on the instance, or locally if SSH is
unavailable. If a helper command fails because `boto3` is missing, install
only the remote-helper dependency and retry:

```bash
python -m pip install -r requirements-remote.txt
```

Never install `requirements.txt` or the scientific stack to make a remote
command work.

1. Obtain two or more registered model names if the user did not provide them.
   Do not invent a second model, and do not run comparison on a single model.
2. Resolve each name by reading `models/__init__.py` and the model's module;
   identify each model's dataset from that source. Do not `import models`
   locally. Stop if the names are not unique or do not share one dataset.
3. Read `.config/summary.ini` and confirm that shared dataset section lists at
   least one enabled registered name. If not, stop and invite the participant
   to run `/configure-summary-stats`; never select statistics silently. Do not
   run local Python to validate the file.
4. Run the command on the shared instance, falling back to the local machine
   if SSH is unavailable:

   ```bash
   python scripts/aws/remote.py run --fallback-local -- python scripts/model-comparison.py <model> <model>
   ```

   Append any additional compared models and any user-provided epoch,
   simulation, batch, diagnostic, seed, CPU, `--output`, or
   `--diagnostics-dir` options. Use repository-relative output paths only; the
   helper rewrites absolute paths that stay inside the repo. Do not add `--show`
   or other display options. Do not add `--cpus` unless the user asked for one;
   the remote helper injects `--cpus 16` only when SSH works, and the local
   script defaults to `--cpus 4` on fallback.

   Run `remote.py` in the normal shell with stdout and stderr attached. Wait
   until it exits. Do not start a second run while the first is still going;
   a retry would `rsync --delete` the in-progress remote tree.
5. Report whether the run was remote or local, the printed model
   probabilities, the probability figure, prior-predictive pairplot, and
   diagnostics paths that were actually generated, and the enabled summary
   names.
6. If execution fails, diagnose the concrete error. Do not change priors,
   model behavior, or summary implementations unless explicitly requested.

The default figures are
`output/comparisons/<model>_vs_<model>/probabilities.png` and
`output/comparisons/<model>_vs_<model>/prior_predictive.png`. Diagnostics, when
not skipped, are written under that directory's `diagnostics/` folder.
Comparison can be computationally expensive. Before overriding defaults with
larger values, state the requested workload clearly.
