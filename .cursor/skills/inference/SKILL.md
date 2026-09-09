---
name: inference
description: Runs posterior inference and predictive diagnostics for a registered model using explicitly configured summary statistics. Use only when explicitly invoked with /inference.
disable-model-invocation: true
---

# Run inference

Do not import or execute the project's scientific Python stack on the local
machine. Read source and generated files instead of running `models`,
`base.observations`, or `scripts.inference` here. The remote helper still
runs `scripts/inference.py` on the instance, or locally if SSH is unavailable.

1. Obtain a registered model name if the user did not provide one.
2. Resolve it by reading `models/__init__.py` and the model's module; identify
   its dataset from that source. Do not `import models` locally.
3. Read `.config/summary.ini` and confirm that dataset section lists at least
   one enabled registered name. If not, stop and invite the participant to
   run `/configure-summary-stats`; never select statistics silently. Do not
   run local Python to validate the file.
4. Run the command on the shared instance, falling back to the local machine
   if SSH is unavailable:

   ```bash
   python scripts/aws/remote.py run --fallback-local -- python scripts/inference.py <model>
   ```

   Append any user-provided training, posterior, predictive, diagnostic, seed,
   CPU, `--output`, or `--diagnostics-dir` options. Use repository-relative
   output paths only; the helper rewrites absolute paths that stay inside the
   repo. Do not add `--show` or other display options. Do not add `--cpus`
   unless the user asked for one; the remote helper injects `--cpus 16` only
   when SSH works, and the local script defaults to `--cpus 4` on fallback.

   Run `remote.py` in the normal shell with stdout and stderr attached. Wait
   until it exits. Do not start a second run while the first is still going;
   a retry would `rsync --delete` the in-progress remote tree.
5. Report whether the run was remote or local, the posterior,
   posterior-predictive, and diagnostics paths that were actually generated,
   and the enabled summary names.
6. If execution fails, diagnose the concrete error. Do not change priors,
   model behavior, or summary implementations unless explicitly requested.

Inference can be computationally expensive. Before overriding defaults with
larger values, state the requested workload clearly.
