# Bayesian modelling

Code and data for simulation-based Bayesian modelling with PyMC and
BayesFlow.

## Table of contents

- [Python requirements](#python-requirements)
- [Installation with Conda (recommended)](#installation-with-conda-recommended)
- [Installation with `venv`](#installation-with-venv)
- [Commands](#commands)
- [Choose summary statistics before running a workshop script](#choose-summary-statistics-before-running-a-workshop-script)

## Python requirements

This project requires **Python 3.11, 3.12, or 3.13**
(`>=3.11,<3.14`). This range is imposed by the pinned
[BayesFlow 2.0.12](https://bayesflow.org/v2.0.12/) dependency. Python 3.12 is
recommended for the broadest compatibility with the scientific Python stack.

## Installation with Conda (recommended)

Conda is useful here for selecting a compatible Python interpreter and keeping
the compiled scientific dependencies isolated. The project dependencies
themselves remain installed from `requirements.txt` with pip.

```bash
conda create --name bayesian-modelling python=3.12 pip
conda activate bayesian-modelling

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install jax
```

BayesFlow requires a Keras backend in addition to the packages in
`requirements.txt`. JAX is the recommended backend in the
[BayesFlow installation guide](https://bayesflow.org/v2.0.12/). Configure it
for this Conda environment, then reactivate the environment so that the setting
takes effect:

```bash
conda env config vars set KERAS_BACKEND=jax
conda deactivate
conda activate bayesian-modelling
```

The command above installs the standard CPU build of JAX. For an NVIDIA GPU or
another accelerator, follow the
[JAX platform-specific installation instructions](https://docs.jax.dev/en/latest/installation.html)
instead of running `python -m pip install jax`.

Verify the environment:

```bash
python -c "import bayesflow, jax, pymc, keras; print(keras.backend.backend())"
```

The command should print `jax`.

## Installation with `venv`

Conda is not required if a compatible Python interpreter is already installed:

```bash
python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install jax
export KERAS_BACKEND=jax
```

Set `KERAS_BACKEND=jax` before importing BayesFlow in each new shell, or add it
to the shell's environment configuration.

## Commands

Run Cursor commands in chat. Run CLI commands from the repository root after
activating the project environment. `—` means that there is no direct
counterpart.

| Task | Cursor | Manual / CLI |
| --- | --- | --- |
| Configure summary statistics | `/configure-summary-stats [MODEL_OR_DATASET]` | Copy `.config/summary.example.ini` to `.config/summary.ini`, then edit the selected dataset's `enabled` list. |
| Add a summary statistic | `/add-summary-stat` | Follow the [summary-statistic workflow](SKILLS/add-summary-statistic/SKILL.md) manually. |
| Add a model | `/add-model` | Follow the [model workflow](SKILLS/add-new-model/SKILL.md) manually. |
| Update a model | `/update-model` | Update the implementation, registration, and tests manually, following the [model workflow](SKILLS/add-new-model/SKILL.md) for stochastic changes. |
| Run prior-predictive simulations | `/simulate MODEL [OPTIONS]` | `python scripts/simulate.py MODEL [OPTIONS]` |
| Run posterior inference | `/inference MODEL [OPTIONS]` | `python scripts/inference.py MODEL [OPTIONS]` |
| Generate a model report | `/report MODEL [OPTIONS]` | `python scripts/report.py MODEL [OPTIONS]` |
| Compare models | — | `python scripts/model-comparison.py MODEL [MODEL ...] [OPTIONS]` |

Use `python scripts/<command>.py --help` to list the options accepted by a CLI
command.

## Choose summary statistics before running a workshop script

The simulation and inference scripts require an explicit, local selection of
summary statistics for the dataset used by the requested model. The selection
lives in `.config/summary.ini` and is intentionally not committed.

Run `/configure-summary-stats` in Cursor and name a model or dataset. Cursor
will:

- explain the recommended starting set;
- list every registered option;
- ask whether you want to accept, remove, reorder, or add statistics; and
- wait for your explicit confirmation before writing the local file.

Recommendations are starting points, not mandatory or automatic choices.
There is no maximum number of enabled statistics. For contacts, the proposed
starting configuration is:

```ini
[contacts]
enabled =
    cumulative_network_connectivity
    cumulative_network_clustering
    cumulative_network_degree_variance
```

Story models start with total activity, concentration, and temporal
persistence. Scientist-convention models start with coauthorship coupling,
citation coupling, and the theory-HEP field effect. The exact names,
recommendations, and complete option lists are documented in
`.config/summary.example.ini`.

If none of the existing options captures the feature you care about, run
`/add-summary-stat` to design and implement one of your own. Use `/add-model`
to create a model and `/update-model` to revise an existing one. These Cursor
workflows ask for the necessary statistical specification before editing code.

If the relevant selection is missing, empty, duplicated, or unknown, the
script stops before simulation, proposes the configuration command, and lists
the available names.

Once configured, run any of the workshop entry points normally:

```bash
python scripts/simulate.py latent_network
python scripts/inference.py latent_network
python scripts/report.py latent_network
python scripts/model-comparison.py reputation_conversation latent_network
```

`report.py` trains the inference network and reuses those training simulations
for the prior-predictive summary pairplot, avoiding a second simulation pass.
It writes `reports/<model>/report.md` with a model description, a prior table
containing natural-scale means, standard deviations, and units, the
prior-predictive and prior/posterior plots, an optional posterior-predictive
plot, and default inference diagnostics. The inference workload can be
adjusted with the same options exposed by `inference.py`.
