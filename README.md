# Bayesian modelling

Code and data for simulation-based Bayesian modelling with PyMC and
BayesFlow.

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

## Choose summary statistics before running a workshop script

The simulation and inference scripts require an explicit, local selection of
summary statistics for the dataset used by the requested model. The selection
lives in `.config/summary.ini` and is intentionally not committed.

Ask Cursor, for example:

> Configure the contact summary statistics for `latent_network`. Enable mean
> contacts per bin, mean pair contact duration, and cumulative network
> connectivity.

Cursor will create or update `.config/summary.ini` with exact names from the
dataset's Python registry. The ordered format is:

```ini
[contacts]
enabled =
    mean_contacts_per_bin
    mean_pair_contact_duration
    cumulative_network_connectivity
```

The available names and sections for all datasets are documented in
`.config/summary.example.ini`. If the relevant selection is missing, empty,
duplicated, or unknown, the script stops before simulation and reports the
available names.

Once configured, run any of the workshop entry points normally:

```bash
python scripts/simulate.py latent_network
python scripts/inference.py latent_network
python scripts/model-comparison.py reputation_conversation latent_network
```
