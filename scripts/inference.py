"""Train a BayesFlow posterior and plot it for an observed contact data set.

Example:
    python scripts/inference.py reputation_conversation
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any

import bayesflow as bf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from base.abm import ContactData, Model, Summaries, compute_summaries
from base.summaries import (
    INTERVAL_SECONDS,
    make_summaries,
)
from models import MODEL_REGISTRY


DEFAULT_DATA = ROOT / "data" / "contacts" / "contacts.parquet"


def load_contacts(path: Path) -> tuple[ContactData, int, int]:
    """Load contacts and normalize IDs and times to the simulator convention."""

    frame = pd.read_parquet(path, columns=["t", "i", "j"])
    if frame.empty:
        raise ValueError("the observed contact data is empty")

    times = frame["t"].to_numpy()
    if np.any(times % INTERVAL_SECONDS):
        raise ValueError("contact times must fall on 20-second boundaries")

    agent_ids = np.unique(
        np.concatenate(
            (frame["i"].to_numpy(), frame["j"].to_numpy())
        )
    )
    start = int(times.min())
    normalized_times = times - start + INTERVAL_SECONDS
    contacts = {
        "t": normalized_times.astype(np.int32),
        "i": np.searchsorted(
            agent_ids,
            frame["i"].to_numpy(),
        ).astype(np.int32),
        "j": np.searchsorted(
            agent_ids,
            frame["j"].to_numpy(),
        ).astype(np.int32),
    }
    n_steps = int(normalized_times.max() // INTERVAL_SECONDS)
    return contacts, len(agent_ids), n_steps


def make_workflow(
    model: Model,
    summaries: Summaries,
    *,
    seed: int,
    context: Mapping[str, int],
) -> bf.BasicWorkflow:
    """Build a BayesFlow workflow conditioned on the expert summaries."""

    simulator = model.to_bayesflow_simulator(
        summaries,
        seed=seed,
        **context,
    )
    adapter = model.make_bayesflow_adapter(
        summaries,
        **context,
    ).rename("summary_variables", "inference_conditions")
    return bf.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        inference_network=bf.networks.FlowMatching(),
        inference_variables=list(model.inference_variables or ()),
        inference_conditions=list(summaries),
        standardize="all",
    )


def prepare_posterior_plot_data(
    posterior: Mapping[str, np.ndarray],
    variable_keys: Sequence[str],
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Restore scalar axes and reduce vector parameters to their mean."""

    prepared: dict[str, np.ndarray] = {}
    variable_names: list[str] = []
    for key in variable_keys:
        if key not in posterior:
            raise ValueError(f"posterior is missing inference variable {key!r}")
        values = np.asarray(posterior[key])
        if values.ndim == 2:
            values = values[..., None]
            variable_names.append(key)
        elif values.ndim == 3:
            is_vector = values.shape[-1] > 1
            values = values.mean(axis=-1, keepdims=True)
            variable_names.append(f"{key}_mean" if is_vector else key)
        else:
            raise ValueError(
                f"posterior variable {key!r} must have dataset, draw, and "
                "optional variable axes"
            )
        prepared[key] = values
    return prepared, variable_names


def run_inference(
    model_name: str,
    *,
    data_path: Path = DEFAULT_DATA,
    output_path: Path | None = None,
    epochs: int = 5,
    batches_per_epoch: int = 20,
    batch_size: int = 8,
    posterior_draws: int = 2_000,
    diagnostic_datasets: int = 50,
    diagnostic_draws: int = 200,
    diagnostics_path: Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Train, infer, and return posterior and diagnostic plots."""

    try:
        model = MODEL_REGISTRY[model_name]()
    except KeyError as exc:
        choices = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"unknown model {model_name!r}; available models: {choices}"
        ) from exc

    contacts, n_agents, n_steps = load_contacts(data_path)
    context = {"n_agents": n_agents, "n_steps": n_steps}
    summaries = make_summaries(**context)
    workflow = make_workflow(
        model,
        summaries,
        seed=seed,
        context=context,
    )
    workflow.fit_online(
        epochs=epochs,
        num_batches_per_epoch=batches_per_epoch,
        batch_size=batch_size,
    )

    observed = {
        name: values[None, ...]
        for name, values in compute_summaries(
            contacts,
            summaries,
        ).items()
    }
    posterior = workflow.sample(
        conditions=observed,
        num_samples=posterior_draws,
    )
    variable_keys: Sequence[str] = model.inference_variables or ()
    posterior_for_plot, variable_names = prepare_posterior_plot_data(
        posterior,
        variable_keys,
    )
    grid = bf.diagnostics.plots.pairs_posterior(
        estimates=posterior_for_plot,
        dataset_id=0,
        variable_keys=list(variable_keys),
        variable_names=variable_names,
    )
    plots = {"posterior": grid}

    if diagnostic_datasets > 0:
        test_data = workflow.simulate(diagnostic_datasets)
        diagnostic_samples = workflow.sample(
            conditions=test_data,
            num_samples=diagnostic_draws,
        )
        test_data = dict(test_data)
        diagnostic_samples = dict(diagnostic_samples)
        for name in variable_keys:
            if test_data[name].ndim == 1:
                test_data[name] = test_data[name][..., None]
                diagnostic_samples[name] = diagnostic_samples[name][
                    ..., None
                ]
        diagnostics = workflow.plot_default_diagnostics(
            test_data=test_data,
            samples=diagnostic_samples,
            variable_keys=list(variable_keys),
        )
        plots.update(diagnostics)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        grid.figure.savefig(output_path, dpi=160, bbox_inches="tight")

    if diagnostic_datasets > 0 and (
        diagnostics_path is not None or output_path is not None
    ):
        diagnostics_path = diagnostics_path or output_path.with_name(
            f"{output_path.stem}_diagnostics"
        )
        diagnostics_path.mkdir(parents=True, exist_ok=True)
        for name, figure in plots.items():
            if name != "posterior":
                figure.savefig(
                    diagnostics_path / f"{name}.png",
                    dpi=160,
                    bbox_inches="tight",
                )
    return plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train BayesFlow on a named contact model and save a posterior "
            "pair plot."
        )
    )
    parser.add_argument("model", choices=sorted(MODEL_REGISTRY))
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="observed contacts parquet file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output PNG (default: posterior_<model>.png)",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batches-per-epoch", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--posterior-draws", type=int, default=2_000)
    parser.add_argument(
        "--diagnostic-datasets",
        type=int,
        default=50,
        help="simulated test datasets for diagnostics; use 0 to skip",
    )
    parser.add_argument(
        "--diagnostic-draws",
        type=int,
        default=200,
        help="posterior draws per diagnostic test dataset",
    )
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        help="diagnostic output directory",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--show",
        action="store_true",
        help="also open the pair plot in a window",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or Path(f"posterior_{args.model}.png")
    run_inference(
        args.model,
        data_path=args.data,
        output_path=output,
        epochs=args.epochs,
        batches_per_epoch=args.batches_per_epoch,
        batch_size=args.batch_size,
        posterior_draws=args.posterior_draws,
        diagnostic_datasets=args.diagnostic_datasets,
        diagnostic_draws=args.diagnostic_draws,
        diagnostics_path=args.diagnostics_dir,
        seed=args.seed,
    )
    print(f"Saved posterior pair plot to {output.resolve()}")
    if args.diagnostic_datasets > 0:
        diagnostics_path = args.diagnostics_dir or output.with_name(
            f"{output.stem}_diagnostics"
        )
        print(f"Saved default diagnostics to {diagnostics_path.resolve()}")
    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
