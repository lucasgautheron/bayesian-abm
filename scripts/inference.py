"""Train a BayesFlow posterior for a selected observed dataset.

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from base.model import Model
from base.observations import load_observations
from base.summaries import Summaries
from models import resolve_model


def make_workflow(
    model: Model,
    summaries: Summaries,
    *,
    seed: int,
    context: Mapping[str, int],
) -> bf.BasicWorkflow:
    """Build a BayesFlow workflow with direct or learned summaries."""

    simulator = model.to_bayesflow_simulator(
        summaries,
        seed=seed,
        **context,
    )
    adapter = model.make_bayesflow_adapter(
        summaries,
        **context,
    )
    summary_network = model.make_bayesflow_summary_network(**context)
    workflow_kwargs: dict[str, Any]
    if summary_network is None:
        adapter = adapter.rename(
            "summary_variables",
            "inference_conditions",
        )
        workflow_kwargs = {"inference_conditions": list(summaries)}
    else:
        workflow_kwargs = {
            "summary_network": summary_network,
            "summary_variables": [
                name for name in summaries if name != "story_mask"
            ],
        }

    return bf.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        inference_network=bf.networks.FlowMatching(),
        inference_variables=list(model.inference_variables or ()),
        standardize="all",
        **workflow_kwargs,
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


def sample_observations(
    workflow: bf.BasicWorkflow,
    conditions: Mapping[str, np.ndarray],
    *,
    num_samples: int,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Sample posteriors for all observations in bounded condition batches."""

    if batch_size < 1:
        raise ValueError("observation batch size must be positive")
    counts = {np.asarray(values).shape[0] for values in conditions.values()}
    if len(counts) != 1:
        raise ValueError("condition arrays must share an observation axis")
    count = counts.pop()
    if count < 1:
        raise ValueError("at least one observation is required")

    chunks: dict[str, list[np.ndarray]] = {}
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        batch = {
            name: np.asarray(values)[start:stop]
            for name, values in conditions.items()
        }
        sampled = workflow.sample(
            conditions=batch,
            num_samples=num_samples,
        )
        for name, values in sampled.items():
            chunks.setdefault(name, []).append(np.asarray(values))
    return {
        name: np.concatenate(values, axis=0)
        for name, values in chunks.items()
    }


def run_inference(
    model_name: str,
    *,
    output_path: Path | None = None,
    epochs: int = 5,
    batches_per_epoch: int = 20,
    batch_size: int = 8,
    posterior_draws: int = 2_000,
    diagnostic_datasets: int = 50,
    diagnostic_draws: int = 200,
    diagnostics_path: Path | None = None,
    observation_batch_size: int = 256,
    seed: int = 42,
) -> dict[str, Any]:
    """Train, infer, and return posterior and diagnostic plots."""

    model = resolve_model(model_name)
    observations = load_observations(model.dataset)
    context = observations.context
    summaries = observations.summaries
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

    posterior = sample_observations(
        workflow,
        observations.conditions,
        num_samples=posterior_draws,
        batch_size=observation_batch_size,
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
            "Train BayesFlow on a dataset-specific model and save a posterior "
            "pair plot for its first observation."
        )
    )
    parser.add_argument(
        "model",
        help="registered model name (selects observed data automatically)",
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
    parser.add_argument(
        "--observation-batch-size",
        type=int,
        default=256,
        help="observed series processed per posterior batch",
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
        output_path=output,
        epochs=args.epochs,
        batches_per_epoch=args.batches_per_epoch,
        batch_size=args.batch_size,
        posterior_draws=args.posterior_draws,
        diagnostic_datasets=args.diagnostic_datasets,
        diagnostic_draws=args.diagnostic_draws,
        diagnostics_path=args.diagnostics_dir,
        observation_batch_size=args.observation_batch_size,
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
