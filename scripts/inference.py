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
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from base.model import Model
from base.observations import condition_batches, load_observations
from base.summaries import Summaries
from models import resolve_model
from scripts.simulate import summary_frame
from visualization.diagnostics import (
    plot_predictive_summary_pairplot,
    plot_prior_posterior_pairplot,
)


def make_workflow(
    model: Model,
    summaries: Summaries,
    *,
    seed: int,
    context: Mapping[str, int],
) -> bf.BasicWorkflow:
    """Build a BayesFlow workflow conditioned on scalar summaries."""

    simulator = model.to_bayesflow_simulator(
        summaries,
        seed=seed,
        **context,
    )
    adapter = model.make_bayesflow_adapter(summaries, **context)
    return bf.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        inference_network=bf.networks.FlowMatching(),
        inference_variables=list(model.inference_variables or ()),
        inference_conditions=list(summaries),
        standardize="all",
    )


def posterior_parameter_draws(
    posterior: Mapping[str, np.ndarray],
    variable_keys: Sequence[str],
    *,
    dataset_id: int = 0,
    draws: int | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Return stacked inference-variable draws for one observed dataset."""

    selected: dict[str, np.ndarray] = {}
    counts: list[int] = []
    for name in variable_keys:
        if name not in posterior:
            raise ValueError(f"posterior is missing inference variable {name!r}")
        values = np.asarray(posterior[name])
        if values.ndim == 2:
            values = values[dataset_id]
        elif values.ndim == 3:
            values = values[dataset_id]
        else:
            raise ValueError(
                f"posterior variable {name!r} must have dataset and draw axes"
            )
        selected[name] = values
        counts.append(int(values.shape[0]))
    if len(set(counts)) != 1:
        raise ValueError("posterior draws must share a draw axis")
    n_available = counts[0]
    if draws is None or draws == n_available:
        return selected
    if draws < 2:
        raise ValueError("predictive runs must be at least 2")
    if draws > n_available:
        raise ValueError("predictive runs cannot exceed the posterior draws")
    generator = np.random.default_rng() if rng is None else rng
    index = generator.choice(n_available, size=draws, replace=False)
    index.sort()
    return {name: values[index] for name, values in selected.items()}


def sample_observations(
    workflow: bf.BasicWorkflow,
    conditions: Mapping[str, np.ndarray],
    *,
    num_samples: int,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Sample posteriors for all observations in bounded condition batches."""

    chunks: dict[str, list[np.ndarray]] = {}
    for batch in condition_batches(conditions, batch_size):
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
    epochs: int = 32,
    num_simulations: int = 2_000,
    batch_size: int = 16,
    posterior_draws: int = 2_000,
    predictive_runs: int = 100,
    diagnostic_datasets: int = 50,
    diagnostic_draws: int = 200,
    diagnostics_path: Path | None = None,
    observation_batch_size: int = 256,
    seed: int = 42,
) -> dict[str, Any]:
    """Train, infer, and return posterior and diagnostic plots."""

    if num_simulations < 1:
        raise ValueError("num_simulations must be positive")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

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
    with tqdm(
        total=num_simulations,
        desc="Training simulations",
        unit="run",
    ) as progress:
        training_simulator = model.to_bayesflow_simulator(
            summaries,
            seed=seed,
            progress=progress.update,
            **context,
        )
        training_data = training_simulator.sample((num_simulations,))
    workflow.fit_offline(
        training_data,
        epochs=epochs,
        batch_size=batch_size,
    )

    posterior = sample_observations(
        workflow,
        observations.conditions,
        num_samples=posterior_draws,
        batch_size=observation_batch_size,
    )
    variable_keys: Sequence[str] = model.inference_variables or ()
    seeds = np.random.SeedSequence(seed).spawn(3)
    prior = model.sample_prior(
        posterior_draws,
        seed=seeds[0],
        **context,
    )
    posterior_figure = plot_prior_posterior_pairplot(
        prior,
        posterior,
        variable_keys,
    )
    plots = {"posterior": posterior_figure}

    if predictive_runs > 0:
        if predictive_runs < 2:
            raise ValueError("predictive runs must be at least 2")
        with tqdm(
            total=predictive_runs,
            desc="Prior predictive",
            unit="run",
        ) as progress:
            prior_simulator = model.to_bayesflow_simulator(
                summaries,
                seed=seeds[1],
                include_parameters=False,
                progress=progress.update,
                **context,
            )
            prior_simulated = prior_simulator.sample((predictive_runs,))
        prior_frame = summary_frame(prior_simulated, runs=predictive_runs)
        posterior_draws_for_plot = posterior_parameter_draws(
            posterior,
            variable_keys,
            draws=min(predictive_runs, posterior_draws),
            rng=np.random.default_rng(seeds[2]),
        )
        n_posterior_runs = int(
            np.asarray(next(iter(posterior_draws_for_plot.values()))).shape[0]
        )
        with tqdm(
            total=n_posterior_runs,
            desc="Posterior predictive",
            unit="run",
        ) as progress:
            posterior_simulated = model.simulate_summaries(
                posterior_draws_for_plot,
                summaries,
                seed=seeds[2],
                progress=progress.update,
                **context,
            )
        posterior_frame = summary_frame(
            posterior_simulated,
            runs=n_posterior_runs,
        )
        observed_frame = summary_frame(
            observations.conditions,
            runs=observations.count,
        )
        observed = (
            observed_frame.iloc[0].to_dict()
            if observations.count == 1
            else observed_frame
        )
        predictive_figure = plot_predictive_summary_pairplot(
            prior_frame,
            posterior_frame,
            observed,
        )
        plots["posterior_predictive"] = predictive_figure

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
        posterior_figure.savefig(output_path, dpi=160, bbox_inches="tight")
        predictive_figure = plots.get("posterior_predictive")
        if predictive_figure is not None:
            predictive_figure.savefig(
                output_path.with_name("posterior_predictive.png"),
                dpi=160,
                bbox_inches="tight",
            )

    if diagnostic_datasets > 0 and (
        diagnostics_path is not None or output_path is not None
    ):
        diagnostics_path = diagnostics_path or output_path.parent / "diagnostics"
        diagnostics_path.mkdir(parents=True, exist_ok=True)
        for name, figure in plots.items():
            if name not in {"posterior", "posterior_predictive"}:
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
        help="output PNG (default: output/<model>/posterior.png)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=32,
        help="passes over the offline training simulations",
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=2_000,
        help="offline training simulations drawn once before fitting",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--posterior-draws", type=int, default=2_000)
    parser.add_argument(
        "--predictive-runs",
        type=int,
        default=100,
        help="prior and posterior predictive simulations; use 0 to skip",
    )
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
    output = args.output or ROOT / "output" / args.model / "posterior.png"
    run_inference(
        args.model,
        output_path=output,
        epochs=args.epochs,
        num_simulations=args.num_simulations,
        batch_size=args.batch_size,
        posterior_draws=args.posterior_draws,
        predictive_runs=args.predictive_runs,
        diagnostic_datasets=args.diagnostic_datasets,
        diagnostic_draws=args.diagnostic_draws,
        diagnostics_path=args.diagnostics_dir,
        observation_batch_size=args.observation_batch_size,
        seed=args.seed,
    )
    print(f"Saved posterior pair plot to {output.resolve()}")
    if args.predictive_runs > 0:
        predictive_path = output.with_name("posterior_predictive.png")
        print(
            "Saved posterior predictive pair plot to "
            f"{predictive_path.resolve()}"
        )
    if args.diagnostic_datasets > 0:
        diagnostics_path = args.diagnostics_dir or output.parent / "diagnostics"
        print(f"Saved default diagnostics to {diagnostics_path.resolve()}")
    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
