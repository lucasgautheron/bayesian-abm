"""Train a BayesFlow classifier for a selected observed dataset.

Example:
    python scripts/model-comparison.py model_a model_b
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any

import bayesflow as bf
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import ArrayLike, NDArray


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from base.model import Model
from base.observations import condition_batches, load_observations
from base.summaries import Summaries
from models import resolve_model


DEFAULT_DIAGNOSTIC_DATASETS = 100


def resolve_models(
    model_names: Sequence[str],
) -> tuple[Model, ...]:
    """Instantiate and validate the requested model collection."""

    models = [resolve_model(name) for name in model_names]
    return Model.validate_collection(models)


def make_model_comparison(
    models: Sequence[Model],
    summaries: Summaries,
    *,
    seed: int,
    context: Mapping[str, int],
) -> tuple[
    bf.approximators.ModelComparisonApproximator,
    bf.simulators.ModelComparisonSimulator,
]:
    """Build an equal-prior BayesFlow model-comparison approximator."""

    models = Model.validate_collection(models)
    child_seeds = np.random.SeedSequence(seed).spawn(len(models))
    simulators = [
        model.to_bayesflow_simulator(
            summaries,
            seed=model_seed,
            include_parameters=False,
            **context,
        )
        for model, model_seed in zip(models, child_seeds)
    ]
    simulator = bf.simulators.ModelComparisonSimulator(
        simulators=simulators,
        use_mixed_batches=True,
        key_conflicts="error",
    )
    approximator = bf.approximators.ModelComparisonApproximator(
        num_models=len(models),
        classifier_network=bf.networks.MLP(
            widths=(64, 64),
            activation="silu",
            dropout=None,
        ),
        adapter=models[0].make_bayesflow_model_comparison_adapter(summaries),
        standardize="inference_conditions",
    )
    return approximator, simulator


def extract_probabilities(
    prediction: ArrayLike,
    model_names: Sequence[str],
) -> NDArray[np.float64]:
    """Validate and return one probability per compared model."""

    probabilities = np.asarray(prediction, dtype=np.float64)
    if probabilities.ndim == 1:
        probabilities = probabilities[None, :]
    if (
        probabilities.ndim != 2
        or probabilities.shape[1] != len(model_names)
        or probabilities.shape[0] < 1
    ):
        raise ValueError(
            "predicted model probabilities have an unexpected shape "
            f"{probabilities.shape}"
        )
    if (
        not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0)
        or not np.allclose(probabilities.sum(axis=1), 1.0)
    ):
        raise ValueError("predicted model probabilities are invalid")
    return probabilities.mean(axis=0)


def predict_observations(
    approximator: bf.approximators.ModelComparisonApproximator,
    conditions: Mapping[str, np.ndarray],
    *,
    batch_size: int,
) -> NDArray[np.float64]:
    """Predict every observation in bounded condition batches."""

    predictions = []
    for batch in condition_batches(conditions, batch_size):
        predictions.append(
            np.asarray(
                approximator.predict(conditions=batch, probs=True),
                dtype=np.float64,
            )
        )
    return np.concatenate(predictions, axis=0)


def plot_model_probabilities(
    probabilities: ArrayLike,
    model_names: Sequence[str],
    *,
    dataset: str,
) -> Figure:
    """Plot mean posterior probabilities for the observed dataset."""

    probabilities = extract_probabilities(probabilities, model_names)
    width = max(6.0, 1.6 * len(model_names))
    figure, axis = plt.subplots(figsize=(width, 4.5))
    bars = axis.bar(model_names, probabilities, color="tab:blue", alpha=0.8)
    axis.bar_label(
        bars,
        labels=[f"{probability:.3f}" for probability in probabilities],
        padding=3,
    )
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Posterior model probability")
    axis.set_title(
        f"BayesFlow model comparison for observed {dataset}"
    )
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    return figure


def run_model_comparison(
    model_names: Sequence[str],
    *,
    output_path: Path | None = None,
    epochs: int = 5,
    batches_per_epoch: int = 20,
    batch_size: int = 8,
    diagnostic_datasets: int = DEFAULT_DIAGNOSTIC_DATASETS,
    diagnostics_path: Path | None = None,
    observation_batch_size: int = 1_024,
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Train a classifier and compare models on an observed dataset."""

    if epochs < 1 or batches_per_epoch < 1 or batch_size < 1:
        raise ValueError("training sizes must be positive")
    if diagnostic_datasets < 0:
        raise ValueError("diagnostic_datasets must be non-negative")

    models = resolve_models(model_names)
    dataset = models[0].dataset
    names = [model.name for model in models]
    observations = load_observations(dataset)
    context = observations.context
    summaries = observations.summaries
    approximator, simulator = make_model_comparison(
        models,
        summaries,
        seed=seed,
        context=context,
    )
    approximator.fit(
        epochs=epochs,
        num_batches=batches_per_epoch,
        batch_size=batch_size,
        simulator=simulator,
    )

    prediction = predict_observations(
        approximator,
        observations.conditions,
        batch_size=observation_batch_size,
    )
    probabilities = extract_probabilities(prediction, names)
    posterior = plot_model_probabilities(
        probabilities,
        names,
        dataset=dataset,
    )
    plots: dict[str, Any] = {"posterior": posterior}

    if diagnostic_datasets > 0:
        test_data = simulator.sample((diagnostic_datasets,))
        predicted_models = approximator.predict(
            conditions=test_data,
            probs=True,
        )
        true_models = test_data["inference_variables"]
        plots["calibration"] = bf.diagnostics.plots.mc_calibration(
            pred_models=predicted_models,
            true_models=true_models,
            model_names=names,
        )
        plots["confusion_matrix"] = (
            bf.diagnostics.plots.mc_confusion_matrix(
                pred_models=predicted_models,
                true_models=true_models,
                model_names=names,
                normalize="true",
            )
        )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        posterior.savefig(output_path, dpi=160, bbox_inches="tight")

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

    probability_map = dict(zip(names, probabilities.tolist()))
    return plots, probability_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a BayesFlow classifier and compare models on a selected "
            "observed dataset."
        )
    )
    parser.add_argument(
        "models",
        nargs="+",
        help=(
            "two or more models to compare "
            "(their shared observed data is selected automatically)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output PNG (default: model_comparison.png)",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batches-per-epoch", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--diagnostic-datasets",
        type=int,
        default=DEFAULT_DIAGNOSTIC_DATASETS,
        help="simulated test datasets for diagnostics; use 0 to skip",
    )
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        help="diagnostic output directory",
    )
    parser.add_argument(
        "--observation-batch-size",
        type=int,
        default=1_024,
        help="observed series processed per classifier batch",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--show",
        action="store_true",
        help="also open the plots in windows",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or Path(
        f"model_comparison_{'_vs_'.join(args.models)}.png"
    )
    plots, probabilities = run_model_comparison(
        args.models,
        output_path=output,
        epochs=args.epochs,
        batches_per_epoch=args.batches_per_epoch,
        batch_size=args.batch_size,
        diagnostic_datasets=args.diagnostic_datasets,
        diagnostics_path=args.diagnostics_dir,
        observation_batch_size=args.observation_batch_size,
        seed=args.seed,
    )
    print(f"Saved model comparison to {output.resolve()}")
    for name, probability in probabilities.items():
        print(f"{name}: {probability:.3f}")
    if args.diagnostic_datasets > 0:
        diagnostics_path = args.diagnostics_dir or output.with_name(
            f"{output.stem}_diagnostics"
        )
        print(f"Saved diagnostics to {diagnostics_path.resolve()}")
    if args.show:
        plt.show()
    else:
        for figure in plots.values():
            plt.close(figure)


if __name__ == "__main__":
    main()
