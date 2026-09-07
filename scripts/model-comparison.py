"""Train a BayesFlow classifier for a selected observed dataset.

Example:
    python scripts/model-comparison.py model_a model_b
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
import sys
from typing import Any

import bayesflow as bf
import keras
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from base.model import Model
from base.observations import condition_batches, load_observations
from base.summaries import Summaries
from base.summary_config import SummaryConfigurationError, load_summary_names
from models import resolve_model
from scripts.parallel import (
    sample_model_comparison_in_processes,
    validate_cpus,
)
from visualization.diagnostics import plot_model_probabilities


DEFAULT_DIAGNOSTIC_DATASETS = 100
INITIAL_LEARNING_RATE = 5.0e-4
WEIGHT_DECAY = 5.0e-3
CLIP_NORM = 1.5

Seed = int | np.integer | np.random.SeedSequence


def _seed_sequence(seed: Seed) -> np.random.SeedSequence:
    """Return ``seed`` as a seed sequence without respawning it."""

    if isinstance(seed, np.random.SeedSequence):
        return seed
    return np.random.SeedSequence(seed)


def _seed_integer(seed: np.random.SeedSequence) -> int:
    """Return one legacy-NumPy-compatible seed."""

    return int(seed.generate_state(1, dtype=np.uint32)[0])


def resolve_models(
    model_names: Sequence[str],
) -> tuple[Model, ...]:
    """Instantiate and validate the requested model collection."""

    models = [resolve_model(name) for name in model_names]
    return Model.validate_collection(models)


def make_model_comparison_simulator(
    models: Sequence[Model],
    summaries: Summaries,
    *,
    seed: Seed,
    context: Mapping[str, int],
    progress: Callable[[int], object] | None = None,
) -> bf.simulators.ModelComparisonSimulator:
    """Build an equal-prior simulator with independently seeded models."""

    models = Model.validate_collection(models)
    child_seeds = _seed_sequence(seed).spawn(len(models))
    simulators = [
        model.to_bayesflow_simulator(
            summaries,
            seed=model_seed,
            include_parameters=False,
            progress=progress,
            **context,
        )
        for model, model_seed in zip(models, child_seeds)
    ]
    return bf.simulators.ModelComparisonSimulator(
        simulators=simulators,
        use_mixed_batches=True,
        key_conflicts="error",
    )


def make_model_comparison_approximator(
    models: Sequence[Model],
    summaries: Summaries,
) -> bf.approximators.ModelComparisonApproximator:
    """Build the classifier used for model comparison."""

    models = Model.validate_collection(models)
    return bf.approximators.ModelComparisonApproximator(
        num_models=len(models),
        classifier_network=bf.networks.MLP(
            widths=(64, 64),
            activation="silu",
            dropout=None,
        ),
        adapter=models[0].make_bayesflow_model_comparison_adapter(summaries),
        standardize="inference_conditions",
    )


def make_model_comparison(
    models: Sequence[Model],
    summaries: Summaries,
    *,
    seed: Seed,
    context: Mapping[str, int],
    progress: Callable[[int], object] | None = None,
) -> tuple[
    bf.approximators.ModelComparisonApproximator,
    bf.simulators.ModelComparisonSimulator,
]:
    """Build an equal-prior BayesFlow model-comparison setup."""

    approximator = make_model_comparison_approximator(models, summaries)
    simulator = make_model_comparison_simulator(
        models,
        summaries,
        seed=seed,
        context=context,
        progress=progress,
    )
    return approximator, simulator


def sample_model_comparison(
    simulator: bf.simulators.ModelComparisonSimulator,
    num_simulations: int,
    *,
    seed: int,
) -> Mapping[str, np.ndarray]:
    """Sample with deterministic model selection without leaking global state."""

    random_state = np.random.get_state()
    np.random.seed(seed)
    try:
        return simulator.sample((num_simulations,))
    finally:
        np.random.set_state(random_state)


def make_offline_optimizer(
    *,
    epochs: int,
    num_batches: int,
) -> keras.optimizers.Optimizer:
    """Match the optimizer policy used by BayesFlow offline workflows."""

    total_steps = epochs * num_batches
    warmup_steps = int(0.05 * total_steps)
    learning_rate = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.1 * INITIAL_LEARNING_RATE,
        warmup_target=INITIAL_LEARNING_RATE,
        warmup_steps=warmup_steps,
        decay_steps=total_steps - warmup_steps,
        alpha=0.0,
    )
    return keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=WEIGHT_DECAY,
        clipnorm=CLIP_NORM,
    )


def fit_comparison_offline(
    approximator: bf.approximators.ModelComparisonApproximator,
    training_data: Mapping[str, np.ndarray],
    *,
    epochs: int,
    batch_size: int,
) -> Any:
    """Train the classifier on fixed, pre-simulated data."""

    dataset = bf.datasets.OfflineDataset(
        data=training_data,
        batch_size=batch_size,
        adapter=approximator.adapter,
    )
    approximator.compile(
        optimizer=make_offline_optimizer(
            epochs=epochs,
            num_batches=dataset.num_batches,
        )
    )
    return approximator.fit(dataset=dataset, epochs=epochs)


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


def run_model_comparison(
    model_names: Sequence[str],
    *,
    output_path: Path | None = None,
    epochs: int = 5,
    num_simulations: int = 2_000,
    batch_size: int = 8,
    diagnostic_datasets: int = DEFAULT_DIAGNOSTIC_DATASETS,
    diagnostics_path: Path | None = None,
    observation_batch_size: int = 1_024,
    seed: int = 42,
    cpus: int = 1,
    summary_names: Sequence[str] | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Train a classifier and compare models on an observed dataset."""

    if num_simulations < 1:
        raise ValueError("num_simulations must be positive")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if diagnostic_datasets < 0:
        raise ValueError("diagnostic_datasets must be non-negative")
    validate_cpus(cpus)

    models = resolve_models(model_names)
    dataset = models[0].dataset
    names = [model.name for model in models]
    selected_names = (
        load_summary_names(dataset)
        if summary_names is None
        else tuple(summary_names)
    )
    observations = load_observations(
        dataset,
        summary_names=selected_names,
    )
    context = observations.context
    summaries = observations.summaries
    keras.utils.set_random_seed(seed)
    (
        training_simulator_seed,
        training_selection_seed,
        diagnostic_simulator_seed,
        diagnostic_selection_seed,
    ) = np.random.SeedSequence(seed).spawn(4)
    with tqdm(
        total=num_simulations,
        desc="Training simulations",
        unit="run",
    ) as progress:
        if cpus == 1:
            approximator, training_simulator = make_model_comparison(
                models,
                summaries,
                seed=training_simulator_seed,
                context=context,
                progress=progress.update,
            )
            training_data = sample_model_comparison(
                training_simulator,
                num_simulations=num_simulations,
                seed=_seed_integer(training_selection_seed),
            )
        else:
            approximator = make_model_comparison_approximator(
                models,
                summaries,
            )
            training_data = sample_model_comparison_in_processes(
                names,
                runs=num_simulations,
                simulator_seed=training_simulator_seed,
                selection_seed=training_selection_seed,
                cpus=cpus,
                summary_names=tuple(summaries),
                progress=progress.update,
            )
    fit_comparison_offline(
        approximator,
        training_data,
        epochs=epochs,
        batch_size=batch_size,
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
        diagnostic_simulator = make_model_comparison_simulator(
            models,
            summaries,
            seed=diagnostic_simulator_seed,
            context=context,
        )
        test_data = sample_model_comparison(
            diagnostic_simulator,
            diagnostic_datasets,
            seed=_seed_integer(diagnostic_selection_seed),
        )
        predicted_models = approximator.predict(
            conditions=test_data,
            probs=True,
        )
        true_models = test_data["model_indices"]
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
        diagnostics_path = diagnostics_path or output_path.parent / "diagnostics"
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
        help=(
            "output PNG "
            "(default: output/comparisons/<models>/probabilities.png)"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="passes over the offline training simulations",
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=2_000,
        help="offline training simulations drawn once before fitting",
    )
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
        "--cpus",
        type=int,
        default=1,
        help="worker processes for offline simulations (default: 1)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="also open the plots in windows",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or (
        ROOT
        / "output"
        / "comparisons"
        / "_vs_".join(args.models)
        / "probabilities.png"
    )
    try:
        plots, probabilities = run_model_comparison(
            args.models,
            output_path=output,
            epochs=args.epochs,
            num_simulations=args.num_simulations,
            batch_size=args.batch_size,
            diagnostic_datasets=args.diagnostic_datasets,
            diagnostics_path=args.diagnostics_dir,
            observation_batch_size=args.observation_batch_size,
            seed=args.seed,
            cpus=args.cpus,
        )
    except SummaryConfigurationError as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(f"Saved model comparison to {output.resolve()}")
    for name, probability in probabilities.items():
        print(f"{name}: {probability:.3f}")
    if args.diagnostic_datasets > 0:
        diagnostics_path = args.diagnostics_dir or output.parent / "diagnostics"
        print(f"Saved diagnostics to {diagnostics_path.resolve()}")
    if args.show:
        plt.show()
    else:
        for figure in plots.values():
            plt.close(figure)


if __name__ == "__main__":
    main()
