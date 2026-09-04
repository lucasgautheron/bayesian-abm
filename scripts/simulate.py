"""Simulate summary statistics and compare them with a selected dataset.

Example:
    python scripts/simulate.py reputation_conversation
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from base.model import Model
from base.observations import load_observations
from models import resolve_model


DEFAULT_RUNS = 100


def _reshape_runs(
    values: ArrayLike,
    runs: int,
) -> NDArray[np.float64]:
    """Return one flattened summary vector per simulation run."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0 or array.shape[0] != runs:
        raise ValueError("simulated summaries must have one value per run")
    flattened = array.reshape(runs, -1)
    if flattened.shape[1] == 0:
        raise ValueError("simulated summaries must not be empty")
    if not np.all(np.isfinite(flattened)):
        raise ValueError("simulated summaries must be finite")
    return flattened


def summary_frame(
    simulated: Mapping[str, ArrayLike],
    *,
    runs: int,
    vector_moments: bool = False,
) -> pd.DataFrame:
    """Represent scalars and vector means, with optional standard deviations."""

    columns: dict[str, NDArray[np.float64]] = {}
    for name, values in simulated.items():
        flattened = _reshape_runs(values, runs)
        if flattened.shape[1] == 1:
            columns[name] = flattened[:, 0]
        else:
            columns[f"{name}_mean"] = flattened.mean(axis=1)
        if flattened.shape[1] > 1 and vector_moments:
            columns[f"{name}_stdev"] = flattened.std(axis=1)
    return pd.DataFrame(columns)


def observed_summary_statistics(
    observed: Mapping[str, ArrayLike],
    *,
    vector_moments: bool = False,
) -> dict[str, float]:
    """Match observed scalar/vector summaries to pair-plot columns."""

    statistics: dict[str, float] = {}
    for name, value in observed.items():
        flattened = np.asarray(value, dtype=np.float64).reshape(-1)
        if flattened.size == 0:
            raise ValueError("observed summaries must not be empty")
        if not np.all(np.isfinite(flattened)):
            raise ValueError("observed summaries must be finite")
        if flattened.size == 1:
            statistics[name] = float(flattened[0])
        else:
            statistics[f"{name}_mean"] = float(flattened.mean())
        if flattened.size > 1 and vector_moments:
            statistics[f"{name}_stdev"] = float(flattened.std())
    return statistics


def _plot_limits(
    values: ArrayLike,
    observed: ArrayLike,
) -> tuple[float, float]:
    combined = np.concatenate(
        (
            np.asarray(values, dtype=np.float64).reshape(-1),
            np.asarray(observed, dtype=np.float64).reshape(-1),
        )
    )
    lower = float(combined.min())
    upper = float(combined.max())
    padding = (upper - lower) * 0.05
    if padding == 0:
        padding = max(abs(lower) * 0.05, 0.5)
    return lower - padding, upper + padding


def plot_summary_pairplot(
    frame: pd.DataFrame,
    observed: Mapping[str, float] | pd.DataFrame,
) -> Figure:
    """Plot simulated summaries against one or more observations."""

    if frame.empty or len(frame.columns) == 0:
        raise ValueError("at least one simulated summary is required")
    names = list(frame.columns)
    observed_frame = (
        pd.DataFrame([observed])
        if isinstance(observed, Mapping)
        else observed
    )
    if observed_frame.empty:
        raise ValueError("at least one observed summary is required")
    if set(observed_frame.columns) != set(names):
        raise ValueError("observed and simulated summaries must match")
    observed_frame = observed_frame[names]
    if not np.all(np.isfinite(observed_frame.to_numpy())):
        raise ValueError("observed summaries must be finite")

    count = len(names)
    figure, axes = plt.subplots(
        count,
        count,
        figsize=(3.2 * count, 3.2 * count),
        squeeze=False,
    )
    limits = {
        name: _plot_limits(frame[name], observed_frame[name])
        for name in names
    }
    single_observation = len(observed_frame) == 1

    for row, y_name in enumerate(names):
        for column, x_name in enumerate(names):
            axis = axes[row, column]
            x_values = frame[x_name].to_numpy()

            if row == column:
                sns.kdeplot(
                    x=x_values,
                    ax=axis,
                    fill=True,
                    color="tab:blue",
                    alpha=0.35,
                    linewidth=1.5,
                    warn_singular=False,
                )
                if single_observation:
                    observed_value = observed_frame[x_name].iloc[0]
                    axis.axvline(
                        observed_value,
                        color="tab:red",
                        linestyle=":",
                        linewidth=1,
                    )
                    axis.scatter(
                        observed_value,
                        0,
                        marker="*",
                        s=140,
                        color="tab:red",
                        edgecolor="white",
                        linewidth=0.7,
                        zorder=3,
                        clip_on=False,
                    )
                else:
                    sns.kdeplot(
                        x=observed_frame[x_name],
                        ax=axis,
                        color="tab:red",
                        linewidth=1.3,
                        warn_singular=False,
                    )
            else:
                y_values = frame[y_name].to_numpy()
                sns.kdeplot(
                    x=x_values,
                    y=y_values,
                    ax=axis,
                    levels=7,
                    thresh=0.05,
                    cmap="Blues",
                    linewidths=1.2,
                    warn_singular=False,
                )
                if single_observation:
                    axis.scatter(
                        observed_frame[x_name],
                        observed_frame[y_name],
                        marker="*",
                        s=170,
                        color="tab:red",
                        edgecolor="white",
                        linewidth=0.7,
                        zorder=3,
                    )
                else:
                    axis.scatter(
                        observed_frame[x_name],
                        observed_frame[y_name],
                        s=8,
                        color="tab:red",
                        alpha=0.12,
                        linewidth=0,
                        rasterized=True,
                    )
                axis.set_ylim(limits[y_name])

            axis.set_xlim(limits[x_name])
            if row == count - 1:
                axis.set_xlabel(x_name.replace("_", " "))
            else:
                axis.tick_params(labelbottom=False)
            if column == 0 and row != column:
                axis.set_ylabel(y_name.replace("_", " "))
            elif column != 0:
                axis.tick_params(labelleft=False)

    figure.suptitle(
        "Prior-predictive summary statistics (observed data: red)"
    )
    figure.tight_layout()
    return figure


def run_simulations(
    model_name: str,
    *,
    output_path: Path | None = None,
    runs: int = DEFAULT_RUNS,
    seed: int = 42,
    vector_moments: bool = False,
) -> Figure:
    """Run a model repeatedly and return its summary-statistic pair plot."""

    if runs < 2:
        raise ValueError("runs must be at least 2 to estimate densities")
    model: Model = resolve_model(model_name)
    observations = load_observations(model.dataset)
    context = observations.context
    summaries = observations.summaries
    from tqdm.auto import tqdm

    with tqdm(total=runs, desc="Simulating", unit="run") as progress:
        simulator = model.to_bayesflow_simulator(
            summaries,
            seed=seed,
            include_parameters=False,
            progress=progress.update,
            **context,
        )
        simulated = simulator.sample((runs,))
    frame = summary_frame(
        simulated,
        runs=runs,
        vector_moments=vector_moments,
    )
    observed_frame = summary_frame(
        observations.conditions,
        runs=observations.count,
        vector_moments=vector_moments,
    )
    observed: Mapping[str, float] | pd.DataFrame
    if observations.count == 1:
        observed = observed_frame.iloc[0].to_dict()
    else:
        observed = observed_frame
    figure = plot_summary_pairplot(frame, observed)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
    return figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run prior-predictive simulations and save a pair plot "
            "of their summary statistics."
        )
    )
    parser.add_argument(
        "model",
        help="registered model name (selects observed data automatically)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output PNG (default: simulations_<model>.png)",
    )
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--vector-moments",
        action="store_true",
        help="include vector summary means and standard deviations",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="also open the pair plot in a window",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or Path(f"simulations_{args.model}.png")
    run_simulations(
        args.model,
        output_path=output,
        runs=args.runs,
        seed=args.seed,
        vector_moments=args.vector_moments,
    )
    print(f"Saved simulation pair plot to {output.resolve()}")
    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
