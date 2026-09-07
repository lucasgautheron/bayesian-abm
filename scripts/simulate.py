"""Simulate summary statistics and compare them with a selected dataset.

Example:
    python scripts/simulate.py reputation_conversation
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from base.model import Model
from base.observations import load_observations
from base.summary_config import SummaryConfigurationError, load_summary_names
from models import resolve_model
from scripts.parallel import sample_model_in_processes, validate_cpus
from visualization.diagnostics import plot_summary_pairplot


DEFAULT_RUNS = 1000


def _scalar_runs(
    values: ArrayLike,
    runs: int,
) -> NDArray[np.float64]:
    """Return one scalar summary per simulation run."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0 or array.shape[0] != runs:
        raise ValueError("simulated summaries must have one value per run")
    flattened = array.reshape(runs, -1)
    if flattened.shape[1] != 1:
        raise ValueError("simulated summary statistics must be scalar")
    if not np.all(np.isfinite(flattened)):
        raise ValueError("simulated summaries must be finite")
    return flattened


def summary_frame(
    simulated: Mapping[str, ArrayLike],
    *,
    runs: int,
) -> pd.DataFrame:
    """Represent scalar summaries as pair-plot columns."""

    columns = {
        name: _scalar_runs(values, runs)[:, 0]
        for name, values in simulated.items()
    }
    return pd.DataFrame(columns)


def observed_summary_statistics(
    observed: Mapping[str, ArrayLike],
) -> dict[str, float]:
    """Return validated observed scalar summaries."""

    statistics: dict[str, float] = {}
    for name, value in observed.items():
        flattened = np.asarray(value, dtype=np.float64).reshape(-1)
        if flattened.size != 1:
            raise ValueError("observed summary statistics must be scalar")
        if not np.all(np.isfinite(flattened)):
            raise ValueError("observed summaries must be finite")
        statistics[name] = float(flattened[0])
    return statistics


def run_simulations(
    model_name: str,
    *,
    output_path: Path | None = None,
    runs: int = DEFAULT_RUNS,
    seed: int = 42,
    cpus: int = 1,
    summary_names: Sequence[str] | None = None,
) -> Figure:
    """Run a model repeatedly and return its summary-statistic pair plot."""

    if runs < 2:
        raise ValueError("runs must be at least 2 to estimate densities")
    validate_cpus(cpus)
    model: Model = resolve_model(model_name)
    selected_names = (
        load_summary_names(model.dataset)
        if summary_names is None
        else tuple(summary_names)
    )
    observations = load_observations(
        model.dataset,
        summary_names=selected_names,
    )
    context = observations.context
    summaries = observations.summaries
    from tqdm.auto import tqdm

    with tqdm(total=runs, desc="Simulating", unit="run") as progress:
        if cpus == 1:
            simulator = model.to_bayesflow_simulator(
                summaries,
                seed=seed,
                include_parameters=False,
                progress=progress.update,
                **context,
            )
            simulated = simulator.sample((runs,))
        else:
            simulated = sample_model_in_processes(
                model_name,
                runs=runs,
                seed=seed,
                cpus=cpus,
                include_parameters=False,
                summary_names=tuple(summaries),
                progress=progress.update,
            )
    frame = summary_frame(
        simulated,
        runs=runs,
    )
    observed_frame = summary_frame(
        observations.conditions,
        runs=observations.count,
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
        help="output PNG (default: output/<model>/simulations.png)",
    )
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cpus",
        type=int,
        default=1,
        help="worker processes for simulations (default: 1)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="also open the pair plot in a window",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or ROOT / "output" / args.model / "simulations.png"
    try:
        run_simulations(
            args.model,
            output_path=output,
            runs=args.runs,
            seed=args.seed,
            cpus=args.cpus,
        )
    except SummaryConfigurationError as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(f"Saved simulation pair plot to {output.resolve()}")
    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
