"""Generate a Markdown report and figures for one registered model.

Example:
    python scripts/report.py latent_network
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from base.observations import load_observations
from base.reporting import summarize_priors, write_report_markdown
from base.summary_config import (
    SummaryConfigurationError,
    format_summary_configuration_error,
    load_summary_names,
)
from models import resolve_model
from scripts.inference import run_inference


def run_report(
    model_name: str,
    *,
    report_dir: Path | None = None,
    epochs: int = 32,
    num_simulations: int = 2_000,
    batch_size: int = 8,
    posterior_draws: int = 2_000,
    predictive_runs: int = 100,
    diagnostic_datasets: int = 50,
    diagnostic_draws: int = 200,
    observation_batch_size: int = 256,
    seed: int = 42,
    cpus: int = 1,
) -> dict[str, Any]:
    """Run inference once and return every generated report figure."""

    model = resolve_model(model_name)
    summary_names = load_summary_names(model.dataset)
    observations = load_observations(
        model.dataset,
        summary_names=summary_names,
    )
    priors = summarize_priors(model, observations.context)
    destination = report_dir or ROOT / "reports" / model_name
    destination.mkdir(parents=True, exist_ok=True)

    inference_plots = run_inference(
        model_name,
        output_path=destination / "posterior.png",
        summary_output_path=destination / "simulations.png",
        epochs=epochs,
        num_simulations=num_simulations,
        batch_size=batch_size,
        posterior_draws=posterior_draws,
        predictive_runs=predictive_runs,
        diagnostic_datasets=diagnostic_datasets,
        diagnostic_draws=diagnostic_draws,
        diagnostics_path=destination / "diagnostics",
        observation_batch_size=observation_batch_size,
        seed=seed,
        cpus=cpus,
        summary_names=summary_names,
    )
    write_report_markdown(
        destination / "report.md",
        model=model,
        summary_names=summary_names,
        priors=priors,
        plot_names=tuple(inference_plots),
    )
    return inference_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one simulation-based inference workflow and save its "
            "Markdown report and figures in one directory."
        )
    )
    parser.add_argument(
        "model",
        help="registered model name (selects observed data automatically)",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        help="report directory (default: reports/<model>)",
    )
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--num-simulations", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--posterior-draws", type=int, default=2_000)
    parser.add_argument("--predictive-runs", type=int, default=100)
    parser.add_argument("--diagnostic-datasets", type=int, default=50)
    parser.add_argument("--diagnostic-draws", type=int, default=200)
    parser.add_argument("--observation-batch-size", type=int, default=256)
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
        help="also open all report figures in windows",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = args.report_dir or ROOT / "reports" / args.model
    try:
        plots = run_report(
            args.model,
            report_dir=destination,
            epochs=args.epochs,
            num_simulations=args.num_simulations,
            batch_size=args.batch_size,
            posterior_draws=args.posterior_draws,
            predictive_runs=args.predictive_runs,
            diagnostic_datasets=args.diagnostic_datasets,
            diagnostic_draws=args.diagnostic_draws,
            observation_batch_size=args.observation_batch_size,
            seed=args.seed,
            cpus=args.cpus,
        )
    except SummaryConfigurationError as exc:
        message = format_summary_configuration_error(
            exc,
            color=sys.stderr.isatty(),
        )
        raise SystemExit(f"error: {message}") from exc

    print(f"Saved report to {(destination / 'report.md').resolve()}")
    if args.show:
        plt.show()
    else:
        for figure in plots.values():
            plt.close(figure)


if __name__ == "__main__":
    main()
