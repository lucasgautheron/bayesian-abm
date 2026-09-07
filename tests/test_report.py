from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


sys.modules.setdefault("bayesflow", ModuleType("bayesflow"))
sys.modules.setdefault("pymc", ModuleType("pymc"))

from base.summary_config import SummaryConfigurationError
from scripts.report import parse_args, run_report


class ReportTests(unittest.TestCase):
    def test_reuses_inference_training_simulations_for_report(self) -> None:
        summary_names = (
            "cumulative_network_connectivity",
            "cumulative_network_clustering",
        )
        simulation_figure = object()
        posterior_figure = object()
        diagnostic_figure = object()
        prior_rows = (object(),)

        with TemporaryDirectory() as directory:
            destination = Path(directory) / "custom-report"
            with (
                patch(
                    "scripts.report.resolve_model",
                    return_value=SimpleNamespace(
                        dataset="contacts",
                        name="latent_network",
                    ),
                ),
                patch(
                    "scripts.report.load_summary_names",
                    return_value=summary_names,
                ) as load_names,
                patch(
                    "scripts.report.load_observations",
                    return_value=SimpleNamespace(context={"n_agents": 4}),
                ) as load_observations,
                patch(
                    "scripts.report.summarize_priors",
                    return_value=prior_rows,
                ) as summarize,
                patch(
                    "scripts.report.run_inference",
                    return_value={
                        "simulations": simulation_figure,
                        "posterior": posterior_figure,
                        "calibration": diagnostic_figure,
                    },
                ) as inference,
                patch(
                    "scripts.report.write_report_markdown"
                ) as write_markdown,
            ):
                figures = run_report(
                    "latent_network",
                    report_dir=destination,
                    epochs=3,
                    num_simulations=40,
                    batch_size=5,
                    posterior_draws=30,
                    predictive_runs=8,
                    diagnostic_datasets=7,
                    diagnostic_draws=6,
                    observation_batch_size=4,
                    seed=9,
                    cpus=2,
                )

            self.assertTrue(destination.is_dir())
            load_names.assert_called_once_with("contacts")
            load_observations.assert_called_once_with(
                "contacts",
                summary_names=summary_names,
            )
            summarize.assert_called_once()
            inference.assert_called_once_with(
                "latent_network",
                output_path=destination / "posterior.png",
                summary_output_path=destination / "simulations.png",
                epochs=3,
                num_simulations=40,
                batch_size=5,
                posterior_draws=30,
                predictive_runs=8,
                diagnostic_datasets=7,
                diagnostic_draws=6,
                diagnostics_path=destination / "diagnostics",
                observation_batch_size=4,
                seed=9,
                cpus=2,
                summary_names=summary_names,
            )
            write_markdown.assert_called_once()
            self.assertEqual(
                write_markdown.call_args.args,
                (destination / "report.md",),
            )

        self.assertEqual(
            figures,
            {
                "simulations": simulation_figure,
                "posterior": posterior_figure,
                "calibration": diagnostic_figure,
            },
        )

    def test_requires_configuration_before_creating_report_directory(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "report"
            with (
                patch(
                    "scripts.report.resolve_model",
                    return_value=SimpleNamespace(dataset="contacts"),
                ),
                patch(
                    "scripts.report.load_summary_names",
                    side_effect=SummaryConfigurationError(
                        "selection required"
                    ),
                ),
            ):
                with self.assertRaisesRegex(
                    SummaryConfigurationError,
                    "selection required",
                ):
                    run_report("latent_network", report_dir=destination)

            self.assertFalse(destination.exists())

    def test_cli_defaults_to_the_standard_report_location(self) -> None:
        with patch.object(sys, "argv", ["report.py", "latent_network"]):
            args = parse_args()

        self.assertIsNone(args.report_dir)
        self.assertFalse(hasattr(args, "simulation_runs"))
        self.assertEqual(args.cpus, 1)


if __name__ == "__main__":
    unittest.main()
