from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


sys.modules.setdefault("bayesflow", ModuleType("bayesflow"))
sys.modules.setdefault("pymc", ModuleType("pymc"))

import pandas as pd

from scripts.inference import (
    make_workflow,
    posterior_parameter_draws,
    run_inference,
    sample_observations,
)
from visualization.diagnostics import (
    plot_predictive_summary_pairplot,
    plot_prior_posterior_pairplot,
    prepare_posterior_plot_data,
)


class FakeWorkflowModel:
    inference_variables = ("theta",)

    def to_bayesflow_simulator(self, summaries, **kwargs):
        self.simulator_arguments = (summaries, kwargs)
        return "simulator"

    def make_bayesflow_adapter(self, summaries, **context):
        self.adapter_arguments = (summaries, context)
        return "adapter"

class PosteriorPlotDataTests(unittest.TestCase):
    def test_restores_scalar_axis_and_averages_vector_parameters(self) -> None:
        prepared, names = prepare_posterior_plot_data(
            {
                "scalar": np.zeros((1, 10)),
                "vector": np.broadcast_to(
                    np.asarray([1.0, 3.0]),
                    (1, 10, 2),
                ),
            },
            ["scalar", "vector"],
        )

        self.assertEqual(prepared["scalar"].shape, (1, 10, 1))
        self.assertEqual(prepared["vector"].shape, (1, 10, 1))
        np.testing.assert_array_equal(prepared["vector"], 2.0)
        self.assertEqual(names, ["scalar", "vector_mean"])

    def test_rejects_missing_variable(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing"):
            prepare_posterior_plot_data({}, ["rate"])

    def test_pairplot_overlays_prior_and_posterior_densities(self) -> None:
        prior = {
            "rate": np.linspace(0.0, 1.0, 20),
            "scale": np.linspace(1.0, 2.0, 20),
        }
        posterior = {
            "rate": np.linspace(0.4, 0.8, 20)[None, :],
            "scale": np.linspace(1.2, 1.6, 20)[None, :],
        }

        with patch("visualization.diagnostics.sns.kdeplot") as kdeplot:
            figure = plot_prior_posterior_pairplot(
                prior,
                posterior,
                ["rate", "scale"],
            )

        self.assertEqual(len(figure.axes), 4)
        self.assertEqual(kdeplot.call_count, 8)
        self.assertEqual(
            [text.get_text() for text in figure.legends[0].get_texts()],
            ["Prior", "Posterior"],
        )

    def test_offline_training_requires_positive_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_simulations"):
            run_inference("latent_network", num_simulations=0)
        with self.assertRaisesRegex(ValueError, "epochs"):
            run_inference("latent_network", epochs=0)
        with self.assertRaisesRegex(ValueError, "batch_size"):
            run_inference("latent_network", batch_size=0)

    def test_workflow_uses_direct_scalar_conditions(self) -> None:
        model = FakeWorkflowModel()
        fake_bf = SimpleNamespace(
            BasicWorkflow=lambda **kwargs: kwargs,
            networks=SimpleNamespace(FlowMatching=lambda: "flow"),
        )

        with patch("scripts.inference.bf", fake_bf):
            workflow = make_workflow(
                model,
                {"contacts": object()},
                seed=4,
                context={"n_agents": 3, "n_steps": 2},
            )

        self.assertEqual(workflow["inference_conditions"], ["contacts"])
        self.assertNotIn("summary_network", workflow)
        self.assertEqual(workflow["adapter"], "adapter")

    def test_samples_all_observations_in_batches(self) -> None:
        class Workflow:
            def __init__(self) -> None:
                self.batch_sizes: list[int] = []

            def sample(self, *, conditions, num_samples):
                values = conditions["mentions"]
                self.batch_sizes.append(len(values))
                return {
                    "rate": np.repeat(
                        values[:, :1],
                        num_samples,
                        axis=1,
                    )
                }

        workflow = Workflow()
        posterior = sample_observations(
            workflow,
            {"mentions": np.arange(10).reshape(5, 2)},
            num_samples=3,
            batch_size=2,
        )

        self.assertEqual(workflow.batch_sizes, [2, 2, 1])
        self.assertEqual(posterior["rate"].shape, (5, 3))

    def test_posterior_parameter_draws_select_one_dataset(self) -> None:
        posterior = {
            "rate": np.arange(12.0).reshape(2, 6),
            "scale": np.arange(12.0, 24.0).reshape(2, 6, 1),
        }

        selected = posterior_parameter_draws(
            posterior,
            ["rate", "scale"],
            dataset_id=1,
            draws=3,
            rng=np.random.default_rng(0),
        )

        self.assertEqual(selected["rate"].shape, (3,))
        self.assertEqual(selected["scale"].shape, (3, 1))
        self.assertTrue(np.all(selected["rate"] >= 6.0))

    def test_predictive_pairplot_overlays_prior_posterior_and_data(
        self,
    ) -> None:
        prior = pd.DataFrame(
            {
                "contacts": np.linspace(1.0, 3.0, 8),
                "clustering": np.linspace(0.1, 0.4, 8),
            }
        )
        posterior = pd.DataFrame(
            {
                "contacts": np.linspace(1.5, 2.5, 8),
                "clustering": np.linspace(0.2, 0.3, 8),
            }
        )

        with patch("visualization.diagnostics.sns.kdeplot") as kdeplot:
            figure = plot_predictive_summary_pairplot(
                prior,
                posterior,
                {"contacts": 2.0, "clustering": 0.25},
            )

        self.assertEqual(len(figure.axes), 4)
        self.assertEqual(kdeplot.call_count, 8)
        self.assertEqual(
            [text.get_text() for text in figure.legends[0].get_texts()],
            ["Prior predictive", "Posterior predictive", "Observed"],
        )


if __name__ == "__main__":
    unittest.main()
