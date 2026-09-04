from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


sys.modules.setdefault("bayesflow", ModuleType("bayesflow"))
sys.modules.setdefault("pymc", ModuleType("pymc"))

from scripts.inference import (
    make_workflow,
    prepare_posterior_plot_data,
    sample_observations,
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


if __name__ == "__main__":
    unittest.main()
