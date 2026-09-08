from __future__ import annotations

import math
import sys
from types import ModuleType
import unittest

import numpy as np


sys.modules.setdefault("bayesflow", ModuleType("bayesflow"))
sys.modules.setdefault("pymc", ModuleType("pymc"))

from base.reporting import (
    PriorSummary,
    SummaryStatisticRow,
    describe_summary_statistic,
    render_report_markdown,
    summarize_observation_statistics,
    summarize_priors,
)


class FakeInput:
    def __init__(self, value: float) -> None:
        self.value = value

    def eval(self) -> np.ndarray:
        return np.asarray(self.value)


class FakeVariable:
    def __init__(self, name: str, distribution: str, *parameters: float) -> None:
        self.name = name
        operation = type(
            "FakeOperation",
            (),
            {
                "name": distribution,
                "ndims_params": (0,) * len(parameters),
            },
        )()
        self.owner = type(
            "FakeOwner",
            (),
            {
                "op": operation,
                "inputs": [object(), object()]
                + [FakeInput(value) for value in parameters],
            },
        )()


class FakePrior:
    def __init__(self, variables: list[FakeVariable]) -> None:
        self.free_RVs = variables

    def eval_rv_shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            variable.name: ((4,) if variable.name == "drift" else ())
            for variable in self.free_RVs
        }


class ExampleModel:
    """A concise generative model for reporting tests."""

    name = "example_model"
    inference_variables = (
        "duration",
        "lengthscale",
        "eta",
        "drift",
        "heterogeneity",
        "probability",
        "chance",
    )
    parameter_units = {
        "duration": "minutes",
        "lengthscale": "minutes",
        "drift": "log odds per year",
    }

    def build_prior(self, **context: object) -> FakePrior:
        del context
        return FakePrior(
            [
                FakeVariable("duration", "lognormal", math.log(3.0), 0.5),
                FakeVariable("lengthscale", "exponential", 60.0),
                FakeVariable("eta", "pareto", 1.5, 1.0),
                FakeVariable("drift", "normal", 0.0, 0.05),
                FakeVariable("heterogeneity", "halfnormal", 0.0, 2.0),
                FakeVariable("probability", "beta", 2.0, 3.0),
                FakeVariable("chance", "uniform", 0.0, 1.0),
                FakeVariable("latent", "normal", 0.0, 1.0),
            ]
        )


class PriorReportingTests(unittest.TestCase):
    def test_summarizes_natural_scale_moments_units_and_shapes(self) -> None:
        priors = summarize_priors(ExampleModel(), {})
        by_name = {prior.name: prior for prior in priors}

        self.assertEqual(tuple(by_name), ExampleModel.inference_variables)
        self.assertEqual(
            by_name["duration"].prior,
            "LogNormal(mu=1.1, sigma=0.5)",
        )
        self.assertAlmostEqual(by_name["duration"].mean, 3.3994453592)
        self.assertAlmostEqual(by_name["duration"].sigma, 1.8117015996)
        self.assertEqual(by_name["duration"].unit, "minutes")
        self.assertEqual(
            by_name["lengthscale"].prior,
            "Exponential(lam=0.0167)",
        )
        self.assertEqual(by_name["lengthscale"].mean, 60.0)
        self.assertEqual(by_name["lengthscale"].sigma, 60.0)
        self.assertEqual(by_name["eta"].mean, 3.0)
        self.assertTrue(math.isinf(by_name["eta"].sigma))
        self.assertEqual(by_name["eta"].unit, "—")
        self.assertEqual(by_name["drift"].shape, (4,))
        self.assertEqual(
            by_name["heterogeneity"].prior,
            "HalfNormal(sigma=2)",
        )
        self.assertAlmostEqual(
            by_name["heterogeneity"].mean,
            2.0 * math.sqrt(2.0 / math.pi),
        )
        self.assertAlmostEqual(by_name["probability"].mean, 0.4)
        self.assertAlmostEqual(by_name["probability"].sigma, 0.2)
        self.assertAlmostEqual(by_name["chance"].mean, 0.5)
        self.assertAlmostEqual(
            by_name["chance"].sigma,
            1.0 / math.sqrt(12.0),
        )
        self.assertNotIn("latent", by_name)

    def test_renders_complete_report_with_dynamic_diagnostics(self) -> None:
        markdown = render_report_markdown(
            model=ExampleModel(),
            summary_names=("connectivity", "clustering"),
            summary_rows=(
                SummaryStatisticRow(
                    name="connectivity",
                    description="Density of the contact network",
                    value=0.25,
                ),
                SummaryStatisticRow(
                    name="clustering",
                    description="Mean local clustering",
                    value=0.125,
                ),
            ),
            priors=(
                PriorSummary(
                    name="duration",
                    prior="LogNormal(mu=1.1, sigma=0.5)",
                    mean=3.4,
                    sigma=1.81,
                    unit="minutes",
                    shape=(),
                ),
            ),
            plot_names=(
                "simulations",
                "posterior",
                "posterior_predictive",
                "calibration_ecdf",
            ),
        )

        for heading in (
            "## Model description",
            "## Summary statistics",
            "## Parameters and prior distributions",
            "## Prior-predictive summary statistics",
            "## Prior and posterior parameter distributions",
            "## Posterior-predictive summary statistics",
            "## Inference and identification diagnostics",
        ):
            self.assertIn(heading, markdown)
        self.assertIn("| Statistic | Description | Value |", markdown)
        self.assertIn("| `connectivity` | Density of the contact network | 0.25 |", markdown)
        self.assertIn("| `clustering` | Mean local clustering | 0.125 |", markdown)
        self.assertIn("| Parameter | Prior | Mean | Sigma | Unit |", markdown)
        self.assertIn("| `duration` |", markdown)
        self.assertIn("| 3.4 | 1.81 | minutes |", markdown)
        self.assertIn("![Prior-predictive", markdown)
        self.assertIn("(simulations.png)", markdown)
        self.assertIn("(posterior.png)", markdown)
        self.assertIn("(posterior_predictive.png)", markdown)
        self.assertIn(
            "(diagnostics/calibration_ecdf.png)",
            markdown,
        )
        self.assertIn("### Calibration ECDF", markdown)
        self.assertIn("misspecification", markdown)
        self.assertIn("missidentification", markdown)
        self.assertNotIn("maps micro-level behavioral assumptions", markdown)

    def test_builds_rows_from_loaded_observation_conditions(self) -> None:
        rows = summarize_observation_statistics(
            ("cumulative_network_connectivity",),
            {"cumulative_network_connectivity": np.array([[0.375]])},
            dataset="contacts",
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].name, "cumulative_network_connectivity")
        self.assertEqual(
            rows[0].description,
            "The density of the cumulative binary contact network",
        )
        self.assertAlmostEqual(rows[0].value, 0.375)
        self.assertEqual(
            describe_summary_statistic(
                "contacts",
                "cumulative_network_clustering",
            ),
            "Mean local clustering of the cumulative contact network",
        )

    def test_marks_disabled_optional_sections(self) -> None:
        markdown = render_report_markdown(
            model=ExampleModel(),
            summary_names=("connectivity",),
            summary_rows=(
                SummaryStatisticRow(
                    name="connectivity",
                    description="Density of the contact network",
                    value=0.5,
                ),
            ),
            priors=(),
            plot_names=("simulations", "posterior"),
        )

        self.assertIn("predictive stage was disabled", markdown)
        self.assertIn("diagnostic stage was disabled", markdown)


if __name__ == "__main__":
    unittest.main()
