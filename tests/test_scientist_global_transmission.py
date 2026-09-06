from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from models import MODEL_REGISTRY, SCIENTIST_MODEL_REGISTRY
from models.scientist_conventions import GlobalTransmissionModel
from models.scientist_conventions.cultural import cultural_probabilities
from tests.scientist_model_helpers import FakePyMC, scientist_context


class GlobalTransmissionModelTests(unittest.TestCase):
    def test_model_is_registered(self) -> None:
        self.assertIs(
            SCIENTIST_MODEL_REGISTRY["scientist_global_transmission"],
            GlobalTransmissionModel,
        )
        self.assertIs(
            MODEL_REGISTRY["scientist_global_transmission"],
            GlobalTransmissionModel,
        )

    def test_prior_separates_hyperparameters_from_nuisance_innovations(self) -> None:
        model = GlobalTransmissionModel()
        with patch("models.scientist_conventions.cultural.pm", FakePyMC):
            prior = model.build_prior(**scientist_context())
        self.assertEqual(
            tuple(name for _, name, _ in prior.variables),
            (
                "baseline_log_odds",
                "annual_drift",
                "innovation_scale",
                "annual_innovation",
            ),
        )
        self.assertEqual(
            model.inference_variables,
            ("baseline_log_odds", "annual_drift", "innovation_scale"),
        )
        variables = {name: arguments for _, name, arguments in prior.variables}
        self.assertEqual(
            variables["baseline_log_odds"],
            {"mu": 0.0, "sigma": 1.5, "shape": 4},
        )
        self.assertEqual(
            variables["annual_drift"],
            {"mu": 0.0, "sigma": 0.05, "shape": 4},
        )
        self.assertEqual(variables["innovation_scale"], {"sigma": 0.1})
        self.assertEqual(
            variables["annual_innovation"],
            {"mu": 0.0, "sigma": 1.0, "shape": (3, 4)},
        )

    def test_constructs_category_specific_logit_random_walk(self) -> None:
        probabilities = cultural_probabilities(
            {
                "baseline_log_odds": np.zeros(4),
                "annual_drift": np.asarray([0.1, 0.0, -0.1, 0.0]),
                "innovation_scale": np.asarray(0.0),
                "annual_innovation": np.zeros((2, 4)),
            },
            start_year=2000,
            end_year=2002,
        )

        np.testing.assert_allclose(probabilities[0], 0.5)
        self.assertGreater(probabilities[-1, 0], 0.5)
        self.assertLess(probabilities[-1, 2], 0.5)
        np.testing.assert_allclose(probabilities[:, 1], 0.5)

    def test_identical_seeds_reproduce_parameters_and_preferences(self) -> None:
        model = GlobalTransmissionModel()
        context = scientist_context()
        parameters = {
            "baseline_log_odds": np.zeros(4),
            "annual_drift": np.asarray([0.01, -0.01, 0.0, 0.02]),
            "innovation_scale": np.asarray(0.1),
            "annual_innovation": np.zeros((3, 4)),
        }
        first_data = model.simulate(
            parameters,
            np.random.default_rng(18),
            **context,
        )
        second_data = model.simulate(
            parameters,
            np.random.default_rng(18),
            **context,
        )
        np.testing.assert_array_equal(
            first_data["preference"],
            second_data["preference"],
        )


if __name__ == "__main__":
    unittest.main()
