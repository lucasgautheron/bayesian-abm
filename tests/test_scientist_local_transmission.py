from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from models import MODEL_REGISTRY, SCIENTIST_MODEL_REGISTRY
from models.scientist_conventions import LocalTransmissionModel
from models.scientist_conventions.local_transmission import (
    simulate_local_transmission,
)
from tests.scientist_model_helpers import FakePyMC, scientist_context


class LocalTransmissionModelTests(unittest.TestCase):
    def test_model_is_registered(self) -> None:
        self.assertIs(
            SCIENTIST_MODEL_REGISTRY["scientist_local_transmission"],
            LocalTransmissionModel,
        )
        self.assertIs(
            MODEL_REGISTRY["scientist_local_transmission"],
            LocalTransmissionModel,
        )

    def test_uniform_imitation_prior_is_an_inference_target(self) -> None:
        model = LocalTransmissionModel()
        with patch("models.scientist_conventions.cultural.pm", FakePyMC):
            prior = model.build_prior(**scientist_context())
        imitation = [
            arguments
            for _, name, arguments in prior.variables
            if name == "imitation_probability"
        ]
        self.assertEqual(imitation, [{"alpha": 1.0, "beta": 1.0}])
        self.assertIn("imitation_probability", model.inference_variables)
        self.assertNotIn("annual_innovation", model.inference_variables)

    def test_imitation_extremes_switch_between_global_and_local_rules(self) -> None:
        probabilities = np.zeros((4, 4), dtype=np.float64)
        probabilities[0, 0] = 1.0
        context = scientist_context()
        global_only = simulate_local_transmission(
            probabilities,
            np.random.default_rng(3),
            primary_area=context["primary_area"],
            career_start_year=context["career_start_year"],
            first_coauthor=context["first_coauthor"],
            start_year=context["start_year"],
            imitation_probability=0.0,
        )
        all_imitate = simulate_local_transmission(
            probabilities,
            np.random.default_rng(3),
            primary_area=context["primary_area"],
            career_start_year=context["career_start_year"],
            first_coauthor=context["first_coauthor"],
            start_year=context["start_year"],
            imitation_probability=1.0,
        )

        np.testing.assert_array_equal(global_only, [1, -1, -1, -1])
        np.testing.assert_array_equal(all_imitate, [1, 1, 1, 1])

    def test_identical_seeds_reproduce_parameters_and_preferences(self) -> None:
        model = LocalTransmissionModel()
        context = scientist_context()
        parameters = {
            "baseline_log_odds": np.zeros(4),
            "annual_drift": np.zeros(4),
            "innovation_scale": np.asarray(0.1),
            "annual_innovation": np.zeros((3, 4)),
            "imitation_probability": np.asarray(0.4),
        }
        first_data = model.simulate(
            parameters,
            np.random.default_rng(19),
            **context,
        )
        second_data = model.simulate(
            parameters,
            np.random.default_rng(19),
            **context,
        )
        np.testing.assert_array_equal(
            first_data["preference"],
            second_data["preference"],
        )


if __name__ == "__main__":
    unittest.main()
