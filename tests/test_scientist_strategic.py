from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix

from models import MODEL_REGISTRY, SCIENTIST_MODEL_REGISTRY
from models.scientist_conventions import StrategicConventionModel
from models.scientist_conventions.strategic import strategic_best_response
from tests.scientist_model_helpers import FakePyMC, scientist_context


class StrategicConventionModelTests(unittest.TestCase):
    def test_model_is_registered(self) -> None:
        self.assertIs(
            SCIENTIST_MODEL_REGISTRY["scientist_strategic"],
            StrategicConventionModel,
        )
        self.assertIs(
            MODEL_REGISTRY["scientist_strategic"],
            StrategicConventionModel,
        )

    def test_prior_and_inference_variables(self) -> None:
        model = StrategicConventionModel()
        with patch("models.scientist_conventions.strategic.pm", FakePyMC):
            prior = model.build_prior(**scientist_context())
        self.assertEqual(
            [(distribution, name) for distribution, name, _ in prior.variables],
            [
                ("Normal", "contextual_advantage"),
                ("Exponential", "coordination_scale"),
            ],
        )
        self.assertEqual(
            model.inference_variables,
            ("contextual_advantage", "coordination_scale"),
        )
        self.assertEqual(
            prior.variables[0][2],
            {"mu": 0.0, "sigma": 1.0, "shape": 4},
        )
        self.assertEqual(prior.variables[1][2], {"lam": 1.0})

    def test_asynchronous_best_response_uses_latest_neighbor_choice(self) -> None:
        graph = csr_matrix(np.asarray([[0.0, 1.0], [1.0, 0.0]]))
        result = strategic_best_response(
            np.asarray([1, -1], dtype=np.int8),
            coauthorship=graph,
            area_shares=np.asarray(
                [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
            ),
            contextual_advantage=np.zeros(4),
            coordination_cost=2.0,
            sweeps=1,
        )
        np.testing.assert_array_equal(result, [-1, -1])

    def test_identical_seeds_reproduce_parameters_and_preferences(self) -> None:
        model = StrategicConventionModel()
        context = scientist_context()
        parameters = {
            "contextual_advantage": np.asarray([0.2, -0.1, 0.3, -0.2]),
            "coordination_scale": np.asarray(0.7),
        }
        first_data = model.simulate(
            parameters,
            np.random.default_rng(17),
            **context,
        )
        second_data = model.simulate(
            parameters,
            np.random.default_rng(17),
            **context,
        )
        np.testing.assert_array_equal(
            first_data["preference"],
            second_data["preference"],
        )


if __name__ == "__main__":
    unittest.main()
