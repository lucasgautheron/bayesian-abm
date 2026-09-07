from __future__ import annotations

import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np


sys.modules.setdefault("pymc", ModuleType("pymc"))
sys.modules.setdefault("bayesflow", ModuleType("bayesflow"))

from base.model import INTERVAL_SECONDS, validate_contacts
from models import CONTACT_MODEL_REGISTRY, MODEL_REGISTRY, RandomWalkModel
from models.contacts.random_walk import (
    network_neighbors,
    simulate_walk_conversations,
)
from tests.contact_model_helpers import FakePyMC


PARAMETERS = {
    "p_edge": np.asarray(0.5),
    "p_move": np.asarray(0.8),
    "p_conversation": np.asarray(0.7),
    "network_edges": np.ones(6, dtype=np.int8),
}


class PriorAndNetworkTests(unittest.TestCase):
    def test_model_is_registered_by_stable_name(self) -> None:
        self.assertIs(
            CONTACT_MODEL_REGISTRY["random_walk"],
            RandomWalkModel,
        )
        self.assertIs(MODEL_REGISTRY["random_walk"], RandomWalkModel)

    def test_prior_and_inference_variables(self) -> None:
        model = RandomWalkModel()

        with patch("models.contacts.random_walk.pm", FakePyMC):
            prior = model.build_prior(n_agents=4, n_steps=10)

        self.assertEqual(
            [(distribution, name) for distribution, name, _ in prior.variables],
            [
                ("Beta", "p_edge"),
                ("Beta", "p_move"),
                ("Beta", "p_conversation"),
                ("Bernoulli", "network_edges"),
            ],
        )
        self.assertEqual(
            model.inference_variables,
            ("p_edge", "p_move", "p_conversation"),
        )
        self.assertEqual(
            prior.variables[0][2],
            {"alpha": 2.0, "beta": 8.0},
        )
        self.assertEqual(
            prior.variables[1][2],
            {"alpha": 2.0, "beta": 2.0},
        )
        self.assertEqual(
            prior.variables[2][2],
            {"alpha": 1.0, "beta": 9.0},
        )
        self.assertEqual(prior.variables[3][2]["shape"], 6)

    def test_network_neighbors_decodes_upper_triangle(self) -> None:
        neighbors = network_neighbors([1, 0, 1, 0, 0, 1], n_agents=4)

        expected = ([1, 3], [0], [3], [0, 2])
        for actual, values in zip(neighbors, expected):
            np.testing.assert_array_equal(
                actual,
                np.asarray(values, dtype=np.int32),
            )

    def test_network_edges_are_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "one value per"):
            network_neighbors([1, 0], n_agents=3)
        with self.assertRaisesRegex(ValueError, "zeroes and ones"):
            network_neighbors([1, 2, 0], n_agents=3)


class SimulationTests(unittest.TestCase):
    def test_reciprocal_successes_are_one_undirected_contact(self) -> None:
        neighbors = (
            np.asarray([1], dtype=np.int32),
            np.asarray([0], dtype=np.int32),
        )

        contacts = simulate_walk_conversations(
            np.random.default_rng(1),
            neighbors=neighbors,
            n_steps=1,
            p_move=1.0,
            p_conversation=1.0,
        )

        np.testing.assert_array_equal(
            contacts["t"],
            np.asarray([INTERVAL_SECONDS], dtype=np.int32),
        )
        np.testing.assert_array_equal(contacts["i"], [0])
        np.testing.assert_array_equal(contacts["j"], [1])

    def test_isolated_walkers_stay_silent(self) -> None:
        empty_neighbors = tuple(
            np.empty(0, dtype=np.int32) for _ in range(3)
        )

        contacts = simulate_walk_conversations(
            np.random.default_rng(2),
            neighbors=empty_neighbors,
            n_steps=5,
            p_move=1.0,
            p_conversation=1.0,
        )

        validate_contacts(contacts)
        self.assertEqual(len(contacts["t"]), 0)

    def test_simulation_is_reproducible_and_schema_compatible(self) -> None:
        model = RandomWalkModel()
        first = model.simulate(
            PARAMETERS,
            np.random.default_rng(42),
            n_agents=4,
            n_steps=20,
        )
        second = model.simulate(
            PARAMETERS,
            np.random.default_rng(42),
            n_agents=4,
            n_steps=20,
        )

        validate_contacts(first)
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
        self.assertTrue(np.all(first["i"] < first["j"]))
        self.assertTrue(np.all(first["t"] % INTERVAL_SECONDS == 0))

    def test_invalid_context_and_probabilities_are_rejected(self) -> None:
        model = RandomWalkModel()
        with self.assertRaisesRegex(ValueError, "n_agents"):
            model.build_prior(n_agents=1)
        with self.assertRaisesRegex(ValueError, "n_agents"):
            model.simulate(
                PARAMETERS,
                np.random.default_rng(1),
                n_agents=1,
                n_steps=2,
            )
        with self.assertRaisesRegex(ValueError, "n_steps"):
            model.simulate(
                PARAMETERS,
                np.random.default_rng(1),
                n_agents=4,
                n_steps=0,
            )
        with self.assertRaisesRegex(ValueError, "p_move"):
            model.simulate(
                {**PARAMETERS, "p_move": np.asarray(1.1)},
                np.random.default_rng(1),
                n_agents=4,
                n_steps=2,
            )
        with self.assertRaisesRegex(ValueError, "p_conversation"):
            model.simulate(
                {**PARAMETERS, "p_conversation": np.asarray(-0.1)},
                np.random.default_rng(1),
                n_agents=4,
                n_steps=2,
            )


if __name__ == "__main__":
    unittest.main()
