from __future__ import annotations

import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np


sys.modules.setdefault("pymc", ModuleType("pymc"))
sys.modules.setdefault("bayesflow", ModuleType("bayesflow"))

from base.model import INTERVAL_SECONDS, validate_contacts
from models import CONTACT_MODEL_REGISTRY, MODEL_REGISTRY
from models.contacts import SpatialConversationModel
from models.contacts.spatial_conversation import (
    move_unpaused,
    toroidal_pair_distances_squared,
    update_conversations,
)
from tests.contact_model_helpers import FakePyMC


PARAMETERS = {
    "interaction_radius": np.asarray(0.2),
    "movement_scale": np.asarray(0.05),
    "start_probability": np.asarray(0.5),
    "mean_duration_minutes": np.asarray(3.0),
}


class SpatialConversationTests(unittest.TestCase):
    def test_model_is_registered_by_stable_name(self) -> None:
        self.assertIs(
            CONTACT_MODEL_REGISTRY["spatial_conversation"],
            SpatialConversationModel,
        )
        self.assertIs(
            MODEL_REGISTRY["spatial_conversation"],
            SpatialConversationModel,
        )

    def test_prior_and_inference_variables(self) -> None:
        model = SpatialConversationModel()

        with patch("models.contacts.spatial_conversation.pm", FakePyMC):
            prior = model.build_prior(n_agents=6, n_steps=20)

        self.assertEqual(
            [(distribution, name) for distribution, name, _ in prior.variables],
            [
                ("Beta", "interaction_radius"),
                ("LogNormal", "movement_scale"),
                ("Beta", "start_probability"),
                ("LogNormal", "mean_duration_minutes"),
            ],
        )
        self.assertEqual(
            model.inference_variables,
            (
                "interaction_radius",
                "movement_scale",
                "start_probability",
                "mean_duration_minutes",
            ),
        )
        self.assertEqual(
            prior.variables[0][2],
            {"alpha": 1.0, "beta": 9.0},
        )
        self.assertEqual(
            prior.variables[1][2],
            {"mu": np.log(0.03), "sigma": 1.0},
        )
        self.assertEqual(
            prior.variables[2][2],
            {"alpha": 1.0, "beta": 4.0},
        )
        self.assertEqual(
            prior.variables[3][2],
            {"mu": np.log(3.0), "sigma": 0.75},
        )

    def test_toroidal_distance_uses_shortest_wrapped_path(self) -> None:
        positions = np.asarray(
            [
                [0.99, 0.50],
                [0.01, 0.50],
                [0.50, 0.50],
            ]
        )

        distances = toroidal_pair_distances_squared(
            positions,
            [0, 0],
            [1, 2],
        )

        np.testing.assert_allclose(distances, [0.02**2, 0.49**2])

    def test_only_unpaused_agents_move_and_positions_wrap(self) -> None:
        class MovementRng:
            def normal(self, loc, scale, size):
                self.arguments = (loc, scale, size)
                return np.asarray([[0.2, 0.0], [0.0, 0.2]])

        rng = MovementRng()
        moved = move_unpaused(
            rng,
            np.asarray([[0.9, 0.1], [0.4, 0.4], [0.2, 0.9]]),
            np.asarray([False, True, False]),
            0.05,
        )

        np.testing.assert_allclose(
            moved,
            [[0.1, 0.1], [0.4, 0.4], [0.2, 0.1]],
        )
        self.assertEqual(rng.arguments, (0.0, 0.05, (2, 2)))

    def test_old_edges_cannot_end_and_restart_in_same_step(self) -> None:
        class TransitionRng:
            def __init__(self) -> None:
                self.draws = iter(
                    [
                        np.asarray([0.25, 0.75]),
                        np.asarray([0.25]),
                    ]
                )

            def random(self, size):
                values = next(self.draws)
                self.size = size
                return values

        updated = update_conversations(
            TransitionRng(),
            active=[True, True, False, False],
            nearby=[True, True, True, False],
            stay_probability=0.5,
            start_probability=0.5,
        )

        np.testing.assert_array_equal(updated, [True, False, True, False])

    def test_multi_person_conversations_pause_every_participant(self) -> None:
        class ScriptedRng:
            def __init__(self) -> None:
                self.normal_calls = 0

            def random(self, size=None):
                if size == (3, 2):
                    return np.asarray(
                        [[0.10, 0.10], [0.11, 0.10], [0.12, 0.10]]
                    )
                return np.zeros(size, dtype=np.float64)

            def normal(self, loc, scale, size):
                del loc, scale
                self.normal_calls += 1
                return np.zeros(size, dtype=np.float64)

        rng = ScriptedRng()
        contacts = SpatialConversationModel().simulate(
            PARAMETERS,
            rng,
            n_agents=3,
            n_steps=2,
        )

        np.testing.assert_array_equal(
            contacts["t"],
            [INTERVAL_SECONDS] * 3 + [2 * INTERVAL_SECONDS] * 3,
        )
        np.testing.assert_array_equal(contacts["i"], [0, 0, 1, 0, 0, 1])
        np.testing.assert_array_equal(contacts["j"], [1, 2, 2, 1, 2, 2])
        self.assertEqual(rng.normal_calls, 1)

    def test_simulation_is_seeded_and_schema_compatible(self) -> None:
        model = SpatialConversationModel()
        first = model.simulate(
            PARAMETERS,
            np.random.default_rng(42),
            n_agents=6,
            n_steps=12,
        )
        second = model.simulate(
            PARAMETERS,
            np.random.default_rng(42),
            n_agents=6,
            n_steps=12,
        )

        validate_contacts(first)
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
        self.assertTrue(np.all(first["i"] < first["j"]))
        self.assertTrue(np.all(first["t"] % INTERVAL_SECONDS == 0))

    def test_empty_simulations_have_the_contact_schema(self) -> None:
        contacts = SpatialConversationModel().simulate(
            PARAMETERS,
            np.random.default_rng(1),
            n_agents=1,
            n_steps=10,
        )

        validate_contacts(contacts)
        self.assertEqual(len(contacts["t"]), 0)

    def test_invalid_context_and_parameters_are_rejected(self) -> None:
        model = SpatialConversationModel()
        with self.assertRaisesRegex(KeyError, "n_agents"):
            model.simulate(
                PARAMETERS,
                np.random.default_rng(1),
                n_steps=1,
            )
        with self.assertRaisesRegex(ValueError, "n_steps"):
            model.simulate(
                PARAMETERS,
                np.random.default_rng(1),
                n_agents=2,
                n_steps=-1,
            )

        invalid_parameters = {
            "interaction_radius": -0.1,
            "movement_scale": np.inf,
            "start_probability": 1.1,
            "mean_duration_minutes": 0.0,
        }
        for name, value in invalid_parameters.items():
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError,
                name,
            ):
                model.simulate(
                    {**PARAMETERS, name: np.asarray(value)},
                    np.random.default_rng(1),
                    n_agents=2,
                    n_steps=1,
                )


if __name__ == "__main__":
    unittest.main()
