from __future__ import annotations

import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np


sys.modules.setdefault("pymc", ModuleType("pymc"))
sys.modules.setdefault("bayesflow", ModuleType("bayesflow"))

from base.model import validate_contacts
from models import CONTACT_MODEL_REGISTRY, MODEL_REGISTRY
from models.contacts import LatentNetworkGravityModel
from models.contacts.latent_network_gravity import (
    GP_LOG_SIGMA,
    LENGTHSCALE_MEAN_MINUTES,
    MEAN_CLIP,
    PARETO_ALPHA,
    PARETO_MINIMUM,
    affinity_means,
    draw_affinities,
    draw_log_ou_path,
    draw_weighted_indices,
    pair_product_sum,
)
from tests.contact_model_helpers import FakePyMC, ScriptedRng


PARAMETERS = {
    "group_rate": np.asarray(2.0),
    "start_rate": np.asarray(0.5),
    "lengthscale": np.asarray(1.0e9),
    "mean_duration_minutes": np.asarray(np.inf),
    "activity_sigma": np.asarray(1.0),
    "mu_within": np.asarray(0.8),
    "mu_ratio": np.asarray(0.25),
    "eta": np.asarray(1.0e6),
}


class LatentNetworkGravityTests(unittest.TestCase):
    def test_model_is_registered_by_stable_name(self) -> None:
        self.assertIs(
            CONTACT_MODEL_REGISTRY["latent_network_gravity"],
            LatentNetworkGravityModel,
        )
        self.assertIs(
            MODEL_REGISTRY["latent_network_gravity"],
            LatentNetworkGravityModel,
        )

    def test_prior_and_inference_variables_preserve_full_model(self) -> None:
        model = LatentNetworkGravityModel()

        with patch("models.contacts.latent_network_gravity.pm", FakePyMC):
            prior = model.build_prior(n_agents=6, n_steps=20)

        self.assertEqual(
            [(distribution, name) for distribution, name, _ in prior.variables],
            [
                ("LogNormal", "group_rate"),
                ("LogNormal", "start_rate"),
                ("Exponential", "lengthscale"),
                ("LogNormal", "mean_duration_minutes"),
                ("HalfNormal", "activity_sigma"),
                ("Beta", "mu_within"),
                ("Beta", "mu_ratio"),
                ("Pareto", "eta"),
            ],
        )
        self.assertEqual(
            model.inference_variables,
            (
                "group_rate",
                "start_rate",
                "lengthscale",
                "mean_duration_minutes",
                "activity_sigma",
                "mu_within",
                "mu_ratio",
                "eta",
            ),
        )
        self.assertEqual(
            prior.variables[2][2],
            {"lam": 1.0 / LENGTHSCALE_MEAN_MINUTES},
        )
        self.assertEqual(
            prior.variables[7][2],
            {"alpha": PARETO_ALPHA, "m": PARETO_MINIMUM},
        )

    def test_pair_hazards_combine_affinity_and_activity_products(self) -> None:
        rng = ScriptedRng(
            communities=[0, 0, 1],
            activities=[1.0, 2.0, 3.0],
        )
        empty = {
            key: np.empty(0, dtype=np.int32)
            for key in ("t", "i", "j")
        }

        with patch(
            "models.contacts.latent_network_gravity."
            "simulate_weighted_conversations",
            return_value=empty,
        ) as simulate:
            LatentNetworkGravityModel().simulate(
                PARAMETERS,
                rng,
                n_agents=3,
                n_steps=2,
            )

        np.testing.assert_allclose(
            simulate.call_args.kwargs["hazards"],
            [1.6, 0.6, 1.2],
        )

    def test_affinity_helpers_preserve_weighted_sbm_behavior(self) -> None:
        first, second, means = affinity_means(
            np.asarray([0, 0, 1, 1]),
            0.8,
            0.25,
        )
        np.testing.assert_array_equal(first, [0, 0, 0, 1, 1, 2])
        np.testing.assert_array_equal(second, [1, 2, 3, 2, 3, 3])
        np.testing.assert_allclose(
            means,
            [0.8, 0.2, 0.2, 0.2, 0.2, 0.8],
        )
        _, _, clipped = affinity_means(np.asarray([0, 1]), 0.0, 0.0)
        np.testing.assert_allclose(clipped, [MEAN_CLIP])

        activities = np.asarray([0.5, 1.0, 1.5, 2.0])
        pair_first, pair_second = np.triu_indices(4, k=1)
        self.assertAlmostEqual(
            pair_product_sum(activities),
            float(
                np.sum(activities[pair_first] * activities[pair_second])
            ),
        )

    def test_draw_helpers_are_deterministic_when_scripted(self) -> None:
        rng = ScriptedRng()
        means = np.asarray([0.8, 0.2])
        np.testing.assert_allclose(
            draw_affinities(rng, means, 10.0),
            means,
        )

        class CdfRng:
            def random(self, size=None):
                del size
                return np.asarray([0.0, 0.49, 0.51])

        np.testing.assert_array_equal(
            draw_weighted_indices(CdfRng(), [1.0, 1.0], 3),
            [0, 0, 1],
        )

    def test_ou_path_matches_hand_computed_recurrence(self) -> None:
        class OuRng:
            def normal(self, loc=0.0, scale=1.0, size=None):
                if size is None:
                    return loc + scale * 0.5
                return loc + scale * np.asarray([0.25, -0.5])

        lengthscale = 2.0
        sigma = 2.0
        phi = np.exp(-1.0 / lengthscale)
        innovation_sd = sigma * np.sqrt(1.0 - phi * phi)
        expected = np.asarray(
            [
                0.5 * sigma,
                phi * 0.5 * sigma + 0.25 * innovation_sd,
                0.0,
            ]
        )
        expected[2] = phi * expected[1] - 0.5 * innovation_sd

        path = draw_log_ou_path(
            OuRng(),
            n_steps=3,
            lengthscale=lengthscale,
            sigma=sigma,
        )

        np.testing.assert_allclose(path, expected)
        self.assertEqual(GP_LOG_SIGMA, 1.0)

    def test_simulation_is_seeded_and_schema_compatible(self) -> None:
        model = LatentNetworkGravityModel()
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


if __name__ == "__main__":
    unittest.main()
