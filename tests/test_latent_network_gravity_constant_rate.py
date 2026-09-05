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
from models.contacts import LatentNetworkGravityConstantRateModel
from models.contacts.latent_network_gravity import (
    PARETO_ALPHA,
    PARETO_MINIMUM,
)
from tests.contact_model_helpers import FakePyMC, ScriptedRng


PARAMETERS = {
    "group_rate": np.asarray(2.0),
    "start_rate": np.asarray(0.5),
    "mean_duration_minutes": np.asarray(np.inf),
    "activity_sigma": np.asarray(1.0),
    "mu_within": np.asarray(0.8),
    "mu_ratio": np.asarray(0.25),
    "eta": np.asarray(1.0e6),
}


class LatentNetworkGravityConstantRateTests(unittest.TestCase):
    def test_model_is_registered_by_renamed_stable_name(self) -> None:
        self.assertIs(
            CONTACT_MODEL_REGISTRY[
                "latent_network_gravity_constant_rate"
            ],
            LatentNetworkGravityConstantRateModel,
        )
        self.assertIs(
            MODEL_REGISTRY["latent_network_gravity_constant_rate"],
            LatentNetworkGravityConstantRateModel,
        )
        self.assertNotIn(
            "latent_network_constant_rate",
            CONTACT_MODEL_REGISTRY,
        )

    def test_prior_and_inference_variables_preserve_constant_model(
        self,
    ) -> None:
        model = LatentNetworkGravityConstantRateModel()

        with patch(
            "models.contacts.latent_network_gravity_constant_rate.pm",
            FakePyMC,
        ):
            prior = model.build_prior(n_agents=6, n_steps=20)

        self.assertEqual(
            [(distribution, name) for distribution, name, _ in prior.variables],
            [
                ("LogNormal", "group_rate"),
                ("LogNormal", "start_rate"),
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
                "mean_duration_minutes",
                "activity_sigma",
                "mu_within",
                "mu_ratio",
                "eta",
            ),
        )
        self.assertEqual(
            prior.variables[6][2],
            {"alpha": PARETO_ALPHA, "m": PARETO_MINIMUM},
        )

    def test_rates_are_constant_and_hazards_are_combined(self) -> None:
        rng = ScriptedRng(
            communities=[0, 0, 1],
            activities=[1.0, 2.0, 3.0],
        )
        empty = {
            key: np.empty(0, dtype=np.int32)
            for key in ("t", "i", "j")
        }

        with patch(
            "models.contacts.latent_network_gravity_constant_rate."
            "simulate_weighted_conversations",
            return_value=empty,
        ) as simulate:
            LatentNetworkGravityConstantRateModel().simulate(
                PARAMETERS,
                rng,
                n_agents=3,
                n_steps=3,
            )

        np.testing.assert_allclose(
            simulate.call_args.kwargs["hazards"],
            [1.6, 0.6, 1.2],
        )
        np.testing.assert_array_equal(
            simulate.call_args.kwargs["rates"],
            [0.5, 0.5, 0.5],
        )

    def test_simulation_is_seeded_and_schema_compatible(self) -> None:
        model = LatentNetworkGravityConstantRateModel()
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
