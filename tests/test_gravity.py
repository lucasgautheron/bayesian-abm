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
from models.contacts import GravityModel
from models.contacts.latent_network_gravity import LENGTHSCALE_MEAN_MINUTES
from tests.contact_model_helpers import FakePyMC, ScriptedRng


PARAMETERS = {
    "start_rate": np.asarray(0.5),
    "lengthscale": np.asarray(1.0e9),
    "mean_duration_minutes": np.asarray(np.inf),
    "activity_sigma": np.asarray(1.0),
}


class GravityTests(unittest.TestCase):
    def test_model_is_registered_by_stable_name(self) -> None:
        self.assertIs(CONTACT_MODEL_REGISTRY["gravity"], GravityModel)
        self.assertIs(MODEL_REGISTRY["gravity"], GravityModel)

    def test_prior_and_inference_variables_only_include_gravity_terms(
        self,
    ) -> None:
        model = GravityModel()

        with patch("models.contacts.gravity.pm", FakePyMC):
            prior = model.build_prior(n_agents=6, n_steps=20)

        self.assertEqual(
            [(distribution, name) for distribution, name, _ in prior.variables],
            [
                ("LogNormal", "start_rate"),
                ("Exponential", "lengthscale"),
                ("LogNormal", "mean_duration_minutes"),
                ("HalfNormal", "activity_sigma"),
            ],
        )
        self.assertEqual(
            model.inference_variables,
            (
                "start_rate",
                "lengthscale",
                "mean_duration_minutes",
                "activity_sigma",
            ),
        )
        self.assertEqual(
            prior.variables[1][2],
            {"lam": 1.0 / LENGTHSCALE_MEAN_MINUTES},
        )

    def test_pair_hazards_are_activity_products_with_uniform_baseline(
        self,
    ) -> None:
        rng = ScriptedRng(activities=[1.0, 2.0, 3.0])
        empty = {
            key: np.empty(0, dtype=np.int32)
            for key in ("t", "i", "j")
        }

        with patch(
            "models.contacts.gravity.simulate_weighted_conversations",
            return_value=empty,
        ) as simulate:
            GravityModel().simulate(
                PARAMETERS,
                rng,
                n_agents=3,
                n_steps=2,
            )

        np.testing.assert_array_equal(
            simulate.call_args.kwargs["hazards"],
            [2.0, 3.0, 6.0],
        )

    def test_simulation_is_seeded_and_schema_compatible(self) -> None:
        model = GravityModel()
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
