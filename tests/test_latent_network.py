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
from models.contacts import LatentNetworkModel
from models.contacts.latent_network import (
    GP_LOG_SIGMA,
    LENGTHSCALE_MEAN_MINUTES,
    draw_log_ou_path,
    draw_sbm_edges,
    edge_probabilities,
    pair_start_probability,
    start_probability,
)


class FakePrior:
    current: FakePrior | None = None

    def __init__(self) -> None:
        self.variables: list[tuple[str, str, dict]] = []

    def __enter__(self) -> FakePrior:
        FakePrior.current = self
        return self

    def __exit__(self, *_: object) -> None:
        FakePrior.current = None


class FakePyMC:
    Model = FakePrior

    @staticmethod
    def _add(distribution: str, name: str, **kwargs: object) -> object:
        assert FakePrior.current is not None
        FakePrior.current.variables.append((distribution, name, kwargs))
        return object()

    @classmethod
    def LogNormal(cls, name: str, **kwargs: object) -> object:
        return cls._add("LogNormal", name, **kwargs)

    @classmethod
    def HalfNormal(cls, name: str, **kwargs: object) -> object:
        return cls._add("HalfNormal", name, **kwargs)

    @classmethod
    def Beta(cls, name: str, **kwargs: object) -> object:
        return cls._add("Beta", name, **kwargs)

    @classmethod
    def Exponential(cls, name: str, **kwargs: object) -> object:
        return cls._add("Exponential", name, **kwargs)


PARAMETERS = {
    "group_rate": np.asarray(2.0),
    "start_rate": np.asarray(0.5),
    "lengthscale": np.asarray(1.0e9),
    "mean_duration_minutes": np.asarray(np.inf),
    "activity_sigma": np.asarray(1.0),
    "between_ratio": np.asarray(0.0),
    "p_edge": np.asarray(1.0),
}


class AlwaysFirstRng:
    def integers(self, high, size=None, dtype=np.int32):
        if size == 4:
            return np.asarray([0, 0, 1, 1], dtype=dtype)
        return np.zeros(size, dtype=dtype)

    def lognormal(self, mean, sigma, size=None):
        return np.ones(size)

    def normal(self, loc=0.0, scale=1.0, size=None):
        if size is None:
            return 0.0
        return np.zeros(size)

    def random(self, size=None):
        if size is None:
            return 0.0
        return np.zeros(size)


class LatentNetworkTests(unittest.TestCase):
    def test_model_is_registered_by_stable_name(self) -> None:
        self.assertIs(
            CONTACT_MODEL_REGISTRY["latent_network"],
            LatentNetworkModel,
        )
        self.assertIs(MODEL_REGISTRY["latent_network"], LatentNetworkModel)
        self.assertNotIn("friend_conversation", MODEL_REGISTRY)
        self.assertNotIn("friend_conversation", CONTACT_MODEL_REGISTRY)

    def test_prior_and_inference_variables(self) -> None:
        model = LatentNetworkModel()

        with patch("models.contacts.latent_network.pm", FakePyMC):
            prior = model.build_prior(n_agents=6, n_steps=20)

        self.assertEqual(
            [(distribution, name) for distribution, name, _ in prior.variables],
            [
                ("LogNormal", "group_rate"),
                ("LogNormal", "start_rate"),
                ("Exponential", "lengthscale"),
                ("LogNormal", "mean_duration_minutes"),
                ("HalfNormal", "activity_sigma"),
                ("Beta", "between_ratio"),
                ("Beta", "p_edge"),
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
                "between_ratio",
                "p_edge",
            ),
        )
        self.assertEqual(
            prior.variables[0][2],
            {"mu": np.log(6.0), "sigma": 0.75},
        )
        self.assertEqual(
            prior.variables[2][2],
            {"lam": 1.0 / LENGTHSCALE_MEAN_MINUTES},
        )
        self.assertEqual(
            prior.variables[6][2],
            {"alpha": 2.0, "beta": 5.0},
        )

    def test_sbm_uses_between_ratio_on_cross_group_pairs(self) -> None:
        first, second, probabilities = edge_probabilities(
            np.asarray([0, 0, 1, 1]),
            0.8,
            0.25,
        )

        np.testing.assert_array_equal(first, [0, 0, 0, 1, 1, 2])
        np.testing.assert_array_equal(second, [1, 2, 3, 2, 3, 3])
        np.testing.assert_allclose(
            probabilities,
            [0.8, 0.2, 0.2, 0.2, 0.2, 0.8],
        )

    def test_draw_sbm_edges_is_undirected_upper_triangle(self) -> None:
        class ScriptedRng:
            def random(self, size=None):
                return np.asarray([0.0, 0.9, 0.9, 0.9, 0.9, 0.0])

        first, second = draw_sbm_edges(
            ScriptedRng(),
            np.asarray([0, 0, 1, 1]),
            1.0,
            0.0,
        )
        np.testing.assert_array_equal(first, [0, 2])
        np.testing.assert_array_equal(second, [1, 3])

    def test_pair_start_probability_is_hand_computed(self) -> None:
        np.testing.assert_allclose(
            pair_start_probability([1.0, 3.0], [2.0, 4.0], 0.5),
            [
                1.0 - np.exp(-0.5 * 1.0 * 2.0),
                1.0 - np.exp(-0.5 * 3.0 * 4.0),
            ],
        )

    def test_simulation_is_seeded_and_schema_compatible(self) -> None:
        model = LatentNetworkModel()

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

    def test_occupancy_uses_only_sbm_edges(self) -> None:
        contacts = LatentNetworkModel().simulate(
            PARAMETERS,
            AlwaysFirstRng(),
            n_agents=4,
            n_steps=1,
        )
        validated = validate_contacts(contacts)
        pairs = set(zip(validated["i"].tolist(), validated["j"].tolist()))
        self.assertEqual(pairs, {(0, 1), (2, 3)})

    def test_empty_sbm_emits_no_contacts(self) -> None:
        contacts = LatentNetworkModel().simulate(
            {**PARAMETERS, "p_edge": np.asarray(0.0)},
            AlwaysFirstRng(),
            n_agents=3,
            n_steps=2,
        )
        validated = validate_contacts(contacts)
        self.assertEqual(len(validated["t"]), 0)

    def test_ou_path_matches_hand_computed_recurrence(self) -> None:
        class ScriptedRng:
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
        expected[2] = phi * expected[1] + (-0.5) * innovation_sd

        path = draw_log_ou_path(
            ScriptedRng(),
            n_steps=3,
            lengthscale=lengthscale,
            sigma=sigma,
        )

        np.testing.assert_allclose(path, expected)
        self.assertEqual(GP_LOG_SIGMA, 1.0)

    def test_ou_path_is_seeded_and_constant_for_huge_lengthscale(self) -> None:
        first = draw_log_ou_path(
            np.random.default_rng(7),
            n_steps=8,
            lengthscale=3.0,
        )
        second = draw_log_ou_path(
            np.random.default_rng(7),
            n_steps=8,
            lengthscale=3.0,
        )
        constant = draw_log_ou_path(
            np.random.default_rng(7),
            n_steps=8,
            lengthscale=np.inf,
        )

        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(constant, constant[0])

    def test_short_lengthscale_changes_occupancy_starts(self) -> None:
        class ThresholdRng(AlwaysFirstRng):
            def random(self, size=None):
                if size is None:
                    return 0.5
                return np.full(size, 0.5)

        with patch(
            "models.contacts.latent_network.draw_log_ou_path",
            return_value=np.asarray([-20.0, 5.0]),
        ):
            contacts = LatentNetworkModel().simulate(
                {
                    **PARAMETERS,
                    "start_rate": np.asarray(1.0),
                    "lengthscale": np.asarray(1.0),
                    "mean_duration_minutes": np.asarray(1.0e-9),
                },
                ThresholdRng(),
                n_agents=2,
                n_steps=2,
            )

        validated = validate_contacts(contacts)
        self.assertEqual(validated["t"].tolist(), [2 * INTERVAL_SECONDS])
        self.assertEqual(validated["i"].tolist(), [0])
        self.assertEqual(validated["j"].tolist(), [1])

    def test_huge_lengthscale_keeps_constant_start_probability(self) -> None:
        products = np.asarray([2.0, 0.5])
        path = draw_log_ou_path(
            AlwaysFirstRng(),
            n_steps=4,
            lengthscale=np.inf,
        )
        rates = 0.4 * np.exp(path)

        np.testing.assert_array_equal(path, 0.0)
        for rate in rates:
            np.testing.assert_allclose(
                start_probability(products, rate),
                start_probability(products, rates[0]),
            )


if __name__ == "__main__":
    unittest.main()
