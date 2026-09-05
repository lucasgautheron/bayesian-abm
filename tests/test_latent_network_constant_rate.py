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
from models.contacts import LatentNetworkConstantRateModel
from models.contacts.latent_network import (
    PARETO_ALPHA,
    PARETO_MINIMUM,
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
    def Pareto(cls, name: str, **kwargs: object) -> object:
        return cls._add("Pareto", name, **kwargs)


PARAMETERS = {
    "group_rate": np.asarray(2.0),
    "start_rate": np.asarray(0.5),
    "mean_duration_minutes": np.asarray(np.inf),
    "activity_sigma": np.asarray(1.0),
    "mu_within": np.asarray(1.0),
    "mu_ratio": np.asarray(0.0),
    "eta": np.asarray(1.0e6),
}


class AlwaysFirstRng:
    def integers(self, high, size=None, dtype=np.int32):
        if size == 4:
            return np.asarray([0, 0, 1, 1], dtype=dtype)
        return np.zeros(size, dtype=dtype)

    def lognormal(self, mean, sigma, size=None):
        return np.ones(size)

    def random(self, size=None):
        if size is None:
            return 0.0
        return np.zeros(size)

    def beta(self, a, b):
        return np.asarray(a, dtype=np.float64) / (
            np.asarray(a, dtype=np.float64) + np.asarray(b, dtype=np.float64)
        )

    def poisson(self, lam, size=None):
        value = 1 if float(np.asarray(lam).reshape(-1)[0]) >= 0.5 else 0
        if size is None:
            return value
        return np.full(size, value)


class LatentNetworkConstantRateTests(unittest.TestCase):
    def test_model_is_registered_by_stable_name(self) -> None:
        self.assertIs(
            CONTACT_MODEL_REGISTRY["latent_network_constant_rate"],
            LatentNetworkConstantRateModel,
        )
        self.assertIs(
            MODEL_REGISTRY["latent_network_constant_rate"],
            LatentNetworkConstantRateModel,
        )

    def test_prior_and_inference_variables(self) -> None:
        model = LatentNetworkConstantRateModel()

        with patch(
            "models.contacts.latent_network_constant_rate.pm",
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
        self.assertNotIn(
            "lengthscale",
            [name for _, name, _ in prior.variables],
        )
        self.assertEqual(
            prior.variables[4][2],
            {"alpha": 2.0, "beta": 5.0},
        )
        self.assertEqual(
            prior.variables[5][2],
            {"alpha": 1.0, "beta": 20.0},
        )
        self.assertEqual(
            prior.variables[6][2],
            {"alpha": PARETO_ALPHA, "m": PARETO_MINIMUM},
        )

    def test_simulation_is_seeded_and_schema_compatible(self) -> None:
        model = LatentNetworkConstantRateModel()

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

    def test_occupancy_prefers_within_group_pairs(self) -> None:
        contacts = LatentNetworkConstantRateModel().simulate(
            PARAMETERS,
            AlwaysFirstRng(),
            n_agents=4,
            n_steps=1,
        )
        validated = validate_contacts(contacts)
        pairs = set(zip(validated["i"].tolist(), validated["j"].tolist()))
        self.assertTrue(pairs)
        self.assertTrue(pairs <= {(0, 1), (2, 3)})

    def test_zero_affinity_emits_no_contacts(self) -> None:
        contacts = LatentNetworkConstantRateModel().simulate(
            {**PARAMETERS, "mu_within": np.asarray(0.0)},
            AlwaysFirstRng(),
            n_agents=3,
            n_steps=2,
        )
        validated = validate_contacts(contacts)
        self.assertEqual(len(validated["t"]), 0)

    def test_start_count_is_capped_at_idle_pairs(self) -> None:
        n_agents = 3
        n_pairs = n_agents * (n_agents - 1) // 2

        class HugePoissonRng(AlwaysFirstRng):
            def poisson(self, lam, size=None):
                return 10**12

            def random(self, size=None):
                count = 1 if size is None else int(np.prod(np.atleast_1d(size)))
                if count > n_pairs:
                    raise AssertionError(f"drew {count} uniforms")
                if size is None:
                    return 0.0
                return np.zeros(size)

        contacts = LatentNetworkConstantRateModel().simulate(
            {**PARAMETERS, "start_rate": np.asarray(1.0e9)},
            HugePoissonRng(),
            n_agents=n_agents,
            n_steps=1,
        )
        validated = validate_contacts(contacts)
        self.assertLessEqual(len(validated["t"]), n_pairs)

    def test_constant_rate_restarts_after_the_idle_gap(self) -> None:
        class ThresholdRng(AlwaysFirstRng):
            def random(self, size=None):
                if size is None:
                    return 0.5
                return np.full(size, 0.5)

        contacts = LatentNetworkConstantRateModel().simulate(
            {
                **PARAMETERS,
                "start_rate": np.asarray(1.0),
                "mean_duration_minutes": np.asarray(1.0e-9),
            },
            ThresholdRng(),
            n_agents=2,
            n_steps=3,
        )

        validated = validate_contacts(contacts)
        self.assertEqual(
            validated["t"].tolist(),
            [INTERVAL_SECONDS, 3 * INTERVAL_SECONDS],
        )
        self.assertEqual(validated["i"].tolist(), [0, 0])
        self.assertEqual(validated["j"].tolist(), [1, 1])

    def test_simulate_does_not_draw_a_gaussian_process(self) -> None:
        class NoNormalRng(AlwaysFirstRng):
            def normal(self, loc=0.0, scale=1.0, size=None):
                raise AssertionError("OU path must not be drawn")

        contacts = LatentNetworkConstantRateModel().simulate(
            PARAMETERS,
            NoNormalRng(),
            n_agents=4,
            n_steps=3,
        )
        validate_contacts(contacts)

    def test_large_simulation_is_seeded(self) -> None:
        model = LatentNetworkConstantRateModel()
        parameters = {
            **PARAMETERS,
            "start_rate": np.asarray(0.001),
            "mu_within": np.asarray(0.3),
            "mu_ratio": np.asarray(0.05),
            "eta": np.asarray(4.0),
        }
        first = model.simulate(
            parameters,
            np.random.default_rng(3),
            n_agents=48,
            n_steps=8,
        )
        second = model.simulate(
            parameters,
            np.random.default_rng(3),
            n_agents=48,
            n_steps=8,
        )
        validate_contacts(first)
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])


if __name__ == "__main__":
    unittest.main()
