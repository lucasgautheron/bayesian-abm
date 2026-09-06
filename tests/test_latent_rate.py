from __future__ import annotations

import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np


sys.modules.setdefault("pymc", ModuleType("pymc"))
sys.modules.setdefault("bayesflow", ModuleType("bayesflow"))

from models import MODEL_REGISTRY, STORY_MODEL_REGISTRY
from datasets.story_daily.schema import validate_story_data
from models.stories import LatentRateModel
from models.stories.latent_rate import daily_death_probability


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
    def Normal(cls, name: str, **kwargs: object) -> object:
        return cls._add("Normal", name, **kwargs)

    @classmethod
    def HalfNormal(cls, name: str, **kwargs: object) -> object:
        return cls._add("HalfNormal", name, **kwargs)


PARAMETERS = {
    "story_rate": np.asarray(2.0),
    "rate_mu": np.asarray(0.0),
    "rate_sigma": np.asarray(1.0),
    "lifetime_scale": np.asarray(np.inf),
}


class LatentRateTests(unittest.TestCase):
    def test_model_is_registered_by_stable_name(self) -> None:
        self.assertIs(STORY_MODEL_REGISTRY["latent_rate"], LatentRateModel)
        self.assertIs(MODEL_REGISTRY["latent_rate"], LatentRateModel)

    def test_prior_and_inference_variables(self) -> None:
        model = LatentRateModel()

        with patch("models.stories.latent_rate.pm", FakePyMC):
            prior = model.build_prior(n_days=184)

        self.assertEqual(
            [(distribution, name) for distribution, name, _ in prior.variables],
            [
                ("LogNormal", "story_rate"),
                ("Normal", "rate_mu"),
                ("HalfNormal", "rate_sigma"),
                ("LogNormal", "lifetime_scale"),
            ],
        )
        self.assertEqual(
            model.inference_variables,
            ("story_rate", "rate_mu", "rate_sigma", "lifetime_scale"),
        )
        self.assertEqual(
            prior.variables[0][2],
            {"mu": np.log(20.0), "sigma": 1.0},
        )
        self.assertEqual(prior.variables[1][2], {"mu": 0.0, "sigma": 1.0})
        self.assertEqual(prior.variables[2][2], {"sigma": 1.0})
        self.assertEqual(
            prior.variables[3][2],
            {"mu": np.log(150.0), "sigma": 0.5},
        )

    def test_daily_death_probability_is_hand_computed(self) -> None:
        self.assertEqual(daily_death_probability(np.inf), 0.0)
        self.assertAlmostEqual(
            daily_death_probability(1.0),
            1.0 - np.exp(-1.0),
        )

    def test_simulation_is_seeded_and_schema_compatible(self) -> None:
        model = LatentRateModel()

        first = model.simulate(
            PARAMETERS,
            np.random.default_rng(42),
            n_days=12,
        )
        second = model.simulate(
            PARAMETERS,
            np.random.default_rng(42),
            n_days=12,
        )

        np.testing.assert_array_equal(first["mentions"], second["mentions"])
        validated = validate_story_data(first, n_days=12)
        self.assertEqual(validated["mentions"].shape[1], 12)
        self.assertEqual(validated["mentions"].dtype, np.float32)

    def test_birth_rate_and_lifetime_are_ordered(self) -> None:
        class ScriptedRng:
            def __init__(self) -> None:
                self.poisson_calls = 0

            def poisson(self, lam, size=None):
                self.poisson_calls += 1
                if self.poisson_calls == 1:
                    self.birth_rate = lam
                    self.birth_size = size
                    return np.asarray([0, 1, 0, 0])
                self.mention_rate = np.asarray(lam)
                return np.full(size, 3, dtype=np.int64)

            def lognormal(self, mean, sigma, size=None):
                self.rate_mu = mean
                self.rate_sigma = sigma
                return np.full(size, 2.0)

            def geometric(self, p, size=None):
                raise AssertionError("infinite lifetime should skip death")

        rng = ScriptedRng()
        mentions = LatentRateModel().simulate(
            PARAMETERS,
            rng,
            n_days=4,
        )["mentions"]

        np.testing.assert_array_equal(mentions, [[0, 3, 3, 3]])
        self.assertEqual((rng.birth_rate, rng.birth_size), (2.0, 4))
        self.assertEqual((rng.rate_mu, rng.rate_sigma), (0.0, 1.0))
        np.testing.assert_array_equal(rng.mention_rate.reshape(-1)[:1], [2.0])

    def test_geometric_death_stops_mentions(self) -> None:
        class DeadRng:
            def poisson(self, lam, size=None):
                if size == 3:
                    births = np.zeros(3, dtype=np.int64)
                    births[0] = 1
                    return births
                return np.full(size, 5, dtype=np.int64)

            def lognormal(self, mean, sigma, size=None):
                return np.ones(size)

            def geometric(self, p, size=None):
                self.death_probability = p
                return np.ones(size, dtype=np.int64)

        rng = DeadRng()
        mentions = LatentRateModel().simulate(
            {**PARAMETERS, "lifetime_scale": np.asarray(1.0)},
            rng,
            n_days=3,
        )["mentions"]

        np.testing.assert_array_equal(mentions, [[5, 0, 0]])
        self.assertAlmostEqual(rng.death_probability, 1.0 - np.exp(-1.0))


if __name__ == "__main__":
    unittest.main()
