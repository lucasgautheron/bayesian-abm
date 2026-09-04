from __future__ import annotations

import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np


sys.modules.setdefault("pymc", ModuleType("pymc"))
sys.modules.setdefault("bayesflow", ModuleType("bayesflow"))

from models import MODEL_REGISTRY, STORY_MODEL_REGISTRY
from models.stories import GeneralizedSIRModel, validate_story_data
from models.stories.generalized_sir import (
    sir_infection_probability,
    weibull_mean_interest_duration,
    weibull_recovery_probability,
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
    def Exponential(cls, name: str, **kwargs: object) -> object:
        return cls._add("Exponential", name, **kwargs)

    @classmethod
    def LogNormal(cls, name: str, **kwargs: object) -> object:
        return cls._add("LogNormal", name, **kwargs)

    @classmethod
    def Uniform(cls, name: str, **kwargs: object) -> object:
        return cls._add("Uniform", name, **kwargs)


PARAMETERS = {
    "story_rate": np.asarray(2.0),
    "reproduction_number": np.asarray(2.0),
    "interest_scale": np.asarray(3.0),
    "interest_shape": np.asarray(1.2),
    "report_alpha": np.asarray(1.0),
    "report_beta": np.asarray(1.0),
    "report_ratio": np.asarray(0.0),
    "forget_scale": np.asarray(np.inf),
}


class GeneralizedSIRTests(unittest.TestCase):
    def test_model_is_registered_by_stable_name(self) -> None:
        self.assertIs(
            STORY_MODEL_REGISTRY["generalized_sir"],
            GeneralizedSIRModel,
        )
        self.assertIs(
            MODEL_REGISTRY["generalized_sir"],
            GeneralizedSIRModel,
        )

    def test_prior_and_inference_variables(self) -> None:
        model = GeneralizedSIRModel()

        with patch("models.stories.generalized_sir.pm", FakePyMC):
            prior = model.build_prior(n_days=184)

        self.assertEqual(
            [(distribution, name) for distribution, name, _ in prior.variables],
            [
                ("LogNormal", "story_rate"),
                ("LogNormal", "reproduction_number"),
                ("LogNormal", "interest_scale"),
                ("Exponential", "interest_shape"),
                ("Exponential", "report_alpha"),
                ("Exponential", "report_beta"),
                ("Uniform", "report_ratio"),
                ("LogNormal", "forget_scale"),
            ],
        )
        self.assertEqual(
            model.inference_variables,
            (
                "story_rate",
                "reproduction_number",
                "interest_scale",
                "interest_shape",
                "report_alpha",
                "report_beta",
                "report_ratio",
                "forget_scale",
            ),
        )
        self.assertEqual(
            prior.variables[0][2],
            {"mu": np.log(20.0), "sigma": 1.0},
        )
        self.assertEqual(
            prior.variables[1][2],
            {"mu": np.log(2.0), "sigma": 0.75},
        )
        self.assertEqual(
            prior.variables[2][2],
            {"mu": np.log(7.0), "sigma": 1.0},
        )
        self.assertEqual(
            prior.variables[6][2],
            {"lower": 0.0, "upper": 1.0},
        )
        self.assertEqual(
            prior.variables[7][2],
            {"mu": np.log(80.0), "sigma": 0.75},
        )
        for index in (3, 4, 5):
            self.assertEqual(prior.variables[index][2], {"lam": 1.0})

    def test_weibull_and_infection_helpers_are_hand_computed(self) -> None:
        mean_duration = weibull_mean_interest_duration(2.0, 1.0)
        recovery = weibull_recovery_probability(
            0,
            scale=2.0,
            shape=1.0,
        )
        infection = sir_infection_probability(
            100,
            population=1_000,
            reproduction_number=2.0,
            mean_interest_duration=4.0,
        )

        self.assertAlmostEqual(mean_duration, 2.0)
        self.assertAlmostEqual(recovery, 1.0 - np.exp(-0.5))
        self.assertAlmostEqual(infection, 1.0 - np.exp(-0.05))
        self.assertEqual(
            sir_infection_probability(
                100,
                population=1_000,
                reproduction_number=0.0,
                mean_interest_duration=4.0,
            ),
            0.0,
        )
        np.testing.assert_allclose(
            sir_infection_probability(
                np.asarray([0, 100]),
                population=1_000,
                reproduction_number=2.0,
                mean_interest_duration=4.0,
            ),
            [0.0, 1.0 - np.exp(-0.05)],
        )

    def test_simulation_is_seeded_and_schema_compatible(self) -> None:
        model = GeneralizedSIRModel()

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

        np.testing.assert_array_equal(
            first["mentions"],
            second["mentions"],
        )
        validated = validate_story_data(first, n_days=12)
        self.assertEqual(validated["mentions"].shape[1], 12)
        self.assertEqual(validated["mentions"].dtype, np.float32)

    def test_infections_and_recoveries_follow_story_birth(self) -> None:
        class ScriptedRng:
            def __init__(self) -> None:
                self.binomial_calls: list[tuple[object, object]] = []

            def poisson(self, lam, size=None):
                self.birth_rate = lam
                self.birth_size = size
                return np.asarray([0, 1, 0])

            def beta(self, a, b, size=None):
                self.beta_call = (a, b, size)
                return np.full(size, 0.5)

            def binomial(self, n, p, size=None):
                self.binomial_calls.append((np.asarray(n), np.asarray(p)))
                if len(self.binomial_calls) == 1:
                    return np.asarray(1)
                if len(self.binomial_calls) == 2:
                    return np.asarray(2)
                if len(self.binomial_calls) == 4:
                    return np.asarray(2)
                return np.zeros_like(np.asarray(n), dtype=np.int64)

        rng = ScriptedRng()
        with patch("models.stories.generalized_sir.SOURCE_POPULATION", 10):
            mentions = GeneralizedSIRModel().simulate(
                {
                    "story_rate": np.asarray(2.0),
                    "reproduction_number": np.asarray(2.0),
                    "interest_scale": np.asarray(1.0),
                    "interest_shape": np.asarray(1.0),
                    "report_alpha": np.asarray(2.0),
                    "report_beta": np.asarray(4.0),
                    "report_ratio": np.asarray(0.0),
                    "forget_scale": np.asarray(np.inf),
                },
                rng,
                n_days=3,
            )["mentions"]

        np.testing.assert_array_equal(mentions, [[0, 1, 2]])
        self.assertEqual((rng.birth_rate, rng.birth_size), (2.0, 3))
        self.assertEqual(rng.beta_call, (2.0, 4.0, 1))
        np.testing.assert_array_equal(
            np.asarray(rng.binomial_calls[0][0]).reshape(-1),
            [1],
        )
        self.assertAlmostEqual(
            float(np.asarray(rng.binomial_calls[0][1]).reshape(-1)[0]),
            0.5,
        )
        self.assertEqual(
            int(np.asarray(rng.binomial_calls[1][0]).reshape(-1)[0]),
            9,
        )
        self.assertAlmostEqual(
            float(np.asarray(rng.binomial_calls[1][1]).reshape(-1)[0]),
            1.0 - np.exp(-0.2),
        )
        np.testing.assert_array_equal(
            np.asarray(rng.binomial_calls[3][0]).reshape(-1),
            [3],
        )
        self.assertAlmostEqual(
            float(np.asarray(rng.binomial_calls[3][1]).reshape(-1)[0]),
            0.5,
        )

    def test_zero_report_probability_suppresses_mentions(self) -> None:
        class SilentRng:
            def poisson(self, lam, size=None):
                return np.asarray([1, 0, 1])

            def beta(self, a, b, size=None):
                return np.zeros(size)

            def binomial(self, n, p, size=None):
                return np.zeros_like(np.asarray(n), dtype=np.int64)

        mentions = GeneralizedSIRModel().simulate(
            PARAMETERS,
            SilentRng(),
            n_days=3,
        )["mentions"]

        self.assertEqual(mentions.sum(), 0)

    def test_recovered_sources_keep_reporting_until_they_die(self) -> None:
        class RecoveredRng:
            def __init__(self) -> None:
                self.binomial_calls = 0

            def poisson(self, lam, size=None):
                return np.asarray([1, 0, 0])

            def beta(self, a, b, size=None):
                return np.ones(size)

            def binomial(self, n, p, size=None):
                self.binomial_calls += 1
                values = np.asarray(n)
                if self.binomial_calls <= 2:
                    return np.zeros_like(values, dtype=np.int64)
                if self.binomial_calls == 3:
                    return values.copy()
                return np.full(values.shape, 4)

        mentions = GeneralizedSIRModel().simulate(
            {**PARAMETERS, "report_ratio": np.asarray(0.5)},
            RecoveredRng(),
            n_days=3,
        )["mentions"]

        np.testing.assert_array_equal(mentions, [[0, 4, 4]])

    def test_dead_sources_stop_reporting(self) -> None:
        class DeadRng:
            def __init__(self) -> None:
                self.binomial_calls = 0

            def poisson(self, lam, size=None):
                return np.asarray([1, 0, 0])

            def beta(self, a, b, size=None):
                return np.ones(size)

            def binomial(self, n, p, size=None):
                self.binomial_calls += 1
                values = np.asarray(n)
                if self.binomial_calls <= 2:
                    return np.zeros_like(values, dtype=np.int64)
                return values.copy()

        mentions = GeneralizedSIRModel().simulate(
            {
                **PARAMETERS,
                "report_ratio": np.asarray(1.0),
                "forget_scale": np.asarray(1e-9),
            },
            DeadRng(),
            n_days=3,
        )["mentions"]

        np.testing.assert_array_equal(mentions, [[0, 0, 0]])


if __name__ == "__main__":
    unittest.main()
