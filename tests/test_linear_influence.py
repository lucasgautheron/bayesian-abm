from __future__ import annotations

import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np


sys.modules.setdefault("pymc", ModuleType("pymc"))
sys.modules.setdefault("bayesflow", ModuleType("bayesflow"))

from models import MODEL_REGISTRY, STORY_MODEL_REGISTRY
from models.stories import LinearInfluenceModel, validate_story_data
from models.stories.linear_influence import (
    exponential_lag_kernel,
    linear_influence_intensity,
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
    def Exponential(cls, name: str, **kwargs: object) -> object:
        return cls._add("Exponential", name, **kwargs)

    @classmethod
    def HalfNormal(cls, name: str, **kwargs: object) -> object:
        return cls._add("HalfNormal", name, **kwargs)


PARAMETERS = {
    "story_rate": np.asarray(2.0),
    "background_rate": np.asarray(0.0),
    "seed_shape": np.asarray(1.0),
    "seed_scale": np.asarray(2.0),
    "branching_alpha": np.asarray(2.0),
    "branching_beta": np.asarray(2.0),
    "influence_decay": np.asarray(0.5),
}


class LinearInfluenceTests(unittest.TestCase):
    def test_model_is_registered_by_stable_name(self) -> None:
        self.assertIs(
            STORY_MODEL_REGISTRY["linear_influence"],
            LinearInfluenceModel,
        )
        self.assertIs(
            MODEL_REGISTRY["linear_influence"],
            LinearInfluenceModel,
        )

    def test_prior_and_inference_variables(self) -> None:
        model = LinearInfluenceModel()

        with patch("models.stories.linear_influence.pm", FakePyMC):
            prior = model.build_prior(n_days=184)

        self.assertEqual(
            [(distribution, name) for distribution, name, _ in prior.variables],
            [
                ("LogNormal", "story_rate"),
                ("LogNormal", "background_rate"),
                ("Exponential", "seed_shape"),
                ("Exponential", "seed_scale"),
                ("Exponential", "branching_alpha"),
                ("Exponential", "branching_beta"),
                ("HalfNormal", "influence_decay"),
            ],
        )
        self.assertEqual(
            model.inference_variables,
            (
                "story_rate",
                "background_rate",
                "seed_shape",
                "seed_scale",
                "branching_alpha",
                "branching_beta",
                "influence_decay",
            ),
        )
        self.assertEqual(
            prior.variables[0][2],
            {"mu": np.log(20.0), "sigma": 1.0},
        )
        self.assertEqual(
            prior.variables[1][2],
            {"mu": np.log(2.0), "sigma": 1.0},
        )
        self.assertEqual(prior.variables[-1][2], {"sigma": 1.0})
        for _, _, kwargs in prior.variables[2:-1]:
            self.assertEqual(kwargs, {"lam": 1.0})

    def test_kernel_and_linear_intensity_are_hand_computed(self) -> None:
        kernel = exponential_lag_kernel(np.log(2.0), lag_count=3)

        np.testing.assert_allclose(kernel, [4.0 / 7, 2.0 / 7, 1.0 / 7])
        self.assertAlmostEqual(kernel.sum(), 1.0)
        self.assertAlmostEqual(
            linear_influence_intensity(
                [2, 4, 8],
                kernel,
                branching_ratio=0.5,
            ),
            3.0,
        )
        self.assertAlmostEqual(
            linear_influence_intensity(
                [1.0],
                exponential_lag_kernel(0.0, lag_count=12),
                branching_ratio=0.5,
            ),
            0.5,
        )
        self.assertAlmostEqual(
            linear_influence_intensity(
                [0.0],
                [1.0],
                branching_ratio=0.5,
                background_rate=2.0,
            ),
            2.0,
        )

    def test_simulation_is_seeded_and_schema_compatible(self) -> None:
        model = LinearInfluenceModel()

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

    def test_birth_seed_and_zero_offspring_are_ordered(self) -> None:
        class ScriptedRng:
            def __init__(self) -> None:
                self.poisson_calls = 0

            def poisson(self, lam, size=None):
                self.poisson_calls += 1
                if self.poisson_calls == 1:
                    self.birth_rate = lam
                    self.birth_size = size
                    return np.asarray([0, 1, 0, 0])
                if self.poisson_calls == 2:
                    self.seed_rates = np.asarray(lam)
                    return np.zeros_like(self.seed_rates, dtype=np.int64)
                self.last_intensity = np.asarray(lam)
                return np.zeros_like(self.last_intensity, dtype=np.int64)

            def gamma(self, shape, scale, size=None):
                self.seed_shape = shape
                self.seed_scale = scale
                return np.full(size, 2.0)

            def beta(self, a, b, size=None):
                self.branching_alpha = a
                self.branching_beta = b
                return np.full(size, 0.5)

        rng = ScriptedRng()
        mentions = LinearInfluenceModel().simulate(
            {
                **PARAMETERS,
                "influence_decay": np.asarray(0.0),
            },
            rng,
            n_days=4,
        )["mentions"]

        np.testing.assert_array_equal(mentions, [[0, 1, 0, 0]])
        self.assertEqual((rng.birth_rate, rng.birth_size), (2.0, 4))
        self.assertEqual((rng.seed_shape, rng.seed_scale), (1.0, 2.0))
        self.assertEqual((rng.branching_alpha, rng.branching_beta), (2.0, 2.0))
        np.testing.assert_array_equal(rng.seed_rates, [2.0])
        self.assertTrue(np.all(rng.last_intensity >= 0))

    def test_influence_uses_the_full_mention_history(self) -> None:
        class HistoryRng:
            def __init__(self) -> None:
                self.poisson_calls = 0
                self.intensities: list[float] = []

            def poisson(self, lam, size=None):
                self.poisson_calls += 1
                if self.poisson_calls == 1:
                    births = np.zeros(size, dtype=np.int64)
                    births[0] = 1
                    return births
                if self.poisson_calls == 2:
                    return np.zeros_like(np.asarray(lam), dtype=np.int64)
                self.intensities.append(float(np.asarray(lam).reshape(-1)[0]))
                return np.zeros_like(np.asarray(lam), dtype=np.int64)

            def gamma(self, shape, scale, size=None):
                return np.ones(size)

            def beta(self, a, b, size=None):
                return np.full(size, 0.5)

        rng = HistoryRng()
        LinearInfluenceModel().simulate(
            {**PARAMETERS, "influence_decay": np.asarray(0.0)},
            rng,
            n_days=12,
        )

        self.assertGreater(rng.intensities[-1], 0.0)
        self.assertAlmostEqual(rng.intensities[-1], 0.5 / 11.0)

    def test_background_rate_keeps_stories_reporting(self) -> None:
        class BackgroundRng:
            def __init__(self) -> None:
                self.poisson_calls = 0

            def poisson(self, lam, size=None):
                self.poisson_calls += 1
                if self.poisson_calls == 1:
                    births = np.zeros(size, dtype=np.int64)
                    births[0] = 1
                    return births
                if self.poisson_calls == 2:
                    return np.zeros_like(np.asarray(lam), dtype=np.int64)
                return np.asarray(lam, dtype=np.int64)

            def gamma(self, shape, scale, size=None):
                return np.ones(size)

            def beta(self, a, b, size=None):
                return np.zeros(size)

        mentions = LinearInfluenceModel().simulate(
            {**PARAMETERS, "background_rate": np.asarray(3.0)},
            BackgroundRng(),
            n_days=4,
        )["mentions"]

        np.testing.assert_array_equal(mentions, [[1, 3, 3, 3]])


if __name__ == "__main__":
    unittest.main()
