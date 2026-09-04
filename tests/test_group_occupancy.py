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
from models.contacts import GroupOccupancyModel
from models.contacts.group_occupancy import (
    end_probability,
    n_groups,
    start_probabilities,
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


PARAMETERS = {
    "group_rate": np.asarray(2.0),
    "start_rate": np.asarray(0.5),
    "mean_duration_minutes": np.asarray(np.inf),
    "activity_sigma": np.asarray(1.0),
    "between_ratio": np.asarray(0.0),
}


class GroupOccupancyTests(unittest.TestCase):
    def test_model_is_registered_by_stable_name(self) -> None:
        self.assertIs(
            CONTACT_MODEL_REGISTRY["group_occupancy"],
            GroupOccupancyModel,
        )
        self.assertIs(MODEL_REGISTRY["group_occupancy"], GroupOccupancyModel)

    def test_prior_and_inference_variables(self) -> None:
        model = GroupOccupancyModel()

        with patch("models.contacts.group_occupancy.pm", FakePyMC):
            prior = model.build_prior(n_agents=6, n_steps=20)

        self.assertEqual(
            [(distribution, name) for distribution, name, _ in prior.variables],
            [
                ("LogNormal", "group_rate"),
                ("LogNormal", "start_rate"),
                ("LogNormal", "mean_duration_minutes"),
                ("HalfNormal", "activity_sigma"),
                ("Beta", "between_ratio"),
            ],
        )
        self.assertEqual(
            model.inference_variables,
            (
                "group_rate",
                "start_rate",
                "mean_duration_minutes",
                "activity_sigma",
                "between_ratio",
            ),
        )
        self.assertEqual(
            prior.variables[0][2],
            {"mu": np.log(6.0), "sigma": 0.75},
        )
        self.assertEqual(
            prior.variables[1][2],
            {"mu": np.log(0.001), "sigma": 1.0},
        )
        self.assertEqual(
            prior.variables[2][2],
            {"mu": np.log(3.0), "sigma": 0.5},
        )
        self.assertEqual(prior.variables[3][2], {"sigma": 1.0})
        self.assertEqual(
            prior.variables[4][2],
            {"alpha": 1.0, "beta": 20.0},
        )

    def test_n_groups_is_rounded_and_clipped(self) -> None:
        self.assertEqual(n_groups(2.4, 6), 2)
        self.assertEqual(n_groups(0.4, 6), 1)
        self.assertEqual(n_groups(20.0, 6), 6)

    def test_end_probability_is_hand_computed(self) -> None:
        self.assertEqual(end_probability(np.inf), 0.0)
        self.assertAlmostEqual(end_probability(1.0), 1.0 - np.exp(-1.0))

    def test_start_probabilities_scale_between_group_pairs(self) -> None:
        first, second, probabilities = start_probabilities(
            np.asarray([0, 0, 1, 1]),
            np.asarray([1.0, 2.0, 3.0, 4.0]),
            0.5,
            0.1,
        )

        np.testing.assert_array_equal(first, [0, 0, 0, 1, 1, 2])
        np.testing.assert_array_equal(second, [1, 2, 3, 2, 3, 3])
        np.testing.assert_allclose(
            probabilities,
            [
                1.0 - np.exp(-0.5 * 1.0 * 2.0),
                1.0 - np.exp(-0.05 * 1.0 * 3.0),
                1.0 - np.exp(-0.05 * 1.0 * 4.0),
                1.0 - np.exp(-0.05 * 2.0 * 3.0),
                1.0 - np.exp(-0.05 * 2.0 * 4.0),
                1.0 - np.exp(-0.5 * 3.0 * 4.0),
            ],
        )

    def test_simulation_is_seeded_and_schema_compatible(self) -> None:
        model = GroupOccupancyModel()

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

    def test_emitted_pairs_are_within_group_and_ordered(self) -> None:
        class ScriptedRng:
            def integers(self, high, size=None, dtype=np.int32):
                self.n_groups = high
                return np.asarray([0, 0, 1, 1], dtype=dtype)

            def lognormal(self, mean, sigma, size=None):
                return np.ones(size)

            def random(self, size=None):
                return np.zeros(size)

        contacts = GroupOccupancyModel().simulate(
            PARAMETERS,
            ScriptedRng(),
            n_agents=4,
            n_steps=2,
        )
        validated = validate_contacts(contacts)

        np.testing.assert_array_equal(validated["i"], [0, 2, 0, 2])
        np.testing.assert_array_equal(validated["j"], [1, 3, 1, 3])
        np.testing.assert_array_equal(
            validated["t"],
            [
                INTERVAL_SECONDS,
                INTERVAL_SECONDS,
                2 * INTERVAL_SECONDS,
                2 * INTERVAL_SECONDS,
            ],
        )
        self.assertTrue(np.all(validated["i"] < validated["j"]))

    def test_between_group_pairs_can_start(self) -> None:
        class ScriptedRng:
            def integers(self, high, size=None, dtype=np.int32):
                return np.asarray([0, 0, 1, 1], dtype=dtype)

            def lognormal(self, mean, sigma, size=None):
                return np.ones(size)

            def random(self, size=None):
                return np.zeros(size)

        contacts = GroupOccupancyModel().simulate(
            {**PARAMETERS, "between_ratio": np.asarray(1.0)},
            ScriptedRng(),
            n_agents=4,
            n_steps=1,
        )
        validated = validate_contacts(contacts)
        pairs = set(zip(validated["i"].tolist(), validated["j"].tolist()))
        self.assertEqual(
            pairs,
            {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)},
        )

    def test_infinite_duration_skips_end_draws_and_stays_on(self) -> None:
        class ScriptedRng:
            def __init__(self) -> None:
                self.random_calls = 0

            def integers(self, high, size=None, dtype=np.int32):
                return np.zeros(size, dtype=dtype)

            def lognormal(self, mean, sigma, size=None):
                return np.ones(size)

            def random(self, size=None):
                self.random_calls += 1
                if self.random_calls == 1:
                    return np.zeros(size)
                raise AssertionError("infinite duration should stay on")

        contacts = GroupOccupancyModel().simulate(
            PARAMETERS,
            ScriptedRng(),
            n_agents=2,
            n_steps=1,
        )

        np.testing.assert_array_equal(contacts["i"], [0])
        np.testing.assert_array_equal(contacts["j"], [1])

    def test_off_pairs_can_start_and_on_pairs_can_end(self) -> None:
        class ScriptedRng:
            def __init__(self) -> None:
                self.random_calls = 0

            def integers(self, high, size=None, dtype=np.int32):
                return np.zeros(size, dtype=dtype)

            def lognormal(self, mean, sigma, size=None):
                return np.ones(size)

            def random(self, size=None):
                self.random_calls += 1
                if self.random_calls == 1:
                    return np.ones(size)
                if self.random_calls == 2:
                    return np.ones(size)
                return np.zeros(size)

        contacts = GroupOccupancyModel().simulate(
            {
                **PARAMETERS,
                "mean_duration_minutes": np.asarray(1.0),
            },
            ScriptedRng(),
            n_agents=2,
            n_steps=2,
        )

        np.testing.assert_array_equal(contacts["t"], [2 * INTERVAL_SECONDS])
        np.testing.assert_array_equal(contacts["i"], [0])
        np.testing.assert_array_equal(contacts["j"], [1])


if __name__ == "__main__":
    unittest.main()
