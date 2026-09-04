from __future__ import annotations

import sys
import unittest
from types import ModuleType
from unittest.mock import patch

import numpy as np


# Keep these simulation tests runnable without the tutorial's heavy optional
# dependencies installed in the active development interpreter.
sys.modules.setdefault("pymc", ModuleType("pymc"))
sys.modules.setdefault("bayesflow", ModuleType("bayesflow"))

from base.abm import validate_contacts
from models import MODEL_CLASSES, MODEL_REGISTRY, ReputationConversationModel
from models.reputation import (
    duration_in_steps,
    partner_probabilities,
    per_step_probability,
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
    def Beta(cls, name: str, **kwargs: object) -> object:
        return cls._add("Beta", name, **kwargs)

    @classmethod
    def LogNormal(cls, name: str, **kwargs: object) -> object:
        return cls._add("LogNormal", name, **kwargs)

    @classmethod
    def Exponential(cls, name: str, **kwargs: object) -> object:
        return cls._add("Exponential", name, **kwargs)

    @classmethod
    def Normal(cls, name: str, **kwargs: object) -> object:
        return cls._add("Normal", name, **kwargs)


PARAMETERS = {
    "p_minute": np.asarray(0.8),
    "mean_duration_minutes": np.asarray(0.5),
    "duration_shape": np.asarray(2.0),
    "reputation": np.asarray([-1.0, 0.0, 1.0, 2.0, 0.5, -0.5]),
}


class PriorAndHelperTests(unittest.TestCase):
    def test_model_is_registered_by_stable_name(self) -> None:
        self.assertIn(ReputationConversationModel, MODEL_CLASSES)
        self.assertIs(
            MODEL_REGISTRY["reputation_conversation"],
            ReputationConversationModel,
        )

    def test_prior_and_inference_variables(self) -> None:
        model = ReputationConversationModel()

        with patch("models.reputation.pm", FakePyMC):
            prior = model.build_prior(n_agents=6, n_steps=20)

        names = [name for _, name, _ in prior.variables]
        self.assertEqual(
            names,
            [
                "p_minute",
                "mean_duration_minutes",
                "duration_shape",
                "reputation_sigma",
                "reputation",
            ],
        )
        self.assertNotIn("reputation", model.inference_variables)
        self.assertIn("reputation_sigma", model.inference_variables)
        reputation_sigma = next(
            variable
            for variable in prior.variables
            if variable[1] == "reputation_sigma"
        )
        self.assertEqual(reputation_sigma[0], "Exponential")
        self.assertEqual(reputation_sigma[2]["lam"], 1.0)

    def test_minute_probability_conversion_is_exact(self) -> None:
        for p_minute in (0.01, 0.1, 0.5, 1.0):
            p_step = per_step_probability(p_minute)
            reconstructed = 1.0 - (1.0 - p_step) ** 3
            self.assertAlmostEqual(reconstructed, p_minute)

    def test_partner_probabilities_are_a_softmax(self) -> None:
        reputations = np.asarray([-1.0, 0.0, 2.0])

        probabilities = partner_probabilities(reputations, [0, 2])

        expected = np.exp([-3.0, 0.0])
        expected /= expected.sum()
        np.testing.assert_allclose(probabilities, expected)
        self.assertGreater(probabilities[1], probabilities[0])

    def test_duration_uses_mean_shape_parameterization(self) -> None:
        class FakeRng:
            def gamma(self, shape: float, *, scale: float) -> float:
                self.shape = shape
                self.scale = scale
                return 1.1

        rng = FakeRng()
        steps = duration_in_steps(rng, mean_minutes=6.0, shape=3.0)

        self.assertEqual(rng.shape, 3.0)
        self.assertEqual(rng.scale, 2.0)
        self.assertEqual(steps, 4)


class SimulationTests(unittest.TestCase):
    def test_simulation_is_reproducible_and_schema_compatible(self) -> None:
        model = ReputationConversationModel()

        first = model.simulate(
            PARAMETERS,
            np.random.default_rng(42),
            n_agents=6,
            n_steps=30,
        )
        second = model.simulate(
            PARAMETERS,
            np.random.default_rng(42),
            n_agents=6,
            n_steps=30,
        )

        validate_contacts(first)
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])

    def test_an_agent_has_at_most_one_partner_per_bin(self) -> None:
        model = ReputationConversationModel()
        contacts = model.simulate(
            {**PARAMETERS, "p_minute": np.asarray(1.0)},
            np.random.default_rng(7),
            n_agents=6,
            n_steps=50,
        )

        for time in np.unique(contacts["t"]):
            active = contacts["t"] == time
            endpoints = np.concatenate(
                (contacts["i"][active], contacts["j"][active])
            )
            self.assertEqual(len(endpoints), len(np.unique(endpoints)))


if __name__ == "__main__":
    unittest.main()
