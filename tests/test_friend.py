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

from base.model import validate_contacts
from models import FriendModel, MODEL_CLASSES, MODEL_REGISTRY
from models.contacts.friend import (
    choose_partner,
    draw_friend_network,
    zero_truncated_poisson,
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
        variable = object()
        FakePrior.current.variables.append(
            (distribution, name, {**kwargs, "variable": variable})
        )
        return variable

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

    @classmethod
    def Uniform(cls, name: str, **kwargs: object) -> object:
        return cls._add("Uniform", name, **kwargs)


PARAMETERS = {
    "p_minute": np.asarray(0.8),
    "mean_duration_minutes": np.asarray(0.5),
    "reputation": np.asarray([-1.0, 0.0, 1.0, 2.0, 0.5, -0.5]),
    "phi": np.asarray(1.5),
    "block_strength": np.asarray(0.8),
    "p_friend": np.asarray(0.7),
}


class PriorAndHelperTests(unittest.TestCase):
    def test_model_is_registered_by_stable_name(self) -> None:
        self.assertIn(FriendModel, MODEL_CLASSES)
        self.assertIs(MODEL_REGISTRY["friend_conversation"], FriendModel)

    def test_prior_and_inference_variables(self) -> None:
        model = FriendModel()

        with patch("models.contacts.friend.pm", FakePyMC):
            prior = model.build_prior(n_agents=6, n_steps=20)

        names = [name for _, name, _ in prior.variables]
        self.assertEqual(
            names,
            [
                "p_minute",
                "mean_duration_minutes",
                "reputation_sigma",
                "reputation",
                "phi",
                "block_strength",
                "p_friend",
            ],
        )
        self.assertNotIn("reputation", model.inference_variables)
        self.assertEqual(
            set(model.inference_variables),
            {
                "p_minute",
                "mean_duration_minutes",
                "reputation_sigma",
                "phi",
                "block_strength",
                "p_friend",
            },
        )

        phi = next(item for item in prior.variables if item[1] == "phi")
        self.assertEqual(phi[0], "Exponential")
        self.assertEqual(phi[2]["lam"], 0.1)

        block_strength = next(
            item
            for item in prior.variables
            if item[1] == "block_strength"
        )
        self.assertEqual(block_strength[0], "Uniform")
        self.assertEqual(block_strength[2]["lower"], 0.0)
        self.assertEqual(block_strength[2]["upper"], 1.0)

    def test_tiny_rate_zero_truncated_poisson_avoids_rejection(self) -> None:
        class FakeRng:
            def random(self) -> float:
                return 0.5

            def poisson(self, _rate: float) -> int:
                raise AssertionError("tiny rates should use inverse sampling")

        self.assertEqual(zero_truncated_poisson(FakeRng(), 1e-6), 1)

    def test_network_uses_zero_truncated_poisson_and_is_undirected(
        self,
    ) -> None:
        class FakeRng:
            def __init__(self) -> None:
                self.poisson_draws = iter([0, 0, 2])
                self.poisson_calls = 0

            def poisson(self, lam: float) -> int:
                self.lam = lam
                self.poisson_calls += 1
                return next(self.poisson_draws)

            def integers(
                self,
                high: int,
                *,
                size: int,
                dtype: type[np.int32],
            ) -> np.ndarray:
                self.high = high
                return np.zeros(size, dtype=dtype)

            def random(self, size: int) -> np.ndarray:
                return np.full(size, 0.5)

        rng = FakeRng()
        n_communities, communities, adjacency = draw_friend_network(
            rng,
            n_agents=4,
            phi=2.5,
            block_strength=1.0,
        )

        self.assertEqual(n_communities, 2)
        self.assertEqual(rng.poisson_calls, 3)
        self.assertEqual(rng.lam, 2.5)
        self.assertEqual(rng.high, 2)
        np.testing.assert_array_equal(communities, np.zeros(4))
        np.testing.assert_array_equal(adjacency, adjacency.T)
        np.testing.assert_array_equal(np.diag(adjacency), np.zeros(4))
        self.assertTrue(adjacency[np.triu_indices(4, k=1)].all())

    def test_within_and_between_community_edge_probabilities(self) -> None:
        class FakeRng:
            def poisson(self, _lam: float) -> int:
                return 2

            def integers(
                self,
                _high: int,
                *,
                size: int,
                dtype: type[np.int32],
            ) -> np.ndarray:
                self.assert_size = size
                return np.asarray([0, 0, 1, 1], dtype=dtype)

            def random(self, size: int) -> np.ndarray:
                return np.full(size, 0.5)

        _, communities, adjacency = draw_friend_network(
            FakeRng(),
            n_agents=4,
            phi=1.0,
            block_strength=1.0,
        )

        expected = communities[:, None] == communities[None, :]
        np.fill_diagonal(expected, False)
        np.testing.assert_array_equal(adjacency, expected)

    def test_friend_choice_reputation_choice_and_failed_attempt(self) -> None:
        class FakeRng:
            def random(self) -> float:
                return 0.0

            def choice(
                self,
                values: list[int],
                p: np.ndarray | None = None,
            ) -> int:
                if p is None:
                    return values[0]
                return values[int(np.argmax(p))]

        rng = FakeRng()
        reputations = np.asarray([0.0, 10.0, -10.0])

        friend = choose_partner(
            rng,
            reputations,
            candidates=[1, 2],
            friends=[2],
            p_friend=1.0,
        )
        reputation_choice = choose_partner(
            rng,
            reputations,
            candidates=[1, 2],
            friends=[2],
            p_friend=0.0,
        )
        failed_attempt = choose_partner(
            rng,
            reputations,
            candidates=[1, 2],
            friends=[0],
            p_friend=1.0,
        )

        self.assertEqual(friend, 2)
        self.assertEqual(reputation_choice, 1)
        self.assertIsNone(failed_attempt)

    def test_network_rejects_invalid_probabilities(self) -> None:
        with self.assertRaisesRegex(ValueError, "phi"):
            draw_friend_network(
                np.random.default_rng(1),
                n_agents=4,
                phi=0.0,
                block_strength=0.8,
            )
        with self.assertRaisesRegex(ValueError, "block_strength"):
            draw_friend_network(
                np.random.default_rng(1),
                n_agents=4,
                phi=1.0,
                block_strength=-0.1,
            )


class SimulationTests(unittest.TestCase):
    def test_simulation_is_reproducible_and_schema_compatible(self) -> None:
        model = FriendModel()

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
        model = FriendModel()
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

    def test_friendship_attempts_fail_without_friends(self) -> None:
        contacts = FriendModel().simulate(
            {
                **PARAMETERS,
                "p_minute": np.asarray(1.0),
                "block_strength": np.asarray(0.0),
                "p_friend": np.asarray(1.0),
            },
            np.random.default_rng(7),
            n_agents=6,
            n_steps=20,
        )

        for values in contacts.values():
            self.assertEqual(len(values), 0)

    def test_simulation_rejects_invalid_context_and_parameters(self) -> None:
        model = FriendModel()

        with self.assertRaisesRegex(ValueError, "n_agents"):
            model.simulate(
                PARAMETERS,
                np.random.default_rng(1),
                n_agents=1,
                n_steps=5,
            )
        with self.assertRaisesRegex(ValueError, "reputation"):
            model.simulate(
                {**PARAMETERS, "reputation": np.asarray([0.0, 1.0])},
                np.random.default_rng(1),
                n_agents=6,
                n_steps=5,
            )
        with self.assertRaisesRegex(ValueError, "p_friend"):
            model.simulate(
                {**PARAMETERS, "p_friend": np.asarray(1.1)},
                np.random.default_rng(1),
                n_agents=6,
                n_steps=5,
            )


if __name__ == "__main__":
    unittest.main()
