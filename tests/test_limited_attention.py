from __future__ import annotations

import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np


sys.modules.setdefault("pymc", ModuleType("pymc"))
sys.modules.setdefault("bayesflow", ModuleType("bayesflow"))

from models import MODEL_REGISTRY, STORY_MODEL_REGISTRY
from models.stories import LimitedAttentionModel, validate_story_data
from models.stories.limited_attention import (
    preferential_attachment_followers,
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
    def Uniform(name: str, **kwargs: object) -> object:
        assert FakePrior.current is not None
        FakePrior.current.variables.append(("Uniform", name, kwargs))
        return object()


PARAMETERS = {
    "p_new": np.asarray(0.45),
    "p_read": np.asarray(0.016),
    "p_memory": np.asarray(0.4),
}


class LimitedAttentionTests(unittest.TestCase):
    def test_model_is_registered_by_stable_name(self) -> None:
        self.assertIs(
            STORY_MODEL_REGISTRY["limited_attention"],
            LimitedAttentionModel,
        )
        self.assertIs(
            MODEL_REGISTRY["limited_attention"],
            LimitedAttentionModel,
        )

    def test_prior_and_inference_variables(self) -> None:
        model = LimitedAttentionModel()

        with patch("models.stories.limited_attention.pm", FakePyMC):
            prior = model.build_prior(n_days=184)

        self.assertEqual(
            [(distribution, name) for distribution, name, _ in prior.variables],
            [
                ("Uniform", "p_new"),
                ("Uniform", "p_read"),
                ("Uniform", "p_memory"),
            ],
        )
        self.assertEqual(
            model.inference_variables,
            ("p_new", "p_read", "p_memory"),
        )
        self.assertEqual(
            prior.variables[0][2],
            {"lower": 0.0, "upper": 1.0},
        )
        self.assertEqual(
            prior.variables[1][2],
            {"lower": 0.0, "upper": 1.0},
        )
        self.assertEqual(
            prior.variables[2][2],
            {"lower": 0.0, "upper": 1.0},
        )

    def test_preferential_attachment_network_is_reproducible(self) -> None:
        first = preferential_attachment_followers(
            20,
            3,
            np.random.default_rng(4),
        )
        second = preferential_attachment_followers(
            20,
            3,
            np.random.default_rng(4),
        )

        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
            self.assertEqual(left.dtype, np.int32)
        self.assertEqual(
            sum(map(len, first)),
            sum(min(3, user) for user in range(1, 20)),
        )
        for user, followers in enumerate(first):
            self.assertTrue(np.all(followers > user))

    def test_simulation_is_seeded_and_schema_compatible(self) -> None:
        model = LimitedAttentionModel()
        with (
            patch("models.stories.limited_attention.N_USERS", 30),
            patch("models.stories.limited_attention.FOLLOWS_PER_USER", 2),
        ):
            first = model.simulate(
                PARAMETERS,
                np.random.default_rng(8),
                n_days=3,
            )
            second = model.simulate(
                PARAMETERS,
                np.random.default_rng(8),
                n_days=3,
            )

        np.testing.assert_array_equal(
            first["mentions"],
            second["mentions"],
        )
        validated = validate_story_data(first, n_days=3)
        self.assertEqual(validated["mentions"].shape[1], 3)
        self.assertEqual(validated["mentions"].dtype, np.float32)

    def test_memory_can_replace_an_attended_story(self) -> None:
        class ScriptedRng:
            def __init__(self) -> None:
                self.integer_values = iter([0, 1, 1, 0, 0])
                self.random_values = iter(
                    [
                        0.0,
                        0.0,
                        0.9,
                        np.asarray([0.0]),
                        0.0,
                        0.9,
                        np.asarray([1.0, 1.0]),
                    ]
                )

            def integers(self, *_args, **_kwargs):
                return next(self.integer_values)

            def random(self, _size=None):
                return next(self.random_values)

        followers = (
            np.asarray([1], dtype=np.int32),
            np.asarray([0], dtype=np.int32),
        )
        parameters = {
            "p_new": np.asarray(0.5),
            "p_read": np.asarray(0.5),
            "p_memory": np.asarray(0.5),
        }
        with (
            patch("models.stories.limited_attention.N_USERS", 2),
            patch("models.stories.limited_attention.RETENTION_DAYS", 2),
            patch(
                "models.stories.limited_attention."
                "preferential_attachment_followers",
                return_value=followers,
            ),
        ):
            mentions = LimitedAttentionModel().simulate(
                parameters,
                ScriptedRng(),
                n_days=2,
            )["mentions"]

        np.testing.assert_array_equal(
            mentions,
            [[1, 0], [1, 1]],
        )

    def test_expired_screen_posts_are_not_rebroadcast(self) -> None:
        class ScriptedRng:
            def __init__(self) -> None:
                self.integer_values = iter([0, 1, 1, 1])
                self.random_values = iter(
                    [0.0, 0.9, np.asarray([0.0]), 0.9, 0.9]
                )

            def integers(self, *_args, **_kwargs):
                return next(self.integer_values)

            def random(self, _size=None):
                return next(self.random_values)

        followers = (
            np.asarray([1], dtype=np.int32),
            np.asarray([], dtype=np.int32),
        )
        parameters = {
            "p_new": np.asarray(0.5),
            "p_read": np.asarray(0.5),
            "p_memory": np.asarray(0.0),
        }
        with (
            patch("models.stories.limited_attention.N_USERS", 2),
            patch("models.stories.limited_attention.RETENTION_DAYS", 1),
            patch(
                "models.stories.limited_attention."
                "preferential_attachment_followers",
                return_value=followers,
            ),
        ):
            mentions = LimitedAttentionModel().simulate(
                parameters,
                ScriptedRng(),
                n_days=2,
            )["mentions"]

        np.testing.assert_array_equal(mentions, [[2, 0]])


if __name__ == "__main__":
    unittest.main()
