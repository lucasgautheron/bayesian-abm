from __future__ import annotations

import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np


sys.modules.setdefault("pymc", ModuleType("pymc"))
sys.modules.setdefault("bayesflow", ModuleType("bayesflow"))

from models import STORY_MODEL_CLASSES, STORY_MODEL_REGISTRY
from models.stories import (
    STORY_SUMMARY_STATISTICS,
    StoryCompetitionModel,
    make_story_summaries,
    ranked_story_mentions,
    validate_story_data,
)
from models.stories.competition import story_choice_probabilities


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
    "report_rate": np.asarray(10.0),
    "beta_age": np.asarray(-1.0),
    "beta_appeal": np.asarray(1.5),
}


class PriorAndHelperTests(unittest.TestCase):
    def test_model_is_registered_by_stable_name(self) -> None:
        self.assertIn(StoryCompetitionModel, STORY_MODEL_CLASSES)
        self.assertIs(
            STORY_MODEL_REGISTRY["story_competition"],
            StoryCompetitionModel,
        )

    def test_prior_and_inference_variables(self) -> None:
        model = StoryCompetitionModel()

        with patch("models.stories.competition.pm", FakePyMC):
            prior = model.build_prior(n_days=184)

        self.assertEqual(
            [(distribution, name) for distribution, name, _ in prior.variables],
            [
                ("LogNormal", "story_rate"),
                ("LogNormal", "report_rate"),
                ("Normal", "beta_age"),
                ("HalfNormal", "beta_appeal"),
            ],
        )
        self.assertEqual(
            model.inference_variables,
            (
                "story_rate",
                "report_rate",
                "beta_age",
                "beta_appeal",
            ),
        )
        story_prior = prior.variables[0][2]
        report_prior = prior.variables[1][2]
        self.assertAlmostEqual(story_prior["mu"], np.log(5.0))
        self.assertEqual(story_prior["sigma"], 0.75)
        self.assertAlmostEqual(report_prior["mu"], np.log(1_000.0))
        self.assertEqual(report_prior["sigma"], 1.5)

    def test_choice_probabilities_use_normalized_age_and_appeal(self) -> None:
        probabilities = story_choice_probabilities(
            [0, 9],
            [1.0, -1.0],
            beta_age=-2.0,
            beta_appeal=0.5,
            n_days=10,
        )

        expected_logits = np.asarray([0.5, -2.5])
        expected = np.exp(expected_logits - expected_logits.max())
        expected /= expected.sum()
        np.testing.assert_allclose(probabilities, expected)

    def test_top_story_selection_is_permutation_invariant(self) -> None:
        mentions = np.asarray(
            [
                [0, 2, 0],
                [1, 0, 0],
                [0, 0, 3],
                [2, 0, 0],
            ],
            dtype=np.float32,
        )
        expected = np.asarray(
            [
                [0, 0, 3],
                [2, 0, 0],
                [0, 2, 0],
                [1, 0, 0],
            ],
            dtype=np.float32,
        )

        first = ranked_story_mentions(
            {"mentions": mentions},
            n_days=3,
            story_count=5,
        )
        second = ranked_story_mentions(
            {"mentions": mentions[[2, 0, 3, 1]]},
            n_days=3,
            story_count=5,
        )
        truncated = ranked_story_mentions(
            {"mentions": mentions},
            n_days=3,
            story_count=3,
        )

        np.testing.assert_array_equal(first, expected)
        np.testing.assert_array_equal(second, expected)
        np.testing.assert_array_equal(truncated, expected[:3])
        self.assertEqual(first.dtype, np.float32)

    def test_story_summaries_are_scalar_and_hand_computed(self) -> None:
        summaries = make_story_summaries(
            n_days=3,
            story_count=2,
        )
        self.assertEqual(set(summaries), set(STORY_SUMMARY_STATISTICS))
        data = {
            "mentions": np.asarray(
                [[1, 0, 1], [0, 2, 0], [0, 0, 1]],
                dtype=np.float32,
            )
        }
        values = {name: summary(data) for name, summary in summaries.items()}

        self.assertEqual(values["selected_story_count"], 2.0)
        self.assertEqual(values["total_mentions"], 4.0)
        self.assertAlmostEqual(
            values["daily_total_stdev"],
            np.std([1, 2, 1]),
        )
        self.assertEqual(values["mean_reporting_story_count"], 1.0)
        self.assertEqual(
            values["story_mentions_coefficient_of_variation"],
            0.0,
        )
        self.assertEqual(values["mean_reporting_lifetime_days"], 2.0)
        self.assertEqual(values["mention_concentration"], 0.5)
        self.assertTrue(
            all(np.asarray(value).shape == () for value in values.values())
        )

    def test_empty_story_population_has_finite_scalar_summaries(self) -> None:
        summaries = make_story_summaries(n_days=3, story_count=2)
        empty = {"mentions": np.empty((0, 3))}

        values = np.asarray(
            [summary(empty) for summary in summaries.values()]
        )

        np.testing.assert_array_equal(values, np.zeros(len(summaries)))


class SimulationTests(unittest.TestCase):
    def test_simulation_is_seeded_and_population_schema_compatible(self) -> None:
        model = StoryCompetitionModel()

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

        validated = validate_story_data(first, n_days=12)
        np.testing.assert_array_equal(
            first["mentions"],
            second["mentions"],
        )
        self.assertEqual(validated["mentions"].shape[1], 12)
        self.assertEqual(validated["mentions"].dtype, np.float32)

    def test_births_precede_reports_and_daily_reports_are_conserved(
        self,
    ) -> None:
        class FakeRng:
            def __init__(self) -> None:
                self.report_counts = iter([4, 5])
                self.normal_call: tuple[float, float, int] | None = None

            def poisson(self, lam, size=None):
                if size is not None:
                    self.birth_rate = lam
                    return np.asarray([0, 2, 1])
                return next(self.report_counts)

            def normal(self, *, loc, scale, size):
                self.normal_call = (loc, scale, size)
                return np.asarray([0.0, 1.0, -1.0])

            def multinomial(self, n, pvals):
                self.probabilities = np.asarray(pvals)
                result = np.zeros(len(pvals), dtype=np.int64)
                result[0] = n
                return result

        rng = FakeRng()
        result = StoryCompetitionModel().simulate(
            PARAMETERS,
            rng,
            n_days=3,
        )["mentions"]

        self.assertEqual(rng.birth_rate, 2.0)
        self.assertEqual(rng.normal_call, (0.0, 1.0, 3))
        self.assertEqual(result.shape, (3, 3))
        np.testing.assert_array_equal(result.sum(axis=0), [0, 4, 5])
        self.assertEqual(result[2, 1], 0)
        self.assertAlmostEqual(rng.probabilities.sum(), 1.0)

    def test_invalid_context_and_parameters_are_rejected(self) -> None:
        model = StoryCompetitionModel()

        with self.assertRaisesRegex(ValueError, "require n_days"):
            model.simulate(PARAMETERS, np.random.default_rng(1))
        with self.assertRaisesRegex(ValueError, "positive"):
            model.simulate(
                {**PARAMETERS, "story_rate": np.asarray(0.0)},
                np.random.default_rng(1),
                n_days=3,
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            model.simulate(
                {**PARAMETERS, "beta_appeal": np.asarray(-1.0)},
                np.random.default_rng(1),
                n_days=3,
            )


if __name__ == "__main__":
    unittest.main()
