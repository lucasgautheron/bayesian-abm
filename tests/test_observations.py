from __future__ import annotations

from datetime import date, timedelta
import sys
from types import ModuleType
import unittest

import numpy as np
import pandas as pd

sys.modules.setdefault("bayesflow", ModuleType("bayesflow"))
sys.modules.setdefault("pymc", ModuleType("pymc"))

from base.observations import (
    STORY_DAILY,
    condition_batches,
    story_daily_frame_observations,
)
from models import model_registry, resolve_model
from models.stories import (
    GeneralizedSIRModel,
    LatentRateModel,
    LimitedAttentionModel,
    LinearInfluenceModel,
    StoryCompetitionModel,
)
from datasets.story_daily.schema import validate_story_data


def story_frame() -> pd.DataFrame:
    first = date(2008, 8, 1)
    dates = [first + timedelta(days=offset) for offset in range(3)]
    return pd.DataFrame(
        {
            "story_id": [10] * 3 + [20] * 3,
            "date": dates + dates,
            "mentions": [0, 2, 1, 4, 0, 3],
        }
    )


class ConditionBatchTests(unittest.TestCase):
    def test_batches_aligned_conditions(self) -> None:
        batches = list(
            condition_batches(
                {
                    "first": np.arange(5),
                    "second": np.arange(10).reshape(5, 2),
                },
                2,
            )
        )

        self.assertEqual([len(batch["first"]) for batch in batches], [2, 2, 1])

    def test_rejects_misaligned_conditions(self) -> None:
        with self.assertRaisesRegex(ValueError, "share an observation axis"):
            list(
                condition_batches(
                    {"first": np.arange(2), "second": np.arange(3)},
                    2,
                )
            )


class StoryObservationTests(unittest.TestCase):
    def test_loads_stories_as_one_scalar_summary_vector(self) -> None:
        observations = story_daily_frame_observations(
            story_frame(),
            story_count=2,
        )

        self.assertEqual(observations.dataset, STORY_DAILY)
        self.assertEqual(observations.context, {"n_days": 3})
        self.assertEqual(observations.count, 1)
        np.testing.assert_array_equal(
            observations.observation_ids,
            [STORY_DAILY],
        )
        expected = {
            "total_mentions": 10.0,
            "daily_total_stdev": np.std([4, 2, 4]),
            "mean_story_autocorrelation": np.mean(
                [
                    np.corrcoef([0, 2], [2, 1])[0, 1],
                    np.corrcoef([4, 0], [0, 3])[0, 1],
                ]
            ),
            "mean_reporting_story_count": 4.0 / 3.0,
            "story_mentions_coefficient_of_variation": 0.4,
            "mean_reporting_lifetime_days": 2.5,
            "mention_concentration": 0.58,
        }
        self.assertEqual(set(observations.conditions), set(expected))
        for name, value in expected.items():
            self.assertEqual(observations.conditions[name].shape, (1, 1))
            self.assertEqual(
                observations.conditions[name].dtype,
                np.float32,
            )
            self.assertAlmostEqual(
                float(observations.conditions[name][0, 0]),
                value,
            )

    def test_computes_only_selected_story_conditions(self) -> None:
        names = ("mention_concentration", "total_mentions")

        observations = story_daily_frame_observations(
            story_frame(),
            story_count=2,
            summary_names=names,
        )

        self.assertEqual(tuple(observations.summaries), names)
        self.assertEqual(tuple(observations.conditions), names)

    def test_rejects_different_story_date_grids(self) -> None:
        frame = story_frame()
        frame.loc[5, "date"] += timedelta(days=1)

        with self.assertRaisesRegex(ValueError, "same date grid"):
            story_daily_frame_observations(frame)

    def test_rejects_noncontiguous_story_blocks(self) -> None:
        frame = story_frame().iloc[[0, 1, 3, 4, 5, 2]].reset_index(drop=True)

        with self.assertRaisesRegex(ValueError, "contiguous"):
            story_daily_frame_observations(frame)

    def test_story_validation_uses_story_specific_context(self) -> None:
        validated = validate_story_data(
            {"mentions": [[0, 2, 1], [4, 0, 3]]},
            n_days=3,
        )
        np.testing.assert_array_equal(
            validated["mentions"],
            [[0, 2, 1], [4, 0, 3]],
        )

        with self.assertRaisesRegex(ValueError, "n_stories, n_days"):
            validate_story_data({"mentions": [1, 2]}, n_days=3)

    def test_story_and_contact_registries_are_separate(self) -> None:
        self.assertTrue(model_registry("contacts"))
        self.assertEqual(
            model_registry(STORY_DAILY),
            {
                StoryCompetitionModel.name: StoryCompetitionModel,
                LimitedAttentionModel.name: LimitedAttentionModel,
                LinearInfluenceModel.name: LinearInfluenceModel,
                GeneralizedSIRModel.name: GeneralizedSIRModel,
                LatentRateModel.name: LatentRateModel,
            },
        )

    def test_resolves_a_model_without_a_dataset_argument(self) -> None:
        model = resolve_model("story_competition")

        self.assertIsInstance(model, StoryCompetitionModel)
        self.assertEqual(model.dataset, STORY_DAILY)


if __name__ == "__main__":
    unittest.main()
