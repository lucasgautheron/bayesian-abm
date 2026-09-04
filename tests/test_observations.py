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
    story_daily_frame_observations,
)
from models import model_registry, resolve_model
from models.stories import StoryCompetitionModel, validate_story_data


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


class StoryObservationTests(unittest.TestCase):
    def test_loads_stories_as_one_ranked_population(self) -> None:
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
        np.testing.assert_array_equal(
            observations.conditions["mentions"],
            [[[4, 0, 3], [0, 2, 1]]],
        )
        np.testing.assert_array_equal(
            observations.conditions["story_mask"],
            [[1, 1]],
        )
        np.testing.assert_array_equal(
            observations.summaries["mentions"](
                {"mentions": [[0, 2, 1], [4, 0, 3]]}
            ),
            observations.conditions["mentions"][0],
        )
        self.assertEqual(
            observations.conditions["mentions"].dtype,
            np.float32,
        )

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
            {StoryCompetitionModel.name: StoryCompetitionModel},
        )

    def test_resolves_a_model_without_a_dataset_argument(self) -> None:
        model = resolve_model("story_competition")

        self.assertIsInstance(model, StoryCompetitionModel)
        self.assertEqual(model.dataset, STORY_DAILY)


if __name__ == "__main__":
    unittest.main()
