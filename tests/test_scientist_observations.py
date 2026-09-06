from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from base.observations import (
    SCIENTIST_CONVENTIONS,
    scientist_convention_frame_observations,
)
from datasets.scientist_conventions.summaries import SCIENTIST_SUMMARY_NAMES


def scientist_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scientists = pd.DataFrame(
        {
            "scientist_id": [0, 1, 2, 3],
            "favorite_convention": pd.array([1, -1, None, 1], dtype="Int8"),
            "primary_area": [0, 1, 2, 3],
            "career_start_year": [2000, 2001, 2002, 2003],
            "area_share_0": [1.0, 0.0, 0.0, 0.0],
            "area_share_1": [0.0, 1.0, 0.0, 0.0],
            "area_share_2": [0.0, 0.0, 1.0, 0.0],
            "area_share_3": [0.0, 0.0, 0.0, 1.0],
        }
    )
    coauthorship = pd.DataFrame(
        {
            "source": [0, 0, 1, 2],
            "target": [1, 2, 2, 3],
            "weight": [1.0, 0.5, 1.0, 1.0],
            "first_year": [2001, 2002, 2001, 2003],
        }
    )
    citations = pd.DataFrame(
        {
            "source": [0, 1, 3],
            "target": [1, 3, 0],
            "weight": [1.0, 0.5, 0.25],
        }
    )
    return scientists, coauthorship, citations


class ScientistObservationTests(unittest.TestCase):
    def test_loads_one_joint_network_observation(self) -> None:
        observations = scientist_convention_frame_observations(
            *scientist_frames()
        )

        self.assertEqual(observations.dataset, SCIENTIST_CONVENTIONS)
        self.assertEqual(observations.count, 1)
        self.assertEqual(set(observations.conditions), set(SCIENTIST_SUMMARY_NAMES))
        for values in observations.conditions.values():
            self.assertEqual(values.shape, (1, 1))
            self.assertEqual(values.dtype, np.float32)
            self.assertTrue(np.all(np.isfinite(values)))
        np.testing.assert_array_equal(
            observations.context["observed_mask"],
            [True, True, False, True],
        )
        np.testing.assert_array_equal(
            observations.context["first_coauthor"],
            [1, 0, 1, 2],
        )
        self.assertEqual(observations.context["start_year"], 1981)

    def test_rejects_noncanonical_coauthorship_edges(self) -> None:
        scientists, coauthorship, citations = scientist_frames()
        coauthorship.loc[0, ["source", "target"]] = [1, 0]

        with self.assertRaisesRegex(ValueError, "source < target"):
            scientist_convention_frame_observations(
                scientists,
                coauthorship,
                citations,
            )

    def test_rejects_non_dense_scientist_ids(self) -> None:
        scientists, coauthorship, citations = scientist_frames()
        scientists.loc[3, "scientist_id"] = 4

        with self.assertRaisesRegex(ValueError, "densely"):
            scientist_convention_frame_observations(
                scientists,
                coauthorship,
                citations,
            )


if __name__ == "__main__":
    unittest.main()
