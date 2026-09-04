from __future__ import annotations

import sys
from types import ModuleType
import unittest

import numpy as np

# Keep these data-shaping tests independent of optional inference packages.
sys.modules.setdefault("seaborn", ModuleType("seaborn"))
sys.modules.setdefault("pymc", ModuleType("pymc"))
sys.modules.setdefault("bayesflow", ModuleType("bayesflow"))

from scripts.simulate import observed_summary_statistics, summary_frame


class SummaryPairplotTests(unittest.TestCase):
    def test_summary_frame_averages_vectors_by_default(self) -> None:
        frame = summary_frame(
            {
                "scalar": np.array([1.0, 2.0]),
                "vector": np.array([[1.0, 3.0], [2.0, 6.0]]),
            },
            runs=2,
        )

        self.assertEqual(list(frame.columns), ["scalar", "vector_mean"])
        np.testing.assert_allclose(frame["vector_mean"], [2.0, 4.0])
        self.assertEqual(
            observed_summary_statistics(
                {
                    "scalar": np.array([2.0]),
                    "vector": np.array([1.0, 3.0]),
                }
            ),
            {"scalar": 2.0, "vector_mean": 2.0},
        )

    def test_summary_frame_keeps_scalars_and_expands_vectors(self) -> None:
        frame = summary_frame(
            {
                "scalar": np.array([1.0, 2.0]),
                "vector": np.array([[1.0, 3.0], [2.0, 6.0]]),
            },
            runs=2,
            vector_moments=True,
        )

        self.assertEqual(
            list(frame.columns),
            ["scalar", "vector_mean", "vector_stdev"],
        )
        np.testing.assert_allclose(frame["scalar"], [1.0, 2.0])
        np.testing.assert_allclose(frame["vector_mean"], [2.0, 4.0])
        np.testing.assert_allclose(frame["vector_stdev"], [1.0, 2.0])

    def test_observed_statistics_match_expanded_columns(self) -> None:
        statistics = observed_summary_statistics(
            {
                "scalar": np.array([2.0]),
                "vector": np.array([1.0, 3.0]),
            },
            vector_moments=True,
        )

        self.assertEqual(
            statistics,
            {
                "scalar": 2.0,
                "vector_mean": 2.0,
                "vector_stdev": 1.0,
            },
        )


if __name__ == "__main__":
    unittest.main()
