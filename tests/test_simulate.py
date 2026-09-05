from __future__ import annotations

import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np

# Keep these data-shaping tests independent of optional inference packages.
sys.modules.setdefault("seaborn", ModuleType("seaborn"))
sys.modules.setdefault("pymc", ModuleType("pymc"))
sys.modules.setdefault("bayesflow", ModuleType("bayesflow"))

from scripts.simulate import (
    observed_summary_statistics,
    parse_args,
    run_simulations,
    summary_frame,
)


class SummaryPairplotTests(unittest.TestCase):
    def test_summary_frame_preserves_scalar_statistics(self) -> None:
        frame = summary_frame(
            {
                "first": np.array([1.0, 2.0]),
                "second": np.array([[3.0], [6.0]]),
            },
            runs=2,
        )

        self.assertEqual(list(frame.columns), ["first", "second"])
        np.testing.assert_allclose(frame["second"], [3.0, 6.0])
        self.assertEqual(
            observed_summary_statistics(
                {
                    "first": np.array([2.0]),
                    "second": np.array([[3.0]]),
                }
            ),
            {"first": 2.0, "second": 3.0},
        )

    def test_vector_statistics_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "scalar"):
            summary_frame(
                {"vector": np.array([[1.0, 3.0], [2.0, 6.0]])},
                runs=2,
            )
        with self.assertRaisesRegex(ValueError, "scalar"):
            observed_summary_statistics(
                {"vector": np.array([1.0, 3.0])}
            )

    def test_cpus_defaults_to_one(self) -> None:
        with patch.object(sys, "argv", ["simulate.py", "latent_network"]):
            args = parse_args()

        self.assertEqual(args.cpus, 1)

    def test_rejects_non_positive_cpus(self) -> None:
        with self.assertRaisesRegex(ValueError, "cpus"):
            run_simulations("latent_network", cpus=0)


if __name__ == "__main__":
    unittest.main()
