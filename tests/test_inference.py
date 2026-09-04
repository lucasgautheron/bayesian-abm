from __future__ import annotations

import sys
from types import ModuleType
import unittest

import numpy as np


sys.modules.setdefault("bayesflow", ModuleType("bayesflow"))
sys.modules.setdefault("pymc", ModuleType("pymc"))

from scripts.inference import prepare_posterior_plot_data


class PosteriorPlotDataTests(unittest.TestCase):
    def test_restores_scalar_axis_and_averages_vector_parameters(self) -> None:
        prepared, names = prepare_posterior_plot_data(
            {
                "scalar": np.zeros((1, 10)),
                "vector": np.broadcast_to(
                    np.asarray([1.0, 3.0]),
                    (1, 10, 2),
                ),
            },
            ["scalar", "vector"],
        )

        self.assertEqual(prepared["scalar"].shape, (1, 10, 1))
        self.assertEqual(prepared["vector"].shape, (1, 10, 1))
        np.testing.assert_array_equal(prepared["vector"], 2.0)
        self.assertEqual(names, ["scalar", "vector_mean"])

    def test_rejects_missing_variable(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing"):
            prepare_posterior_plot_data({}, ["rate"])


if __name__ == "__main__":
    unittest.main()
