from __future__ import annotations

import sys
from types import ModuleType
import unittest

import matplotlib.pyplot as plt
import numpy as np


sys.modules.setdefault("bayesflow", ModuleType("bayesflow"))
sys.modules.setdefault("keras", ModuleType("keras"))

from slides.make_prior_posterior_ternary_frames import (
    log_ratios_to_mixture,
    make_figure,
    mixture_to_log_ratios,
    observed_conditions,
)
from slides.make_summary_pairplot_gif import (
    DATA_PATH,
    observed_summary_from_csv,
    simulate_mixture_summary,
)


class IPDSimulationTests(unittest.TestCase):
    def test_pure_cooperation_has_full_cooperation_and_no_inequality(
        self,
    ) -> None:
        summary = simulate_mixture_summary(
            np.asarray([0.0, 0.0, 1.0]),
            np.random.default_rng(4),
        )
        np.testing.assert_allclose(summary, [1.0, 0.0])

    def test_rejects_invalid_mixture(self) -> None:
        with self.assertRaisesRegex(ValueError, "probability vector"):
            simulate_mixture_summary(
                np.asarray([0.5, 0.5, 0.5]),
                np.random.default_rng(4),
            )


class TernaryPosteriorTests(unittest.TestCase):
    def test_log_ratio_round_trip_stays_on_simplex(self) -> None:
        mixtures = np.random.default_rng(8).dirichlet(
            np.ones(3),
            size=20,
        )
        restored = log_ratios_to_mixture(
            mixture_to_log_ratios(mixtures)
        )

        np.testing.assert_allclose(restored, mixtures)
        np.testing.assert_allclose(restored.sum(axis=1), 1.0)
        self.assertTrue(np.all(restored > 0.0))

    def test_empirical_summary_builds_one_condition_batch(self) -> None:
        observed = observed_summary_from_csv(DATA_PATH)
        conditions = observed_conditions(observed)

        self.assertEqual(set(conditions), {"Cooperation", "Inequality"})
        for values in conditions.values():
            self.assertEqual(values.shape, (1, 1))
            self.assertEqual(values.dtype, np.float32)

    def test_complete_figure_contains_two_ternaries(self) -> None:
        rng = np.random.default_rng(12)
        prior = rng.dirichlet(np.ones(3), size=80)
        posterior = rng.dirichlet(np.asarray([4.0, 2.0, 5.0]), size=100)
        figure = make_figure(
            prior,
            posterior,
            np.asarray([0.5, 0.1]),
            show_posterior=True,
        )
        try:
            self.assertEqual(len(figure.axes), 2)
            self.assertEqual(
                [axis.texts[0].get_text() for axis in figure.axes],
                ["Prior", "Posterior"],
            )
        finally:
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
