from __future__ import annotations

import unittest

import numpy as np

from slides.make_extended_summary_pairplot_gif import (
    decision_entropy,
    extended_summary,
    individual_decision_entropy,
    observed_summary,
)


class IndividualDecisionEntropyTests(unittest.TestCase):
    def test_balanced_decisions_have_one_bit_of_entropy(self) -> None:
        entropy = decision_entropy(["C", "C", "D", "D"])
        self.assertAlmostEqual(entropy, 1.0)

    def test_constant_decisions_have_zero_entropy(self) -> None:
        entropy = decision_entropy(["C", "C", "C", "C"])
        self.assertAlmostEqual(entropy, 0.0)

    def test_individual_entropies_are_averaged_across_players(self) -> None:
        entropy = individual_decision_entropy(
            [(["C", "D"], ["C", "C"])]
        )
        self.assertAlmostEqual(entropy, 0.5)

    def test_no_players_return_zero(self) -> None:
        self.assertEqual(individual_decision_entropy([]), 0.0)

    def test_observed_extended_summary_is_finite(self) -> None:
        summary = observed_summary()

        self.assertEqual(summary.shape, (3,))
        self.assertTrue(np.all(np.isfinite(summary)))
        self.assertTrue(np.all((0.0 <= summary) & (summary <= 1.0)))

    def test_extended_summary_requires_sequences(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            extended_summary([])


if __name__ == "__main__":
    unittest.main()
