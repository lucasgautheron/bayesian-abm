from __future__ import annotations

import unittest

import numpy as np

from datasets.scientist_conventions.schema import validate_preference_data
from datasets.scientist_conventions.summaries import (
    SCIENTIST_SUMMARY_NAMES,
    fit_ising_parameters,
    make_scientist_summaries,
)
from models.scientist_conventions import GlobalTransmissionModel
from tests.scientist_model_helpers import scientist_context


class ScientistConventionBaseTests(unittest.TestCase):
    def test_validates_binary_int8_preferences(self) -> None:
        preference = np.asarray([1, -1, 1, -1], dtype=np.int8)
        validated = validate_preference_data(
            {"preference": preference},
            n_scientists=4,
        )
        np.testing.assert_array_equal(validated["preference"], preference)

        with self.assertRaisesRegex(TypeError, "int8"):
            validate_preference_data(
                {"preference": preference.astype(np.int64)},
                n_scientists=4,
            )
        with self.assertRaisesRegex(ValueError, "-1 or \\+1"):
            validate_preference_data(
                {"preference": np.asarray([1, 0, 1, -1], dtype=np.int8)},
                n_scientists=4,
            )

    def test_fits_six_finite_regularized_ising_parameters(self) -> None:
        context = scientist_context()
        fitted = fit_ising_parameters(
            np.ones(4, dtype=np.int8),
            primary_area=context["primary_area"],
            observed_mask=context["observed_mask"],
            coauthorship=context["coauthorship"],
            citations=context["citations"],
        )

        self.assertEqual(fitted.shape, (6,))
        self.assertTrue(np.all(np.isfinite(fitted)))
        self.assertGreater(fitted[4], 0.0)
        self.assertGreater(fitted[5], 0.0)

    def test_builds_scalar_summary_registry(self) -> None:
        summaries = make_scientist_summaries(**scientist_context())
        self.assertEqual(tuple(summaries), SCIENTIST_SUMMARY_NAMES)
        data = {"preference": np.asarray([1, -1, 1, -1], dtype=np.int8)}
        values = np.asarray([summary(data) for summary in summaries.values()])
        self.assertEqual(values.shape, (6,))
        self.assertTrue(np.all(np.isfinite(values)))

    def test_builds_an_ordered_scientist_summary_subset(self) -> None:
        names = ("citation_coupling", "field_theory_hep")
        context = scientist_context()

        summaries = make_scientist_summaries(
            **context,
            summary_names=names,
        )
        values = GlobalTransmissionModel().summarize(
            {"preference": np.asarray([1, -1, 1, -1], dtype=np.int8)},
            summaries,
            **context,
        )

        self.assertEqual(tuple(summaries), names)
        self.assertEqual(tuple(values), names)
        with self.assertRaisesRegex(ValueError, "at least one"):
            make_scientist_summaries(
                **scientist_context(),
                summary_names=(),
            )
        with self.assertRaisesRegex(ValueError, "unknown"):
            make_scientist_summaries(
                **scientist_context(),
                summary_names=("not_registered",),
            )

    def test_ignores_unobserved_spins(self) -> None:
        context = scientist_context()
        context["observed_mask"] = np.asarray(
            [True, True, True, False],
            dtype=np.bool_,
        )
        first = fit_ising_parameters(
            [1, 1, -1, 1],
            primary_area=context["primary_area"],
            observed_mask=context["observed_mask"],
            coauthorship=context["coauthorship"],
            citations=context["citations"],
        )
        second = fit_ising_parameters(
            [1, 1, -1, -1],
            primary_area=context["primary_area"],
            observed_mask=context["observed_mask"],
            coauthorship=context["coauthorship"],
            citations=context["citations"],
        )
        np.testing.assert_allclose(first, second)


if __name__ == "__main__":
    unittest.main()
