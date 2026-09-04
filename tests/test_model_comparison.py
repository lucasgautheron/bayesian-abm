from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


sys.modules.setdefault("bayesflow", ModuleType("bayesflow"))
sys.modules.setdefault("pymc", ModuleType("pymc"))

SCRIPT = Path(__file__).parents[1] / "scripts" / "model-comparison.py"
SPEC = importlib.util.spec_from_file_location("model_comparison", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
model_comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(model_comparison)


class ModelComparisonTests(unittest.TestCase):
    def test_extracts_single_observed_probability_vector(self) -> None:
        probabilities = model_comparison.extract_probabilities(
            [[0.25, 0.75]],
            ["first", "second"],
        )

        np.testing.assert_allclose(probabilities, [0.25, 0.75])

    def test_averages_probabilities_across_observed_series(self) -> None:
        probabilities = model_comparison.extract_probabilities(
            [[0.25, 0.75], [0.75, 0.25]],
            ["first", "second"],
        )

        np.testing.assert_allclose(probabilities, [0.5, 0.5])

    def test_rejects_invalid_probabilities(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid"):
            model_comparison.extract_probabilities(
                [0.25, 0.25],
                ["first", "second"],
            )

    def test_rejects_duplicate_models(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            model_comparison.resolve_models(
                ["reputation_conversation", "reputation_conversation"]
            )

    def test_rejects_models_from_different_datasets(self) -> None:
        with self.assertRaisesRegex(ValueError, "same dataset"):
            model_comparison.resolve_models(
                ["reputation_conversation", "story_competition"]
            )

    def test_predicts_observations_in_batches(self) -> None:
        class Approximator:
            def __init__(self) -> None:
                self.batch_sizes: list[int] = []

            def predict(self, *, conditions, probs):
                self.assert_probs = probs
                count = len(conditions["mentions"])
                self.batch_sizes.append(count)
                return np.tile([0.25, 0.75], (count, 1))

        approximator = Approximator()
        prediction = model_comparison.predict_observations(
            approximator,
            {"mentions": np.arange(10).reshape(5, 2)},
            batch_size=2,
        )

        self.assertEqual(approximator.batch_sizes, [2, 2, 1])
        self.assertEqual(prediction.shape, (5, 2))

    def test_uses_direct_scalar_conditions_for_all_models(self) -> None:
        class ScalarModel:
            dataset = "story_daily"

            def __init__(self, name):
                self.name = name

            def to_bayesflow_simulator(self, summaries, **kwargs):
                return (self.name, summaries, kwargs)

            def make_bayesflow_model_comparison_adapter(self, summaries):
                self.adapter_summaries = summaries
                return "scalar_adapter"

        class ComparisonSimulator:
            def __init__(self, **kwargs):
                self.arguments = kwargs

        fake_bf = SimpleNamespace(
            simulators=SimpleNamespace(
                ModelComparisonSimulator=ComparisonSimulator,
            ),
            approximators=SimpleNamespace(
                ModelComparisonApproximator=lambda **kwargs: kwargs,
            ),
            networks=SimpleNamespace(
                MLP=lambda **kwargs: ("mlp", kwargs),
            ),
        )
        models = [ScalarModel("first"), ScalarModel("second")]
        summaries = {"first_scalar": object(), "second_scalar": object()}

        with patch.object(model_comparison, "bf", fake_bf):
            approximator, _ = model_comparison.make_model_comparison(
                models,
                summaries,
                seed=3,
                context={"n_days": 4},
            )

        self.assertNotIn("summary_network", approximator)
        self.assertEqual(approximator["adapter"], "scalar_adapter")
        self.assertEqual(
            approximator["standardize"],
            "inference_conditions",
        )


if __name__ == "__main__":
    unittest.main()
