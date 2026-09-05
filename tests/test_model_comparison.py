from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


class FakeCosineDecay:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeAdamW:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


fake_keras = ModuleType("keras")
fake_keras.optimizers = SimpleNamespace(
    Optimizer=object,
    AdamW=FakeAdamW,
    schedules=SimpleNamespace(CosineDecay=FakeCosineDecay),
)
fake_keras.utils = SimpleNamespace(set_random_seed=lambda seed: None)

sys.modules.setdefault("bayesflow", ModuleType("bayesflow"))
sys.modules.setdefault("keras", fake_keras)
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

    def test_offline_training_requires_positive_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_simulations"):
            model_comparison.run_model_comparison(
                ["reputation_conversation", "latent_network"],
                num_simulations=0,
            )
        with self.assertRaisesRegex(ValueError, "epochs"):
            model_comparison.run_model_comparison(
                ["reputation_conversation", "latent_network"],
                epochs=0,
            )
        with self.assertRaisesRegex(ValueError, "batch_size"):
            model_comparison.run_model_comparison(
                ["reputation_conversation", "latent_network"],
                batch_size=0,
            )

    def test_fits_offline_from_pre_simulated_data(self) -> None:
        class Approximator:
            def __init__(self) -> None:
                self.adapter = "adapter"
                self.compile_kwargs: dict | None = None
                self.fit_kwargs: dict | None = None

            def compile(self, **kwargs):
                self.compile_kwargs = kwargs

            def fit(self, **kwargs):
                if self.compile_kwargs is None:
                    raise ValueError("You must call `compile()` before using the model.")
                self.fit_kwargs = kwargs

        class OfflineDataset:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.num_batches = 3

        fake_bf = SimpleNamespace(
            datasets=SimpleNamespace(OfflineDataset=OfflineDataset),
        )
        approximator = Approximator()

        with patch.object(model_comparison, "bf", fake_bf):
            model_comparison.fit_comparison_offline(
                approximator,
                {"x": np.zeros((10, 1))},
                epochs=3,
                batch_size=4,
            )

        dataset = approximator.fit_kwargs["dataset"]
        optimizer = approximator.compile_kwargs["optimizer"]
        self.assertIsInstance(optimizer, FakeAdamW)
        self.assertEqual(optimizer.kwargs["weight_decay"], 5.0e-3)
        self.assertEqual(optimizer.kwargs["clipnorm"], 1.5)
        schedule = optimizer.kwargs["learning_rate"]
        self.assertIsInstance(schedule, FakeCosineDecay)
        self.assertEqual(schedule.kwargs["warmup_target"], 5.0e-4)
        self.assertEqual(schedule.kwargs["decay_steps"], 9)
        self.assertIsInstance(dataset, OfflineDataset)
        self.assertEqual(dataset.kwargs["batch_size"], 4)
        self.assertEqual(dataset.kwargs["adapter"], "adapter")
        np.testing.assert_array_equal(dataset.kwargs["data"]["x"], np.zeros((10, 1)))
        self.assertEqual(approximator.fit_kwargs["epochs"], 3)
        self.assertNotIn("simulator", approximator.fit_kwargs)

    def test_model_selection_sampling_is_seeded_and_isolated(self) -> None:
        class Simulator:
            def sample(self, shape):
                return {"draws": np.random.random(shape[0])}

        np.random.seed(11)
        state = np.random.get_state()
        first = model_comparison.sample_model_comparison(
            Simulator(),
            5,
            seed=7,
        )
        after = np.random.random()

        np.random.set_state(state)
        expected_after = np.random.random()
        second = model_comparison.sample_model_comparison(
            Simulator(),
            5,
            seed=7,
        )

        np.testing.assert_array_equal(first["draws"], second["draws"])
        self.assertEqual(after, expected_after)

    def test_forwards_progress_to_each_simulator(self) -> None:
        class ProgressModel:
            dataset = "story_daily"

            def __init__(self, name):
                self.name = name
                self.progress = None

            def to_bayesflow_simulator(self, summaries, **kwargs):
                self.progress = kwargs.get("progress")
                return self.name

            def make_bayesflow_model_comparison_adapter(self, summaries):
                return "adapter"

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
        models = [ProgressModel("first"), ProgressModel("second")]
        progress = object()

        with patch.object(model_comparison, "bf", fake_bf):
            model_comparison.make_model_comparison(
                models,
                {"scalar": object()},
                seed=3,
                context={"n_days": 4},
                progress=progress,
            )

        self.assertIs(models[0].progress, progress)
        self.assertIs(models[1].progress, progress)


if __name__ == "__main__":
    unittest.main()
