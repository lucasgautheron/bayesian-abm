from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest

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

    def test_rejects_invalid_probabilities(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid"):
            model_comparison.extract_probabilities(
                [0.25, 0.25],
                ["first", "second"],
            )

    def test_rejects_duplicate_models(self) -> None:
        model_name = next(iter(model_comparison.MODEL_REGISTRY))

        with self.assertRaisesRegex(ValueError, "unique"):
            model_comparison.resolve_models([model_name, model_name])


if __name__ == "__main__":
    unittest.main()
