from __future__ import annotations

import sys
import unittest
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np


class FakeVariable:
    def __init__(
        self,
        name: str,
        transform: str | None = None,
        ndim: int = 0,
    ) -> None:
        self.name = name
        self.ndim = ndim
        self.owner = SimpleNamespace(inputs=())
        self.transform = SimpleNamespace(name=transform) if transform else None


class FakePrior:
    def __init__(self, n_people: int) -> None:
        self.n_people = n_people
        self.free_RVs = [
            FakeVariable("rate", "logodds"),
            FakeVariable("propensities", "log", ndim=1),
            FakeVariable("nuisance"),
        ]
        self.deterministics = [FakeVariable("private_scale")]
        self.rvs_to_transforms = {
            variable: variable.transform for variable in self.free_RVs
        }


class FakeAdapter:
    def __init__(self) -> None:
        self.operations: list[tuple[Any, ...]] = []

    def constrain(self, name, *, lower, upper):
        self.operations.append(("constrain", name, lower, upper))
        return self

    def to_array(self, *, include):
        self.operations.append(("to_array", include))
        return self

    def convert_dtype(self, from_dtype, to_dtype, *, include):
        self.operations.append(
            ("convert_dtype", from_dtype, to_dtype, include)
        )
        return self

    def expand_dims(self, names, *, axis):
        self.operations.append(("expand_dims", names, axis))
        return self

    def concatenate(self, names, *, into):
        self.operations.append(("concatenate", names, into))
        return self


class FakeLambdaSimulator:
    def __init__(self, sample_fn, *, is_batched):
        self.sample_fn = sample_fn
        self.is_batched = is_batched

    def sample(self, batch_shape):
        return self.sample_fn(batch_shape)


fake_pm = ModuleType("pymc")


def sample_prior_predictive(
    *, draws, model, var_names, random_seed, return_inferencedata
):
    rate = random_seed.uniform(0.1, 0.9, size=draws)
    available = {
        "rate": rate,
        "propensities": random_seed.lognormal(
            np.log(rate)[:, None],
            0.1,
            size=(draws, model.n_people),
        ),
        "nuisance": random_seed.normal(size=draws),
        "private_scale": rate * 2,
    }
    return {name: available[name] for name in var_names}


fake_pm.sample_prior_predictive = sample_prior_predictive
fake_bf = ModuleType("bayesflow")
fake_bf.Adapter = FakeAdapter
fake_bf.simulators = SimpleNamespace(
    Simulator=object,
    LambdaSimulator=FakeLambdaSimulator,
)
sys.modules["pymc"] = fake_pm
sys.modules["bayesflow"] = fake_bf

from base.abm import Model, compute_summaries, validate_contacts


SUMMARIES = {
    "contact_count": lambda contacts: len(contacts["t"]),
    "people": lambda contacts: np.array(
        [np.unique(contacts["i"]).size, np.unique(contacts["j"]).size]
    ),
}


class ToyModel(Model):
    name = "toy"
    inference_variables = ("rate", "propensities")

    def build_prior(self, **context):
        return FakePrior(context["n_people"])

    def simulate(self, parameters, rng, **context):
        assert "nuisance" in parameters
        count = int(rng.integers(4, 8))
        first = np.arange(count, dtype=np.int32) % context["n_people"]
        return {
            "t": ((np.arange(count, dtype=np.int32) // 2 + 1) * 20),
            "i": first,
            "j": (first + 1) % context["n_people"],
        }


class OtherModel(ToyModel):
    name = "other"


class ModelTests(unittest.TestCase):
    def test_sample_is_reproducible(self):
        first = ToyModel().sample(seed=4, n_people=3)
        second = ToyModel().sample(seed=4, n_people=3)
        for left, right in zip(first, second):
            for name in left:
                np.testing.assert_array_equal(left[name], right[name])

    def test_default_targets_all_free_variables(self):
        class DefaultModel(ToyModel):
            inference_variables = None

        parameters, _ = DefaultModel().sample(seed=1, n_people=3)
        self.assertEqual(
            set(parameters), {"rate", "propensities", "nuisance"}
        )

    def test_bayesflow_callable(self):
        simulator = ToyModel().as_bayesflow_simulator(
            SUMMARIES, seed=1, n_people=3
        )
        result = simulator()
        self.assertEqual(
            set(result),
            {"rate", "propensities", "contact_count", "people"},
        )

    def test_batched_bayesflow_simulator(self):
        simulator = ToyModel().to_bayesflow_simulator(
            SUMMARIES, seed=1, n_people=3
        )
        result = simulator.sample((5,))
        self.assertEqual(result["rate"].shape, (5,))
        self.assertEqual(result["propensities"].shape, (5, 3))
        self.assertEqual(result["people"].shape, (5, 2))

    def test_adapter_uses_pymc_transforms(self):
        adapter = ToyModel().make_bayesflow_adapter(
            SUMMARIES, n_people=3
        )
        self.assertIn(
            ("constrain", "rate", 0.0, 1.0), adapter.operations
        )
        self.assertIn(
            ("constrain", "propensities", 0.0, None),
            adapter.operations,
        )
        self.assertIn(
            ("expand_dims", ["rate"], -1),
            adapter.operations,
        )

    def test_model_comparison_adapter_uses_summaries_as_conditions(self):
        adapter = Model.make_bayesflow_model_comparison_adapter(SUMMARIES)

        self.assertIn(
            (
                "concatenate",
                ["contact_count", "people"],
                "inference_conditions",
            ),
            adapter.operations,
        )

    def test_model_collection(self):
        self.assertEqual(
            len(Model.validate_collection([ToyModel(), OtherModel()])), 2
        )
        with self.assertRaises(ValueError):
            Model.validate_collection([ToyModel(), ToyModel()])


class DataTests(unittest.TestCase):
    def test_contacts_and_summaries(self):
        contacts = {
            "t": np.array([20, 40], dtype=np.int32),
            "i": np.array([0, 1], dtype=np.int32),
            "j": np.array([1, 2], dtype=np.int32),
        }
        validated = validate_contacts(contacts)
        summaries = compute_summaries(validated, SUMMARIES)
        self.assertEqual(summaries["contact_count"], 2)
        self.assertEqual(summaries["contact_count"].shape, (1,))
        np.testing.assert_array_equal(summaries["people"], [2, 2])

    def test_wrong_contact_dtype(self):
        contacts = {
            "t": np.array([20]),
            "i": np.array([0], dtype=np.int32),
            "j": np.array([1], dtype=np.int32),
        }
        with self.assertRaises(TypeError):
            validate_contacts(contacts)


if __name__ == "__main__":
    unittest.main()
