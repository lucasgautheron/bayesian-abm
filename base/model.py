"""PyMC priors and likelihood-free contact models for BayesFlow."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Union

import bayesflow as bf
import numpy as np
import pymc as pm
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from base.summaries import Summaries


INTERVAL_SECONDS = 20
CONTACT_KEYS = ("t", "i", "j")

ContactData = dict[str, NDArray[np.int32]]
ParameterData = dict[str, NDArray[Any]]
Seed = Union[
    int,
    np.integer,
    np.random.SeedSequence,
    np.random.Generator,
    None,
]


def validate_contacts(contacts: Mapping[str, ArrayLike]) -> ContactData:
    """Return contacts after checking the Parquet-compatible schema."""

    if set(contacts) != set(CONTACT_KEYS):
        raise ValueError("contacts must contain exactly 't', 'i', and 'j'")

    result = {name: np.asarray(contacts[name]) for name in CONTACT_KEYS}
    if any(values.ndim != 1 for values in result.values()):
        raise ValueError("contact columns must be one-dimensional")
    if any(values.dtype != np.int32 for values in result.values()):
        raise TypeError("contact columns must have dtype int32")
    if len({len(values) for values in result.values()}) != 1:
        raise ValueError("contact columns must have equal lengths")
    return result


def _names(variables: Sequence[Any]) -> tuple[str, ...]:
    return tuple(variable.name for variable in variables)


def _bound_value(value: Any) -> float | None:
    if value is None:
        return None
    value = value.eval() if hasattr(value, "eval") else value
    value = float(np.asarray(value))
    return None if np.isinf(value) else value


class Model(ABC):
    """A PyMC prior paired with a likelihood-free contact simulator.

    Every free variable is an inference target by default. Set
    ``inference_variables`` to a subset when the prior also contains nuisance
    variables needed only by the simulator.
    """

    name: ClassVar[str] = ""
    inference_variables: ClassVar[Sequence[str] | None] = None

    @abstractmethod
    def build_prior(self, **context: Any) -> pm.Model:
        """Build and return a prior-only PyMC model."""

    @abstractmethod
    def simulate(
        self,
        parameters: Mapping[str, NDArray[Any]],
        rng: np.random.Generator,
        **context: Any,
    ) -> Mapping[str, ArrayLike]:
        """Simulate contact records without evaluating a likelihood."""

    def _prior(
        self,
        context: Mapping[str, Any],
    ) -> tuple[pm.Model, tuple[str, ...], tuple[str, ...]]:
        prior = self.build_prior(**context)
        free_names = _names(prior.free_RVs)
        inference_names = (
            free_names
            if self.inference_variables is None
            else tuple(self.inference_variables)
        )
        unknown = set(inference_names).difference(free_names)
        if unknown:
            raise ValueError(
                f"inference variables are not in the PyMC prior: {unknown}"
            )
        simulator_names = free_names + _names(prior.deterministics)
        return prior, inference_names, simulator_names

    @staticmethod
    def _draw_prior(
        prior: pm.Model,
        names: Sequence[str],
        draws: int,
        rng: np.random.Generator,
    ) -> dict[str, NDArray[Any]]:
        return {
            name: np.asarray(values)
            for name, values in pm.sample_prior_predictive(
                draws=draws,
                model=prior,
                var_names=names,
                random_seed=rng,
                return_inferencedata=False,
            ).items()
        }

    def sample(
        self,
        *,
        seed: Seed = None,
        **context: Any,
    ) -> tuple[ParameterData, ContactData]:
        """Draw parameters and one native contact simulation."""

        rng = np.random.default_rng(seed)
        prior, inference_names, simulator_names = self._prior(context)
        draws = self._draw_prior(prior, simulator_names, 1, rng)
        parameters = {name: values[0] for name, values in draws.items()}
        contacts = validate_contacts(self.simulate(parameters, rng, **context))
        inferred = {name: parameters[name] for name in inference_names}
        return inferred, contacts

    def as_bayesflow_simulator(
        self,
        summaries: Summaries,
        *,
        seed: Seed = None,
        include_parameters: bool = True,
        **fixed_context: Any,
    ) -> Callable[..., dict[str, NDArray[Any]]]:
        """Return an unbatched simulator accepted by ``bf.make_simulator``."""

        from base.summaries import compute_summaries

        rng = np.random.default_rng(seed)

        def simulator(**context: Any) -> dict[str, NDArray[Any]]:
            parameters, contacts = self.sample(
                seed=rng,
                **{**fixed_context, **context},
            )
            result = compute_summaries(contacts, summaries)
            return {**parameters, **result} if include_parameters else result

        return simulator

    def _batched_simulator(
        self,
        summaries: Summaries,
        *,
        seed: Seed,
        include_parameters: bool,
        progress: Callable[[int], object] | None,
        context: Mapping[str, Any],
    ) -> Callable[..., dict[str, NDArray[Any]]]:
        from base.summaries import compute_summaries

        prior, inference_names, simulator_names = self._prior(context)
        rng = np.random.default_rng(seed)

        def simulator(
            batch_shape: Sequence[int],
            **runtime_context: Any,
        ) -> dict[str, NDArray[Any]]:
            batch_shape = tuple(batch_shape)
            count = int(np.prod(batch_shape))
            draws = self._draw_prior(prior, simulator_names, count, rng)
            summary_draws = {name: [] for name in summaries}
            simulation_context = {**context, **runtime_context}

            for index in range(count):
                parameters = {
                    name: values[index] for name, values in draws.items()
                }
                contacts = validate_contacts(
                    self.simulate(parameters, rng, **simulation_context)
                )
                for name, value in compute_summaries(
                    contacts,
                    summaries,
                ).items():
                    summary_draws[name].append(value)
                if progress is not None:
                    progress(1)

            result = {
                name: np.stack(values).reshape(
                    batch_shape + values[0].shape
                )
                for name, values in summary_draws.items()
            }
            if include_parameters:
                result.update(
                    {
                        name: draws[name].reshape(
                            batch_shape + draws[name].shape[1:]
                        )
                        for name in inference_names
                    }
                )
            return result

        return simulator

    def to_bayesflow_simulator(
        self,
        summaries: Summaries,
        *,
        seed: Seed = None,
        include_parameters: bool = True,
        progress: Callable[[int], object] | None = None,
        **context: Any,
    ) -> bf.simulators.Simulator:
        """Build an efficient batched BayesFlow simulator."""

        sample_fn = self._batched_simulator(
            summaries,
            seed=seed,
            include_parameters=include_parameters,
            progress=progress,
            context=context,
        )
        return bf.simulators.LambdaSimulator(sample_fn, is_batched=True)

    def _parameter_bounds(
        self,
        prior: pm.Model,
        names: Sequence[str],
    ) -> dict[str, tuple[float | None, float | None]]:
        variables = {variable.name: variable for variable in prior.free_RVs}
        bounds = {}
        for name in names:
            variable = variables[name]
            transform = prior.rvs_to_transforms.get(variable)
            transform_name = getattr(transform, "name", None)
            if transform_name in {"log", "log_exp_m1"}:
                bounds[name] = (0.0, None)
            elif transform_name == "logodds":
                bounds[name] = (0.0, 1.0)
            elif transform_name == "interval":
                lower, upper = transform.args_fn(*variable.owner.inputs)
                bounds[name] = (_bound_value(lower), _bound_value(upper))
        return bounds

    def make_bayesflow_adapter(
        self,
        summaries: Summaries,
        **context: Any,
    ) -> bf.Adapter:
        """Build a parameter adapter using PyMC's support transforms."""

        prior, inference_names, _ = self._prior(context)
        variables = {variable.name: variable for variable in prior.free_RVs}
        scalar_names = [
            name for name in inference_names if variables[name].ndim == 0
        ]
        adapter = bf.Adapter()
        for name, (lower, upper) in self._parameter_bounds(
            prior,
            inference_names,
        ).items():
            adapter.constrain(name, lower=lower, upper=upper)

        summary_names = list(summaries)
        names = list(inference_names) + summary_names
        return (
            adapter.to_array(include=names)
            .convert_dtype("float64", "float32", include=names)
            .expand_dims(scalar_names, axis=-1)
            .concatenate(
                list(inference_names),
                into="inference_variables",
            )
            .concatenate(summary_names, into="summary_variables")
        )

    @staticmethod
    def make_bayesflow_model_comparison_adapter(
        summaries: Summaries,
    ) -> bf.Adapter:
        """Build the shared adapter used for model comparison."""

        names = list(summaries)
        return (
            bf.Adapter()
            .to_array(include=names)
            .convert_dtype("float64", "float32", include=names)
            .concatenate(names, into="inference_conditions")
        )

    @staticmethod
    def validate_collection(models: Sequence[Model]) -> tuple[Model, ...]:
        """Check that a model-comparison collection has unique names."""

        models = tuple(models)
        names = [model.name for model in models]
        if len(models) < 2:
            raise ValueError("model comparison requires at least two models")
        if len(names) != len(set(names)):
            raise ValueError("model names must be unique")
        return models


__all__ = [
    "CONTACT_KEYS",
    "INTERVAL_SECONDS",
    "ContactData",
    "Model",
    "ParameterData",
    "validate_contacts",
]
