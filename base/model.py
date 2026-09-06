"""PyMC priors and likelihood-free simulators for BayesFlow."""

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


INTERVAL_SECONDS = 60
CONTACT_KEYS = ("t", "i", "j")

ContactData = dict[str, NDArray[np.int32]]
SimulationData = dict[str, NDArray[Any]]
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


def _stacked_draw_count(
    parameters: Mapping[str, NDArray[Any]],
    names: Sequence[str],
) -> int:
    counts: list[int] = []
    for name in names:
        if name not in parameters:
            raise ValueError(f"missing parameter {name!r}")
        values = np.asarray(parameters[name])
        if values.ndim == 0:
            raise ValueError(f"parameter {name!r} must include a draw axis")
        counts.append(int(values.shape[0]))
    if len(set(counts)) != 1:
        raise ValueError("parameter draws must share a leading axis")
    return counts[0]


def _bound_value(value: Any) -> float | None:
    if value is None:
        return None
    value = value.eval() if hasattr(value, "eval") else value
    value = float(np.asarray(value))
    return None if np.isinf(value) else value


class Model(ABC):
    """A PyMC prior paired with a likelihood-free simulator.

    Every free variable is an inference target by default. Set
    ``inference_variables`` to a subset when the prior also contains nuisance
    variables needed only by the simulator. The default simulation contract is
    the contact schema; dataset-specific model bases may override the
    validation and summarization hooks.
    """

    name: ClassVar[str] = ""
    dataset: ClassVar[str] = "contacts"
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
        """Simulate native records without evaluating a likelihood."""

    def validate_simulation(
        self,
        simulation: Mapping[str, ArrayLike],
        **context: Any,
    ) -> SimulationData:
        """Validate and normalize one native simulation."""

        del context
        return validate_contacts(simulation)

    def summarize(
        self,
        simulation: Mapping[str, ArrayLike],
        summaries: Summaries,
        **context: Any,
    ) -> dict[str, NDArray[Any]]:
        """Validate one simulation and compute its inference conditions."""

        from datasets.contacts.summaries import compute_summaries

        return compute_summaries(
            self.validate_simulation(simulation, **context),
            summaries,
        )

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
    ) -> tuple[ParameterData, SimulationData]:
        """Draw parameters and one native simulation."""

        rng = np.random.default_rng(seed)
        prior, inference_names, simulator_names = self._prior(context)
        draws = self._draw_prior(prior, simulator_names, 1, rng)
        parameters = {name: values[0] for name, values in draws.items()}
        simulation = self.validate_simulation(
            self.simulate(parameters, rng, **context),
            **context,
        )
        inferred = {name: parameters[name] for name in inference_names}
        return inferred, simulation

    def sample_prior(
        self,
        draws: int,
        *,
        seed: Seed = None,
        **context: Any,
    ) -> ParameterData:
        """Draw inference variables from the prior without simulating data."""

        if draws < 1:
            raise ValueError("prior draws must be positive")
        prior, inference_names, _ = self._prior(context)
        return self._draw_prior(
            prior,
            inference_names,
            draws,
            np.random.default_rng(seed),
        )

    def complete_parameter_draws(
        self,
        inferred: Mapping[str, NDArray[Any]],
        rng: np.random.Generator,
        **context: Any,
    ) -> ParameterData:
        """Fill simulator-only variables for a stack of inferred draws.

        Extra free or deterministic variables are drawn with ``pymc.do`` so
        hierarchical children use the inferred hyperparameters.
        """

        prior, inference_names, simulator_names = self._prior(context)
        n_draws = _stacked_draw_count(inferred, inference_names)
        completed = {
            name: np.asarray(inferred[name]) for name in inference_names
        }
        extra = [
            name for name in simulator_names if name not in inference_names
        ]
        if not extra:
            return completed

        extras = {name: [] for name in extra}
        free_variables = {
            variable.name: variable for variable in prior.free_RVs
        }
        for index in range(n_draws):
            replacements = {
                free_variables[name]: np.asarray(completed[name])[index]
                for name in inference_names
            }
            draws = self._draw_prior(
                pm.do(prior, replacements),
                extra,
                1,
                rng,
            )
            for name in extra:
                extras[name].append(np.asarray(draws[name])[0])
        completed.update(
            {name: np.stack(values) for name, values in extras.items()}
        )
        return completed

    def simulate_summaries(
        self,
        parameters: Mapping[str, NDArray[Any]],
        summaries: Summaries,
        *,
        seed: Seed = None,
        progress: Callable[[int], object] | None = None,
        **context: Any,
    ) -> dict[str, NDArray[Any]]:
        """Simulate scalar summaries from stacked parameter draws."""

        rng = np.random.default_rng(seed)
        completed = self.complete_parameter_draws(parameters, rng, **context)
        n_draws = _stacked_draw_count(completed, tuple(completed))
        summary_draws = {name: [] for name in summaries}
        for index in range(n_draws):
            draw = {
                name: np.asarray(values)[index]
                for name, values in completed.items()
            }
            simulation = self.validate_simulation(
                self.simulate(draw, rng, **context),
                **context,
            )
            for name, value in self.summarize(
                simulation,
                summaries,
                **context,
            ).items():
                summary_draws[name].append(value)
            if progress is not None:
                progress(1)
        return {
            name: np.stack(values)
            for name, values in summary_draws.items()
        }

    def _batched_simulator(
        self,
        summaries: Summaries,
        *,
        seed: Seed,
        include_parameters: bool,
        progress: Callable[[int], object] | None,
        context: Mapping[str, Any],
    ) -> Callable[..., dict[str, NDArray[Any]]]:
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
                simulation = self.validate_simulation(
                    self.simulate(parameters, rng, **simulation_context),
                    **simulation_context,
                )
                for name, value in self.summarize(
                    simulation,
                    summaries,
                    **simulation_context,
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
            .concatenate(summary_names, into="inference_conditions")
        )

    @staticmethod
    def make_bayesflow_model_comparison_adapter(
        summaries: Summaries,
    ) -> bf.Adapter:
        """Build the shared adapter used for model comparison."""

        names = list(summaries)
        keys = [*names, "model_indices"]
        return (
            bf.Adapter()
            .to_array(include=keys)
            .convert_dtype("float64", "float32", include=keys)
            .concatenate(["model_indices"], into="inference_variables")
            .concatenate(names, into="inference_conditions")
            .keep(["inference_variables", "inference_conditions"])
        )

    @staticmethod
    def validate_collection(models: Sequence[Model]) -> tuple[Model, ...]:
        """Check that compared models are unique and share a dataset."""

        models = tuple(models)
        names = [model.name for model in models]
        if len(models) < 2:
            raise ValueError("model comparison requires at least two models")
        if len(names) != len(set(names)):
            raise ValueError("model names must be unique")
        datasets = {model.dataset for model in models}
        if len(datasets) != 1:
            raise ValueError("compared models must use the same dataset")
        return models


__all__ = [
    "CONTACT_KEYS",
    "INTERVAL_SECONDS",
    "ContactData",
    "Model",
    "ParameterData",
    "SimulationData",
    "validate_contacts",
]
