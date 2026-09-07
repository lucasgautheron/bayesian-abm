"""Human-readable model metadata and Markdown workshop reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import inspect
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from base.model import Model


@dataclass(frozen=True)
class PriorSummary:
    """Natural-scale summary of one inferred model parameter."""

    name: str
    prior: str
    mean: float
    sigma: float
    unit: str
    shape: tuple[int, ...]


MomentFunction = Callable[[tuple[float, ...]], tuple[float, float]]


def _normal(parameters: tuple[float, ...]) -> tuple[float, float]:
    mu, sigma = parameters
    return mu, sigma


def _half_normal(parameters: tuple[float, ...]) -> tuple[float, float]:
    location, scale = parameters
    return (
        location + scale * math.sqrt(2.0 / math.pi),
        scale * math.sqrt(1.0 - 2.0 / math.pi),
    )


def _log_normal(parameters: tuple[float, ...]) -> tuple[float, float]:
    mu, sigma = parameters
    variance = math.expm1(sigma**2) * math.exp(2.0 * mu + sigma**2)
    return math.exp(mu + sigma**2 / 2.0), math.sqrt(variance)


def _exponential(parameters: tuple[float, ...]) -> tuple[float, float]:
    (scale,) = parameters
    return scale, scale


def _beta(parameters: tuple[float, ...]) -> tuple[float, float]:
    alpha, beta = parameters
    total = alpha + beta
    variance = alpha * beta / (total**2 * (total + 1.0))
    return alpha / total, math.sqrt(variance)


def _uniform(parameters: tuple[float, ...]) -> tuple[float, float]:
    lower, upper = parameters
    return (lower + upper) / 2.0, (upper - lower) / math.sqrt(12.0)


def _pareto(parameters: tuple[float, ...]) -> tuple[float, float]:
    alpha, minimum = parameters
    mean = math.inf if alpha <= 1.0 else alpha * minimum / (alpha - 1.0)
    sigma = (
        math.inf
        if alpha <= 2.0
        else minimum
        * math.sqrt(alpha / ((alpha - 1.0) ** 2 * (alpha - 2.0)))
    )
    return mean, sigma


_DISTRIBUTIONS: dict[
    str,
    tuple[str, tuple[str, ...], MomentFunction],
] = {
    "beta": ("Beta", ("alpha", "beta"), _beta),
    "exponential": ("Exponential", ("scale",), _exponential),
    "halfnormal": ("HalfNormal", ("loc", "sigma"), _half_normal),
    "lognormal": ("LogNormal", ("mu", "sigma"), _log_normal),
    "normal": ("Normal", ("mu", "sigma"), _normal),
    "pareto": ("Pareto", ("alpha", "m"), _pareto),
    "uniform": ("Uniform", ("lower", "upper"), _uniform),
}


def _scalar(value: Any) -> float:
    evaluated = value.eval() if hasattr(value, "eval") else value
    array = np.asarray(evaluated, dtype=np.float64)
    if array.size != 1:
        raise ValueError("prior hyperparameters must be scalar")
    return float(array.reshape(-1)[0])


def _format_number(value: float) -> str:
    if math.isnan(value):
        return "undefined"
    if math.isinf(value):
        return "∞" if value > 0 else "−∞"
    return f"{value:.3g}"


def _distribution_summary(variable: Any) -> tuple[str, float, float]:
    owner = variable.owner
    if owner is None:
        raise ValueError(f"prior variable {variable.name!r} has no owner")
    operation = owner.op
    operation_name = getattr(operation, "name", "")
    try:
        display_name, parameter_names, moments = _DISTRIBUTIONS[operation_name]
    except KeyError as exc:
        raise ValueError(
            f"unsupported prior distribution {operation_name!r} "
            f"for parameter {variable.name!r}"
        ) from exc
    count = len(getattr(operation, "ndims_params", parameter_names))
    inputs = owner.inputs[-count:] if count else ()
    parameters = tuple(_scalar(value) for value in inputs)
    if len(parameters) != len(parameter_names):
        raise ValueError(
            f"unexpected prior parameterization for {variable.name!r}"
        )
    displayed_names = parameter_names
    displayed_parameters = parameters
    if operation_name == "halfnormal" and parameters[0] == 0.0:
        displayed_names = ("sigma",)
        displayed_parameters = parameters[1:]
    elif operation_name == "exponential":
        displayed_names = ("lam",)
        displayed_parameters = (1.0 / parameters[0],)
    arguments = ", ".join(
        f"{name}={_format_number(value)}"
        for name, value in zip(displayed_names, displayed_parameters)
    )
    mean, sigma = moments(parameters)
    return f"{display_name}({arguments})", mean, sigma


def summarize_priors(
    model: Model,
    context: Mapping[str, Any],
) -> tuple[PriorSummary, ...]:
    """Describe the priors of the variables inferred by BayesFlow."""

    prior = model.build_prior(**context)
    variables = {variable.name: variable for variable in prior.free_RVs}
    names = tuple(model.inference_variables or variables)
    unknown = set(names).difference(variables)
    if unknown:
        raise ValueError(f"inference variables are not in the prior: {unknown}")
    shapes = prior.eval_rv_shapes()
    summaries: list[PriorSummary] = []
    for name in names:
        prior_text, mean, sigma = _distribution_summary(variables[name])
        summaries.append(
            PriorSummary(
                name=name,
                prior=prior_text,
                mean=mean,
                sigma=sigma,
                unit=model.parameter_units.get(name, "—"),
                shape=tuple(int(size) for size in shapes[name]),
            )
        )
    return tuple(summaries)


def model_description(model: Model) -> str:
    """Return the model class documentation as one readable paragraph."""

    description = inspect.getdoc(type(model))
    if not description:
        raise ValueError(f"model {model.name!r} needs a class docstring")
    return " ".join(description.split())


def _escape_table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _parameter_label(prior: PriorSummary) -> str:
    label = f"`{prior.name}`"
    if prior.shape:
        dimensions = " × ".join(str(size) for size in prior.shape)
        label += f" ({dimensions})"
    return label


def _figure_or_note(
    *,
    present: bool,
    alt: str,
    path: str,
    disabled_note: str,
) -> str:
    if present:
        return f"![{alt}]({path})"
    return f"*{disabled_note}*"


def render_report_markdown(
    *,
    model: Model,
    summary_names: Sequence[str],
    priors: Sequence[PriorSummary],
    plot_names: Sequence[str],
) -> str:
    """Render a complete model report using workshop terminology."""

    available = set(plot_names)
    summaries = ", ".join(f"`{name}`" for name in summary_names)
    rows = [
        "| Parameter | Prior | Mean | Sigma | Unit |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    rows.extend(
        "| {} | `{}` | {} | {} | {} |".format(
            _parameter_label(prior),
            _escape_table_cell(prior.prior),
            _format_number(prior.mean),
            _format_number(prior.sigma),
            _escape_table_cell(prior.unit),
        )
        for prior in priors
    )

    diagnostic_names = [
        name
        for name in plot_names
        if name not in {"simulations", "posterior", "posterior_predictive"}
    ]
    diagnostics = (
        "\n\n".join(
            "### {}\n\n![{} diagnostic](diagnostics/{}.png)".format(
                name.replace("_", " ").title(),
                name.replace("_", " "),
                name,
            )
            for name in diagnostic_names
        )
        if diagnostic_names
        else "*Inference diagnostics were not generated because the diagnostic "
        "stage was disabled.*"
    )

    sections = [
        f"# Model report: {model.name.replace('_', ' ').title()}",
        "## Model description\n\n"
        f"{model_description(model)} This probabilistic program maps "
        "micro-level behavioral assumptions and parameter values to a "
        "synthetic outcome with the same basic structure as the observed data.",
        "## Parameters and prior distributions\n\n"
        "Parameters describe individual or population traits, strategies, "
        "environmental features, or latent social structure. Their priors are "
        "sampled anew for each simulation; the mean and sigma below are the "
        "implied moments on the parameter's natural scale.\n\n"
        + "\n".join(rows),
        "## Prior-predictive summary statistics\n\n"
        "The pairplot shows the joint distribution of the selected summary "
        f"statistics under repeated prior simulations ({summaries}), with the "
        "observed summaries overlaid. Marginal or pairwise incompatibility "
        "indicates model misspecification.\n\n"
        + _figure_or_note(
            present="simulations" in available,
            alt="Prior-predictive summary-statistic pairplot",
            path="simulations.png",
            disabled_note="The prior-predictive pairplot was not generated.",
        ),
        "## Prior and posterior parameter distributions\n\n"
        "Simulation-based inference uses the observed summary statistics "
        "$S$ to update the prior $P(θ)$ to the posterior $P(θ \\mid S)$. "
        "Comparing the two distributions shows the direction and extent of "
        "parameter learning supplied by the data.\n\n"
        + _figure_or_note(
            present="posterior" in available,
            alt="Prior and posterior parameter distributions",
            path="posterior.png",
            disabled_note="The prior/posterior plot was not generated.",
        ),
        "## Posterior-predictive summary statistics\n\n"
        "Posterior parameter draws are propagated through the simulator and "
        "summarization pipeline. Comparing their marginal and pairwise "
        "distributions with the observed summaries tests whether the inferred "
        "model reproduces the macro-level patterns it was intended to explain."
        "\n\n"
        + _figure_or_note(
            present="posterior_predictive" in available,
            alt="Posterior-predictive summary-statistic pairplot",
            path="posterior_predictive.png",
            disabled_note=(
                "The posterior-predictive pairplot was not generated because "
                "the predictive stage was disabled."
            ),
        ),
        "## Inference and identification diagnostics\n\n"
        "These diagnostics assess whether simulation-based inference is "
        "reliable and whether the parameters are properly identified. They "
        "help distinguish incompatibility with the data (misspecification) "
        "from compatibility without unique parameter recovery "
        "(missidentification), which should be resolved before substantive "
        "interpretation.\n\n"
        + diagnostics,
    ]
    return "\n\n".join(sections) + "\n"


def write_report_markdown(
    path: Path,
    *,
    model: Model,
    summary_names: Sequence[str],
    priors: Sequence[PriorSummary],
    plot_names: Sequence[str],
) -> None:
    """Write a complete Markdown model report."""

    path.write_text(
        render_report_markdown(
            model=model,
            summary_names=summary_names,
            priors=priors,
            plot_names=plot_names,
        ),
        encoding="utf-8",
    )


__all__ = [
    "PriorSummary",
    "model_description",
    "render_report_markdown",
    "summarize_priors",
    "write_report_markdown",
]
