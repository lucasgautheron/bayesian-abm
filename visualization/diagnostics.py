"""Plots for simulation checks and likelihood-free inference diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import seaborn as sns


MODEL_COMPARISON_COLORS = (
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:olive",
    "tab:cyan",
)


def _padded_limits(
    *values: ArrayLike,
    fraction: float = 0.05,
) -> tuple[float, float]:
    """Return finite data limits with proportional padding."""

    combined = np.concatenate(
        [
            np.asarray(value, dtype=np.float64).reshape(-1)
            for value in values
        ]
    )
    if combined.size == 0 or not np.all(np.isfinite(combined)):
        raise ValueError("plot values must be non-empty and finite")
    lower = float(combined.min())
    upper = float(combined.max())
    padding = (upper - lower) * fraction
    if padding == 0:
        padding = max(abs(lower) * fraction, 0.5)
    return lower - padding, upper + padding


def _format_pair_axis(
    axis: Axes,
    *,
    row: int,
    column: int,
    labels: Sequence[str],
    limits: Sequence[tuple[float, float]],
    diagonal_density_label: bool,
) -> None:
    """Apply shared limits and outer labels to one pair-plot axis."""

    count = len(labels)
    axis.set_xlim(limits[column])
    if row != column:
        axis.set_ylim(limits[row])

    if row == count - 1:
        axis.set_xlabel(labels[column].replace("_", " "))
    else:
        axis.set_xlabel("")
        axis.tick_params(labelbottom=False)

    if column == 0:
        if row == column and diagonal_density_label:
            axis.set_ylabel("Density")
        elif row != column:
            axis.set_ylabel(labels[row].replace("_", " "))
        else:
            axis.set_ylabel("")
    else:
        axis.set_ylabel("")
        axis.tick_params(labelleft=False)


def prepare_posterior_plot_data(
    posterior: Mapping[str, np.ndarray],
    variable_keys: Sequence[str],
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Restore scalar axes and reduce vector parameters to their mean."""

    prepared: dict[str, np.ndarray] = {}
    variable_names: list[str] = []
    for key in variable_keys:
        if key not in posterior:
            raise ValueError(f"posterior is missing inference variable {key!r}")
        values = np.asarray(posterior[key])
        if values.ndim == 2:
            values = values[..., None]
            variable_names.append(key)
        elif values.ndim == 3:
            is_vector = values.shape[-1] > 1
            values = values.mean(axis=-1, keepdims=True)
            variable_names.append(f"{key}_mean" if is_vector else key)
        else:
            raise ValueError(
                f"posterior variable {key!r} must have dataset, draw, and "
                "optional variable axes"
            )
        prepared[key] = values
    return prepared, variable_names


def plot_prior_posterior_pairplot(
    prior: Mapping[str, np.ndarray],
    posterior: Mapping[str, np.ndarray],
    variable_keys: Sequence[str],
    *,
    dataset_id: int = 0,
) -> Figure:
    """Plot prior and posterior marginal KDEs and joint-density contours."""

    posterior_prepared, variable_names = prepare_posterior_plot_data(
        posterior,
        variable_keys,
    )
    prior_prepared, prior_names = prepare_posterior_plot_data(
        {
            key: np.asarray(values)[None, ...]
            for key, values in prior.items()
        },
        variable_keys,
    )
    if prior_names != variable_names:
        raise ValueError("prior and posterior variables have incompatible shapes")

    prior_values: dict[str, np.ndarray] = {}
    posterior_values: dict[str, np.ndarray] = {}
    for key in variable_keys:
        prior_array = np.asarray(prior_prepared[key], dtype=np.float64)
        posterior_array = np.asarray(
            posterior_prepared[key],
            dtype=np.float64,
        )
        if dataset_id < 0 or dataset_id >= posterior_array.shape[0]:
            raise IndexError(f"dataset_id {dataset_id} is out of bounds")
        prior_value = prior_array[0, :, 0]
        posterior_value = posterior_array[dataset_id, :, 0]
        if prior_value.size < 2 or posterior_value.size < 2:
            raise ValueError(
                "KDE plots require at least two prior and posterior draws"
            )
        if (
            not np.all(np.isfinite(prior_value))
            or not np.all(np.isfinite(posterior_value))
        ):
            raise ValueError("prior and posterior draws must be finite")
        prior_values[key] = prior_value
        posterior_values[key] = posterior_value

    count = len(variable_keys)
    if count == 0:
        raise ValueError("at least one inference variable is required")
    figure, axes = plt.subplots(
        count,
        count,
        figsize=(3.0 * count, 3.0 * count),
        squeeze=False,
    )
    limits = [
        _padded_limits(
            prior_values[key],
            posterior_values[key],
            fraction=0.03,
        )
        for key in variable_keys
    ]

    for row, y_key in enumerate(variable_keys):
        for column, x_key in enumerate(variable_keys):
            axis = axes[row, column]
            if row == column:
                sns.kdeplot(
                    x=prior_values[x_key],
                    ax=axis,
                    color="0.45",
                    linestyle="--",
                    linewidth=1.5,
                    warn_singular=False,
                )
                sns.kdeplot(
                    x=posterior_values[x_key],
                    ax=axis,
                    color="tab:blue",
                    fill=True,
                    alpha=0.25,
                    linewidth=1.5,
                    warn_singular=False,
                )
            else:
                sns.kdeplot(
                    x=prior_values[x_key],
                    y=prior_values[y_key],
                    ax=axis,
                    color="0.45",
                    levels=6,
                    linewidths=1.0,
                    linestyles="--",
                    thresh=0.05,
                    warn_singular=False,
                )
                sns.kdeplot(
                    x=posterior_values[x_key],
                    y=posterior_values[y_key],
                    ax=axis,
                    color="tab:blue",
                    levels=6,
                    linewidths=1.3,
                    thresh=0.05,
                    warn_singular=False,
                )
            _format_pair_axis(
                axis,
                row=row,
                column=column,
                labels=variable_names,
                limits=limits,
                diagonal_density_label=True,
            )

    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="0.45",
                linestyle="--",
                linewidth=1.5,
                label="Prior",
            ),
            Line2D(
                [0],
                [0],
                color="tab:blue",
                linewidth=1.5,
                label="Posterior",
            ),
        ],
        loc="upper right",
        frameon=False,
    )
    figure.suptitle("Prior and posterior parameter densities")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return figure


def plot_predictive_summary_pairplot(
    prior: pd.DataFrame,
    posterior: pd.DataFrame,
    observed: Mapping[str, float] | pd.DataFrame,
) -> Figure:
    """Plot prior and posterior predictive summaries against the data."""

    if prior.empty or posterior.empty or len(prior.columns) == 0:
        raise ValueError("at least one simulated summary is required")
    if list(prior.columns) != list(posterior.columns):
        raise ValueError("prior and posterior predictive summaries must match")
    if len(prior) < 2 or len(posterior) < 2:
        raise ValueError("KDE plots require at least two predictive draws")
    names = list(prior.columns)
    observed_frame = (
        pd.DataFrame([observed])
        if isinstance(observed, Mapping)
        else observed
    )
    if observed_frame.empty:
        raise ValueError("at least one observed summary is required")
    if set(observed_frame.columns) != set(names):
        raise ValueError("observed and simulated summaries must match")
    observed_frame = observed_frame[names]
    if (
        not np.all(np.isfinite(prior.to_numpy()))
        or not np.all(np.isfinite(posterior.to_numpy()))
        or not np.all(np.isfinite(observed_frame.to_numpy()))
    ):
        raise ValueError("predictive and observed summaries must be finite")

    count = len(names)
    figure, axes = plt.subplots(
        count,
        count,
        figsize=(3.2 * count, 3.2 * count),
        squeeze=False,
    )
    limits = [
        _padded_limits(
            prior[name],
            posterior[name],
            observed_frame[name],
            fraction=0.03,
        )
        for name in names
    ]
    single_observation = len(observed_frame) == 1

    for row, y_name in enumerate(names):
        for column, x_name in enumerate(names):
            axis = axes[row, column]
            if row == column:
                sns.kdeplot(
                    x=prior[x_name],
                    ax=axis,
                    color="0.45",
                    linestyle="--",
                    linewidth=1.5,
                    warn_singular=False,
                )
                sns.kdeplot(
                    x=posterior[x_name],
                    ax=axis,
                    color="tab:blue",
                    fill=True,
                    alpha=0.25,
                    linewidth=1.5,
                    warn_singular=False,
                )
                if single_observation:
                    observed_value = observed_frame[x_name].iloc[0]
                    axis.axvline(
                        observed_value,
                        color="tab:red",
                        linestyle=":",
                        linewidth=1,
                    )
                    axis.scatter(
                        observed_value,
                        0,
                        marker="*",
                        s=140,
                        color="tab:red",
                        edgecolor="white",
                        linewidth=0.7,
                        zorder=3,
                        clip_on=False,
                    )
                else:
                    sns.kdeplot(
                        x=observed_frame[x_name],
                        ax=axis,
                        color="tab:red",
                        linewidth=1.3,
                        warn_singular=False,
                    )
            else:
                sns.kdeplot(
                    x=prior[x_name],
                    y=prior[y_name],
                    ax=axis,
                    color="0.45",
                    levels=6,
                    linewidths=1.0,
                    linestyles="--",
                    thresh=0.05,
                    warn_singular=False,
                )
                sns.kdeplot(
                    x=posterior[x_name],
                    y=posterior[y_name],
                    ax=axis,
                    color="tab:blue",
                    levels=6,
                    linewidths=1.3,
                    thresh=0.05,
                    warn_singular=False,
                )
                if single_observation:
                    axis.scatter(
                        observed_frame[x_name],
                        observed_frame[y_name],
                        marker="*",
                        s=170,
                        color="tab:red",
                        edgecolor="white",
                        linewidth=0.7,
                        zorder=3,
                    )
                else:
                    axis.scatter(
                        observed_frame[x_name],
                        observed_frame[y_name],
                        s=8,
                        color="tab:red",
                        alpha=0.12,
                        linewidth=0,
                        rasterized=True,
                    )
            _format_pair_axis(
                axis,
                row=row,
                column=column,
                labels=names,
                limits=limits,
                diagonal_density_label=True,
            )

    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="0.45",
                linestyle="--",
                linewidth=1.5,
                label="Prior predictive",
            ),
            Line2D(
                [0],
                [0],
                color="tab:blue",
                linewidth=1.5,
                label="Posterior predictive",
            ),
            Line2D(
                [0],
                [0],
                color="tab:red",
                marker="*",
                linestyle="None",
                markersize=10,
                label="Observed",
            ),
        ],
        loc="upper right",
        frameon=False,
    )
    figure.suptitle(
        "Prior and posterior predictive summaries (observed data: red)"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return figure


def plot_summary_pairplot(
    frame: pd.DataFrame,
    observed: Mapping[str, float] | pd.DataFrame,
) -> Figure:
    """Plot simulated summaries against one or more observations."""

    if frame.empty or len(frame.columns) == 0:
        raise ValueError("at least one simulated summary is required")
    names = list(frame.columns)
    observed_frame = (
        pd.DataFrame([observed])
        if isinstance(observed, Mapping)
        else observed
    )
    if observed_frame.empty:
        raise ValueError("at least one observed summary is required")
    if set(observed_frame.columns) != set(names):
        raise ValueError("observed and simulated summaries must match")
    observed_frame = observed_frame[names]
    if not np.all(np.isfinite(observed_frame.to_numpy())):
        raise ValueError("observed summaries must be finite")

    count = len(names)
    figure, axes = plt.subplots(
        count,
        count,
        figsize=(3.2 * count, 3.2 * count),
        squeeze=False,
    )
    limits = [
        _padded_limits(frame[name], observed_frame[name])
        for name in names
    ]
    single_observation = len(observed_frame) == 1

    for row, y_name in enumerate(names):
        for column, x_name in enumerate(names):
            axis = axes[row, column]
            x_values = frame[x_name].to_numpy()

            if row == column:
                sns.kdeplot(
                    x=x_values,
                    ax=axis,
                    fill=True,
                    color="tab:blue",
                    alpha=0.35,
                    linewidth=1.5,
                    warn_singular=False,
                )
                if single_observation:
                    observed_value = observed_frame[x_name].iloc[0]
                    axis.axvline(
                        observed_value,
                        color="tab:red",
                        linestyle=":",
                        linewidth=1,
                    )
                    axis.scatter(
                        observed_value,
                        0,
                        marker="*",
                        s=140,
                        color="tab:red",
                        edgecolor="white",
                        linewidth=0.7,
                        zorder=3,
                        clip_on=False,
                    )
                else:
                    sns.kdeplot(
                        x=observed_frame[x_name],
                        ax=axis,
                        color="tab:red",
                        linewidth=1.3,
                        warn_singular=False,
                    )
            else:
                y_values = frame[y_name].to_numpy()
                sns.kdeplot(
                    x=x_values,
                    y=y_values,
                    ax=axis,
                    levels=7,
                    thresh=0.05,
                    cmap="Blues",
                    linewidths=1.2,
                    warn_singular=False,
                )
                if single_observation:
                    axis.scatter(
                        observed_frame[x_name],
                        observed_frame[y_name],
                        marker="*",
                        s=170,
                        color="tab:red",
                        edgecolor="white",
                        linewidth=0.7,
                        zorder=3,
                    )
                else:
                    axis.scatter(
                        observed_frame[x_name],
                        observed_frame[y_name],
                        s=8,
                        color="tab:red",
                        alpha=0.12,
                        linewidth=0,
                        rasterized=True,
                    )
            _format_pair_axis(
                axis,
                row=row,
                column=column,
                labels=names,
                limits=limits,
                diagonal_density_label=True,
            )

    figure.suptitle(
        "Prior-predictive summary statistics (observed data: red)"
    )
    figure.tight_layout()
    return figure


def plot_model_comparison_pairplot(
    predictions: Mapping[str, pd.DataFrame],
    observed: Mapping[str, float] | pd.DataFrame,
    *,
    model_names: Sequence[str] | None = None,
) -> Figure:
    """Plot each model's prior-predictive summaries against the data."""

    names = list(model_names) if model_names is not None else list(predictions)
    if len(names) < 2:
        raise ValueError("at least two models are required")
    missing = [name for name in names if name not in predictions]
    if missing:
        raise ValueError(
            "predictions are missing models: " + ", ".join(missing)
        )
    frames = [predictions[name] for name in names]
    if any(frame.empty or len(frame.columns) == 0 for frame in frames):
        raise ValueError("at least one simulated summary is required")
    summaries = list(frames[0].columns)
    if any(list(frame.columns) != summaries for frame in frames):
        raise ValueError("compared models must share the same summaries")
    if any(len(frame) < 2 for frame in frames):
        raise ValueError("KDE plots require at least two predictive draws")
    observed_frame = (
        pd.DataFrame([observed])
        if isinstance(observed, Mapping)
        else observed
    )
    if observed_frame.empty:
        raise ValueError("at least one observed summary is required")
    if set(observed_frame.columns) != set(summaries):
        raise ValueError("observed and simulated summaries must match")
    observed_frame = observed_frame[summaries]
    stacked = [frame[summaries] for frame in frames]
    if (
        any(not np.all(np.isfinite(frame.to_numpy())) for frame in stacked)
        or not np.all(np.isfinite(observed_frame.to_numpy()))
    ):
        raise ValueError("predictive and observed summaries must be finite")

    count = len(summaries)
    figure, axes = plt.subplots(
        count,
        count,
        figsize=(3.2 * count, 3.2 * count),
        squeeze=False,
    )
    limits = [
        _padded_limits(
            *(frame[name] for frame in stacked),
            observed_frame[name],
            fraction=0.03,
        )
        for name in summaries
    ]
    colors = [
        MODEL_COMPARISON_COLORS[index % len(MODEL_COMPARISON_COLORS)]
        for index in range(len(names))
    ]
    single_observation = len(observed_frame) == 1

    for row, y_name in enumerate(summaries):
        for column, x_name in enumerate(summaries):
            axis = axes[row, column]
            if row == column:
                for frame, color in zip(stacked, colors):
                    sns.kdeplot(
                        x=frame[x_name],
                        ax=axis,
                        color=color,
                        fill=True,
                        alpha=0.25,
                        linewidth=1.5,
                        warn_singular=False,
                    )
                if single_observation:
                    observed_value = observed_frame[x_name].iloc[0]
                    axis.axvline(
                        observed_value,
                        color="tab:red",
                        linestyle=":",
                        linewidth=1,
                    )
                    axis.scatter(
                        observed_value,
                        0,
                        marker="*",
                        s=140,
                        color="tab:red",
                        edgecolor="white",
                        linewidth=0.7,
                        zorder=3,
                        clip_on=False,
                    )
                else:
                    sns.kdeplot(
                        x=observed_frame[x_name],
                        ax=axis,
                        color="tab:red",
                        linewidth=1.3,
                        warn_singular=False,
                    )
            else:
                for frame, color in zip(stacked, colors):
                    sns.kdeplot(
                        x=frame[x_name],
                        y=frame[y_name],
                        ax=axis,
                        color=color,
                        levels=6,
                        linewidths=1.2,
                        thresh=0.05,
                        warn_singular=False,
                    )
                if single_observation:
                    axis.scatter(
                        observed_frame[x_name],
                        observed_frame[y_name],
                        marker="*",
                        s=170,
                        color="tab:red",
                        edgecolor="white",
                        linewidth=0.7,
                        zorder=3,
                    )
                else:
                    axis.scatter(
                        observed_frame[x_name],
                        observed_frame[y_name],
                        s=8,
                        color="tab:red",
                        alpha=0.12,
                        linewidth=0,
                        rasterized=True,
                    )
            _format_pair_axis(
                axis,
                row=row,
                column=column,
                labels=summaries,
                limits=limits,
                diagonal_density_label=True,
            )

    figure.legend(
        handles=[
            *(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    linewidth=1.5,
                    label=name.replace("_", " "),
                )
                for name, color in zip(names, colors)
            ),
            Line2D(
                [0],
                [0],
                color="tab:red",
                marker="*",
                linestyle="None",
                markersize=10,
                label="Observed",
            ),
        ],
        loc="upper right",
        frameon=False,
    )
    figure.suptitle(
        "Prior-predictive summary statistics by model (observed data: red)"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return figure


def plot_model_probabilities(
    probabilities: ArrayLike,
    model_names: Sequence[str],
    *,
    dataset: str,
) -> Figure:
    """Plot mean posterior probabilities for the observed dataset."""

    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(values) != len(model_names):
        raise ValueError("model names and probabilities must have equal length")
    width = max(6.0, 1.6 * len(model_names))
    figure, axis = plt.subplots(figsize=(width, 4.5))
    bars = axis.bar(model_names, values, color="tab:blue", alpha=0.8)
    axis.bar_label(
        bars,
        labels=[f"{probability:.3f}" for probability in values],
        padding=3,
    )
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Posterior model probability")
    axis.set_title(f"BayesFlow model comparison for observed {dataset}")
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    return figure


__all__ = [
    "plot_model_comparison_pairplot",
    "plot_model_probabilities",
    "plot_predictive_summary_pairplot",
    "plot_prior_posterior_pairplot",
    "plot_summary_pairplot",
    "prepare_posterior_plot_data",
]
