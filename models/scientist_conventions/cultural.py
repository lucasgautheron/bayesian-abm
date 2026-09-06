"""Shared latent cultural trajectories for convention transmission models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pymc as pm
from numpy.typing import NDArray

from datasets.scientist_conventions.schema import (
    RESEARCH_AREA_COUNT,
    validate_scientist_context,
)


CULTURAL_INFERENCE_VARIABLES = (
    "baseline_log_odds",
    "annual_drift",
    "innovation_scale",
)


def build_cultural_prior(
    *,
    include_imitation: bool = False,
    **context: Any,
) -> pm.Model:
    """Build the confirmed category-specific logit-random-walk prior."""

    validate_scientist_context(context)
    n_years = int(context["end_year"]) - int(context["start_year"]) + 1
    if n_years < 1:
        raise ValueError("end_year must not precede start_year")
    with pm.Model() as prior:
        pm.Normal(
            "baseline_log_odds",
            mu=0.0,
            sigma=1.5,
            shape=RESEARCH_AREA_COUNT,
        )
        pm.Normal(
            "annual_drift",
            mu=0.0,
            sigma=0.05,
            shape=RESEARCH_AREA_COUNT,
        )
        pm.HalfNormal("innovation_scale", sigma=0.1)
        pm.Normal(
            "annual_innovation",
            mu=0.0,
            sigma=1.0,
            shape=(max(1, n_years - 1), RESEARCH_AREA_COUNT),
        )
        if include_imitation:
            pm.Beta("imitation_probability", alpha=1.0, beta=1.0)
    return prior


def cultural_probabilities(
    parameters: Mapping[str, NDArray[Any]],
    *,
    start_year: int,
    end_year: int,
) -> NDArray[np.float64]:
    """Return year-by-area probabilities from one latent random-walk draw."""

    n_years = int(end_year) - int(start_year) + 1
    if n_years < 1:
        raise ValueError("end_year must not precede start_year")
    baseline = np.asarray(parameters["baseline_log_odds"], dtype=np.float64)
    drift = np.asarray(parameters["annual_drift"], dtype=np.float64)
    innovation_scale = float(parameters["innovation_scale"])
    innovations = np.asarray(parameters["annual_innovation"], dtype=np.float64)
    if baseline.shape != (RESEARCH_AREA_COUNT,):
        raise ValueError("baseline_log_odds must have shape (4,)")
    if drift.shape != (RESEARCH_AREA_COUNT,):
        raise ValueError("annual_drift must have shape (4,)")
    if innovations.ndim != 2 or innovations.shape[1] != RESEARCH_AREA_COUNT:
        raise ValueError("annual_innovation must have four area columns")
    if len(innovations) < max(1, n_years - 1):
        raise ValueError("annual_innovation does not span the career-year range")
    if not np.isfinite(innovation_scale) or innovation_scale < 0.0:
        raise ValueError("innovation_scale must be finite and non-negative")

    logits = np.empty((n_years, RESEARCH_AREA_COUNT), dtype=np.float64)
    logits[0] = baseline
    if n_years > 1:
        increments = drift + innovation_scale * innovations[: n_years - 1]
        logits[1:] = baseline + np.cumsum(increments, axis=0)
    probabilities = np.empty_like(logits)
    positive = logits >= 0.0
    probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exponential = np.exp(logits[~positive])
    probabilities[~positive] = exponential / (1.0 + exponential)
    return probabilities


def global_preference_draws(
    probabilities: NDArray[np.float64],
    rng: np.random.Generator,
    *,
    primary_area: NDArray[np.integer],
    career_start_year: NDArray[np.integer],
    start_year: int,
) -> NDArray[np.int8]:
    """Draw independent preferences at each scientist's career start."""

    areas = np.asarray(primary_area, dtype=np.intp)
    years = (
        np.maximum(np.asarray(career_start_year, dtype=np.int64), int(start_year))
        - int(start_year)
    )
    if areas.ndim != 1 or years.shape != areas.shape:
        raise ValueError("scientist areas and career years must be aligned vectors")
    if (
        np.any(areas < 0)
        or np.any(areas >= probabilities.shape[1])
        or np.any(years < 0)
        or np.any(years >= probabilities.shape[0])
    ):
        raise ValueError("scientist area or career year falls outside the trajectory")
    chance_plus = probabilities[years, areas]
    return np.where(rng.random(len(areas)) < chance_plus, 1, -1).astype(np.int8)


__all__ = [
    "CULTURAL_INFERENCE_VARIABLES",
    "build_cultural_prior",
    "cultural_probabilities",
    "global_preference_draws",
]
