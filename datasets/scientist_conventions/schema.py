"""Data contract for scientists' binary convention preferences."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix

SCIENTIST_CONVENTIONS_DATASET = "scientist_conventions"
PREFERENCE_KEYS = ("preference",)
RESEARCH_AREA_COUNT = 4
CULTURAL_BASELINE_YEAR = 1981
PreferenceData = dict[str, NDArray[np.int8]]


def validate_preference_data(
    data: Mapping[str, ArrayLike],
    *,
    n_scientists: int,
) -> PreferenceData:
    """Validate a complete binary convention assignment."""

    if set(data) != set(PREFERENCE_KEYS):
        raise ValueError("scientist data must contain exactly 'preference'")
    if (
        isinstance(n_scientists, (bool, np.bool_))
        or not isinstance(n_scientists, (int, np.integer))
        or n_scientists < 1
    ):
        raise ValueError("n_scientists must be a positive integer")
    preference = np.asarray(data["preference"])
    if preference.ndim != 1 or len(preference) != int(n_scientists):
        raise ValueError("preference must have shape (n_scientists,)")
    if preference.dtype != np.int8:
        raise TypeError("preference must have dtype int8")
    if not np.all(np.isin(preference, (-1, 1))):
        raise ValueError("preference values must be -1 or +1")
    return {"preference": preference}


def validate_scientist_context(context: Mapping[str, Any]) -> None:
    """Validate graph and covariate shapes shared by scientist models."""

    n_scientists = int(context["n_scientists"])
    if n_scientists < 1:
        raise ValueError("n_scientists must be positive")
    primary_area = np.asarray(context["primary_area"])
    area_shares = np.asarray(context["area_shares"])
    career_start_year = np.asarray(context["career_start_year"])
    observed_mask = np.asarray(context["observed_mask"])
    if primary_area.shape != (n_scientists,):
        raise ValueError("primary_area must have shape (n_scientists,)")
    if (
        np.any(primary_area < 0)
        or np.any(primary_area >= RESEARCH_AREA_COUNT)
    ):
        raise ValueError("primary_area values must be between 0 and 3")
    if area_shares.shape != (n_scientists, RESEARCH_AREA_COUNT):
        raise ValueError("area_shares must have shape (n_scientists, 4)")
    if (
        not np.all(np.isfinite(area_shares))
        or np.any(area_shares < 0.0)
        or np.any(area_shares > 1.0)
        or np.any(np.isclose(area_shares.sum(axis=1), 0.0))
    ):
        raise ValueError(
            "area_shares must be finite per-area publication probabilities"
        )
    if career_start_year.shape != (n_scientists,):
        raise ValueError("career_start_year must have shape (n_scientists,)")
    if observed_mask.shape != (n_scientists,) or observed_mask.dtype != np.bool_:
        raise ValueError("observed_mask must be a boolean scientist vector")
    if not np.any(observed_mask):
        raise ValueError("at least one favorite convention must be observed")
    for name in ("coauthorship", "citations"):
        graph = context[name]
        if not isinstance(graph, csr_matrix):
            raise TypeError(f"{name} must be a scipy.sparse.csr_matrix")
        if graph.shape != (n_scientists, n_scientists):
            raise ValueError(
                f"{name} must have shape (n_scientists, n_scientists)"
            )
        if graph.nnz and (
            not np.all(np.isfinite(graph.data)) or np.any(graph.data <= 0.0)
        ):
            raise ValueError(f"{name} weights must be finite and positive")


__all__ = [
    "CULTURAL_BASELINE_YEAR",
    "PREFERENCE_KEYS",
    "PreferenceData",
    "RESEARCH_AREA_COUNT",
    "SCIENTIST_CONVENTIONS_DATASET",
    "validate_preference_data",
    "validate_scientist_context",
]
