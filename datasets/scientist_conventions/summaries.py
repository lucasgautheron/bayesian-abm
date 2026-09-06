"""Summary statistics for scientists' convention preferences."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

from datasets.scientist_conventions.schema import (
    PreferenceData,
    RESEARCH_AREA_COUNT,
    validate_preference_data,
    validate_scientist_context,
)

SCIENTIST_SUMMARY_NAMES = (
    "field_phenomenology_hep",
    "field_theory_hep",
    "field_gravitation_cosmology",
    "field_astrophysics",
    "coauthorship_coupling",
    "citation_coupling",
)
ScientistSummary = Callable[[PreferenceData], float]


def fit_ising_parameters(
    preference: ArrayLike,
    *,
    primary_area: ArrayLike,
    observed_mask: ArrayLike,
    coauthorship: csr_matrix,
    citations: csr_matrix,
) -> NDArray[np.float64]:
    """Return four fields and two normalized couplings from pseudo-likelihood."""

    spins = np.asarray(preference, dtype=np.float64)
    areas = np.asarray(primary_area, dtype=np.intp)
    observed = np.asarray(observed_mask, dtype=np.bool_)
    n_scientists = len(spins)
    if spins.shape != (n_scientists,) or not np.all(
        np.isin(spins, (-1.0, 1.0))
    ):
        raise ValueError("preference must be a one-dimensional -1/+1 vector")
    if areas.shape != (n_scientists,):
        raise ValueError("primary_area must align with preference")
    if observed.shape != (n_scientists,) or not np.any(observed):
        raise ValueError("observed_mask must select at least one scientist")
    if coauthorship.shape != (n_scientists, n_scientists):
        raise ValueError("coauthorship shape must align with preference")
    if citations.shape != (n_scientists, n_scientists):
        raise ValueError("citations shape must align with preference")

    masked_spins = spins * observed
    coauthor_scale = float(np.asarray(coauthorship.sum(axis=1)).mean())
    citation_scale = float(np.asarray(citations.sum(axis=1)).mean())
    if not np.isfinite(coauthor_scale) or coauthor_scale <= 0.0:
        raise ValueError(
            "coauthorship graph must have positive mean weighted degree"
        )
    if not np.isfinite(citation_scale) or citation_scale <= 0.0:
        raise ValueError(
            "citation graph must have positive mean weighted out-degree"
        )

    indices = np.flatnonzero(observed)
    selected_spins = spins[indices]
    selected_areas = areas[indices]
    interaction = (
        np.asarray(coauthorship @ masked_spins)[indices] * selected_spins
    )
    citation_interaction = (
        np.asarray(citations @ masked_spins)[indices] * selected_spins
    )
    design = np.zeros(
        (len(indices), RESEARCH_AREA_COUNT + 2),
        dtype=np.float64,
    )
    design[np.arange(len(indices)), selected_areas] = 2.0 * selected_spins
    design[:, RESEARCH_AREA_COUNT] = 4.0 * interaction / coauthor_scale
    design[:, RESEARCH_AREA_COUNT + 1] = (
        4.0 * citation_interaction / citation_scale
    )

    def objective(parameters: NDArray[np.float64]) -> float:
        logits = design @ parameters
        return float(
            np.logaddexp(0.0, -logits).sum()
            + 0.5 * np.dot(parameters, parameters)
        )

    def gradient(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        logits = design @ parameters
        inverse_logit = np.exp(-np.logaddexp(0.0, logits))
        return parameters - design.T @ inverse_logit

    result = minimize(
        objective,
        np.zeros(RESEARCH_AREA_COUNT + 2, dtype=np.float64),
        jac=gradient,
        method="L-BFGS-B",
        options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 500},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(
            "Ising pseudo-likelihood optimization failed: "
            f"{result.message}"
        )
    fitted = np.asarray(result.x, dtype=np.float64)
    fitted[RESEARCH_AREA_COUNT] /= coauthor_scale
    fitted[RESEARCH_AREA_COUNT + 1] /= citation_scale
    return fitted


def make_scientist_summaries(**context: Any) -> dict[str, ScientistSummary]:
    """Build cached scalar Ising summaries for one fixed scientist network."""

    validate_scientist_context(context)
    n_scientists = int(context["n_scientists"])
    primary_area = np.asarray(context["primary_area"])
    observed_mask = np.asarray(context["observed_mask"])
    coauthorship = context["coauthorship"]
    citations = context["citations"]
    cached_preference: bytes | None = None
    cached_fit = np.zeros(len(SCIENTIST_SUMMARY_NAMES), dtype=np.float64)

    def fitted(data: PreferenceData) -> NDArray[np.float64]:
        nonlocal cached_preference, cached_fit
        preference = validate_preference_data(
            data,
            n_scientists=n_scientists,
        )["preference"]
        key = preference.tobytes()
        if key != cached_preference:
            cached_fit = fit_ising_parameters(
                preference,
                primary_area=primary_area,
                observed_mask=observed_mask,
                coauthorship=coauthorship,
                citations=citations,
            )
            cached_preference = key
        return cached_fit

    return {
        name: lambda data, index=index: float(fitted(data)[index])
        for index, name in enumerate(SCIENTIST_SUMMARY_NAMES)
    }


__all__ = [
    "SCIENTIST_SUMMARY_NAMES",
    "ScientistSummary",
    "fit_ising_parameters",
    "make_scientist_summaries",
]
