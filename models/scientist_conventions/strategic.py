"""Strategic best-response model for scientist convention preferences."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pymc as pm
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix

from datasets.scientist_conventions.schema import (
    RESEARCH_AREA_COUNT,
    validate_scientist_context,
)
from .base import ScientistConventionModel


BEST_RESPONSE_SWEEPS = 50


def strategic_best_response(
    initial_preference: ArrayLike,
    *,
    coauthorship: csr_matrix,
    area_shares: ArrayLike,
    contextual_advantage: ArrayLike,
    coordination_cost: float,
    sweeps: int = BEST_RESPONSE_SWEEPS,
) -> NDArray[np.int8]:
    """Run deterministic-order asynchronous cost-minimizing updates."""

    preference = np.asarray(initial_preference, dtype=np.int8).copy()
    shares = np.asarray(area_shares, dtype=np.float64)
    advantage = np.asarray(contextual_advantage, dtype=np.float64)
    n_scientists = len(preference)
    if preference.shape != (n_scientists,) or not np.all(
        np.isin(preference, (-1, 1))
    ):
        raise ValueError("initial_preference must be a -1/+1 vector")
    if coauthorship.shape != (n_scientists, n_scientists):
        raise ValueError("coauthorship must align with initial_preference")
    if shares.shape != (n_scientists, RESEARCH_AREA_COUNT):
        raise ValueError("area_shares must have shape (n_scientists, 4)")
    if advantage.shape != (RESEARCH_AREA_COUNT,):
        raise ValueError("contextual_advantage must have shape (4,)")
    if not np.isfinite(coordination_cost) or coordination_cost < 0.0:
        raise ValueError("coordination_cost must be finite and non-negative")
    if sweeps < 1:
        raise ValueError("sweeps must be positive")

    adaptation = shares @ advantage
    neighbor_field = np.asarray(coauthorship @ preference, dtype=np.float64)
    for _ in range(int(sweeps)):
        for scientist_id in range(n_scientists):
            current = int(preference[scientist_id])
            cost_difference = (
                current
                + coordination_cost * neighbor_field[scientist_id]
                + adaptation[scientist_id]
            )
            if cost_difference < 0.0:
                updated = -1
            elif cost_difference > 0.0:
                updated = 1
            else:
                updated = current
            if updated != current:
                preference[scientist_id] = updated
                start = coauthorship.indptr[scientist_id]
                stop = coauthorship.indptr[scientist_id + 1]
                neighbors = coauthorship.indices[start:stop]
                weights = coauthorship.data[start:stop]
                neighbor_field[neighbors] += weights * (updated - current)
    return preference


class StrategicConventionModel(ScientistConventionModel):
    """Scientists balance switching, coordination, and maladaptation costs."""

    name = "scientist_strategic"
    inference_variables = ("contextual_advantage", "coordination_scale")

    def build_prior(self, **context: Any) -> pm.Model:
        validate_scientist_context(context)
        with pm.Model() as prior:
            pm.Normal(
                "contextual_advantage",
                mu=0.0,
                sigma=1.0,
                shape=RESEARCH_AREA_COUNT,
            )
            pm.Exponential("coordination_scale", lam=1.0)
        return prior

    def simulate(
        self,
        parameters: Mapping[str, NDArray[Any]],
        rng: np.random.Generator,
        **context: Any,
    ) -> Mapping[str, ArrayLike]:
        validate_scientist_context(context)
        coauthorship = context["coauthorship"]
        mean_weighted_degree = float(
            np.asarray(coauthorship.sum(axis=1)).mean()
        )
        if mean_weighted_degree <= 0.0:
            raise ValueError("coauthorship graph must have positive mean degree")
        initial = np.where(
            rng.random(int(context["n_scientists"])) < 0.5,
            -1,
            1,
        ).astype(np.int8)
        return {
            "preference": strategic_best_response(
                initial,
                coauthorship=coauthorship,
                area_shares=context["area_shares"],
                contextual_advantage=parameters["contextual_advantage"],
                coordination_cost=(
                    float(parameters["coordination_scale"])
                    / mean_weighted_degree
                ),
            )
        }


__all__ = [
    "BEST_RESPONSE_SWEEPS",
    "StrategicConventionModel",
    "strategic_best_response",
]
