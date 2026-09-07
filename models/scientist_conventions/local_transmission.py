"""Local cultural transmission through a scientist's first coauthor."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pymc as pm
from numpy.typing import ArrayLike, NDArray

from datasets.scientist_conventions.schema import validate_scientist_context
from .base import ScientistConventionModel
from .cultural import (
    CULTURAL_INFERENCE_VARIABLES,
    CULTURAL_PARAMETER_UNITS,
    build_cultural_prior,
    cultural_probabilities,
)


def simulate_local_transmission(
    probabilities: NDArray[np.float64],
    rng: np.random.Generator,
    *,
    primary_area: ArrayLike,
    career_start_year: ArrayLike,
    first_coauthor: ArrayLike,
    start_year: int,
    imitation_probability: float,
) -> NDArray[np.int8]:
    """Assign preferences chronologically, imitating an available first coauthor."""

    areas = np.asarray(primary_area, dtype=np.intp)
    career_years = np.asarray(career_start_year, dtype=np.int64)
    first = np.asarray(first_coauthor, dtype=np.int64)
    n_scientists = len(areas)
    if career_years.shape != (n_scientists,) or first.shape != (n_scientists,):
        raise ValueError("local-transmission covariates must align by scientist")
    if (
        not np.isfinite(imitation_probability)
        or imitation_probability < 0.0
        or imitation_probability > 1.0
    ):
        raise ValueError("imitation_probability must be between zero and one")
    if np.any((first < -1) | (first >= n_scientists)):
        raise ValueError("first_coauthor contains an unknown scientist ID")

    preference = np.zeros(n_scientists, dtype=np.int8)
    order = np.lexsort((np.arange(n_scientists), career_years))
    for scientist_id in order:
        scientist_id = int(scientist_id)
        coauthor = int(first[scientist_id])
        can_imitate = coauthor >= 0 and preference[coauthor] != 0
        if can_imitate and rng.random() < imitation_probability:
            preference[scientist_id] = preference[coauthor]
            continue
        year_index = (
            max(int(career_years[scientist_id]), int(start_year))
            - int(start_year)
        )
        chance_plus = probabilities[year_index, int(areas[scientist_id])]
        preference[scientist_id] = 1 if rng.random() < chance_plus else -1
    return preference


class LocalTransmissionModel(ScientistConventionModel):
    """Global cultural adoption with learnable first-coauthor imitation."""

    name = "scientist_local_transmission"
    inference_variables = (
        *CULTURAL_INFERENCE_VARIABLES,
        "imitation_probability",
    )
    parameter_units = CULTURAL_PARAMETER_UNITS

    def build_prior(self, **context: Any) -> pm.Model:
        return build_cultural_prior(include_imitation=True, **context)

    def simulate(
        self,
        parameters: Mapping[str, NDArray[Any]],
        rng: np.random.Generator,
        **context: Any,
    ) -> Mapping[str, ArrayLike]:
        validate_scientist_context(context)
        probabilities = cultural_probabilities(
            parameters,
            start_year=int(context["start_year"]),
            end_year=int(context["end_year"]),
        )
        return {
            "preference": simulate_local_transmission(
                probabilities,
                rng,
                primary_area=context["primary_area"],
                career_start_year=context["career_start_year"],
                first_coauthor=context["first_coauthor"],
                start_year=int(context["start_year"]),
                imitation_probability=float(parameters["imitation_probability"]),
            )
        }


__all__ = ["LocalTransmissionModel", "simulate_local_transmission"]
