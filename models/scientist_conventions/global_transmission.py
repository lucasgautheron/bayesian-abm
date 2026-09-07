"""Global cultural transmission of scientists' convention preferences."""

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
    global_preference_draws,
)


class GlobalTransmissionModel(ScientistConventionModel):
    """Scientists adopt a time- and area-dependent convention at career start."""

    name = "scientist_global_transmission"
    inference_variables = CULTURAL_INFERENCE_VARIABLES
    parameter_units = CULTURAL_PARAMETER_UNITS

    def build_prior(self, **context: Any) -> pm.Model:
        return build_cultural_prior(**context)

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
            "preference": global_preference_draws(
                probabilities,
                rng,
                primary_area=np.asarray(context["primary_area"]),
                career_start_year=np.asarray(context["career_start_year"]),
                start_year=int(context["start_year"]),
            )
        }


__all__ = ["GlobalTransmissionModel"]
