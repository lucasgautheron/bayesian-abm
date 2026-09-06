"""Base contract for scientist-convention models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from numpy.typing import ArrayLike, NDArray

from base.model import Model, SimulationData
from base.summaries import compute_scalar_summaries
from datasets.scientist_conventions.schema import (
    SCIENTIST_CONVENTIONS_DATASET,
    validate_preference_data,
)
from datasets.scientist_conventions.summaries import SCIENTIST_SUMMARY_NAMES


class ScientistConventionModel(Model):
    """A model assigning one binary convention to every scientist."""

    dataset = SCIENTIST_CONVENTIONS_DATASET

    def validate_simulation(
        self,
        simulation: Mapping[str, ArrayLike],
        **context: Any,
    ) -> SimulationData:
        return validate_preference_data(
            simulation,
            n_scientists=context["n_scientists"],
        )

    def summarize(
        self,
        simulation: Mapping[str, ArrayLike],
        summaries: Mapping[str, Any],
        **context: Any,
    ) -> dict[str, NDArray[Any]]:
        data = self.validate_simulation(simulation, **context)
        if set(summaries) != set(SCIENTIST_SUMMARY_NAMES):
            raise ValueError(
                "scientist summaries do not match the registered set"
            )
        return compute_scalar_summaries(
            data,
            summaries,
            label="scientist summary",
        )


__all__ = ["ScientistConventionModel"]
