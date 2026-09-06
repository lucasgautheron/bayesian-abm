"""Base contract for models of daily MemeTracker story activity."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from numpy.typing import ArrayLike, NDArray

from base.model import Model, SimulationData
from base.summaries import compute_scalar_summaries
from datasets.story_daily.schema import (
    STORY_DATASET,
    validate_story_data,
)
from datasets.story_daily.summaries import STORY_SUMMARY_STATISTICS


class StoryModel(Model):
    """A model whose native simulation is a population of daily series."""

    dataset = STORY_DATASET

    def validate_simulation(
        self,
        simulation: Mapping[str, ArrayLike],
        **context: Any,
    ) -> SimulationData:
        return validate_story_data(simulation, n_days=context["n_days"])

    def summarize(
        self,
        simulation: Mapping[str, ArrayLike],
        summaries: Mapping[str, Any],
        **context: Any,
    ) -> dict[str, NDArray[Any]]:
        data = self.validate_simulation(simulation, **context)
        if set(summaries) != set(STORY_SUMMARY_STATISTICS):
            raise ValueError("story summaries do not match the registered set")
        return compute_scalar_summaries(
            data,
            summaries,
            label="story summary",
        )


__all__ = ["StoryModel"]
