"""Base contract for models of daily MemeTracker story activity."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from base.model import Model, SimulationData, bf


STORY_DATASET = "story_daily"
STORY_KEYS = ("mentions",)
STORY_SUMMARY_KEYS = ("mentions", "story_mask")
DEFAULT_STORY_SUMMARY_COUNT = 1_000
StoryData = dict[str, NDArray[np.float32]]


def validate_story_data(
    data: Mapping[str, ArrayLike],
    *,
    n_days: int,
) -> StoryData:
    """Validate a variable-size population of daily mentions series."""

    if set(data) != set(STORY_KEYS):
        raise ValueError("story data must contain exactly 'mentions'")
    if (
        isinstance(n_days, (bool, np.bool_))
        or not isinstance(n_days, (int, np.integer))
        or n_days < 1
    ):
        raise ValueError("n_days must be a positive integer")

    mentions = np.asarray(data["mentions"], dtype=np.float32)
    if mentions.ndim != 2 or mentions.shape[1] != int(n_days):
        raise ValueError(
            "mentions must have shape (n_stories, n_days)"
        )
    if not np.all(np.isfinite(mentions)):
        raise ValueError("mentions must be finite")
    if np.any(mentions < 0):
        raise ValueError("mentions must be non-negative")
    return {"mentions": mentions}


def ranked_story_mentions(
    data: Mapping[str, ArrayLike],
    *,
    n_days: int,
    story_count: int = DEFAULT_STORY_SUMMARY_COUNT,
) -> NDArray[np.float32]:
    """Return a fixed-size panel of the most-mentioned stories."""

    if (
        isinstance(story_count, (bool, np.bool_))
        or not isinstance(story_count, (int, np.integer))
        or story_count < 1
    ):
        raise ValueError("story_count must be a positive integer")

    mentions = validate_story_data(data, n_days=n_days)["mentions"]
    ranked = np.zeros((int(story_count), int(n_days)), dtype=np.float32)
    if len(mentions):
        totals = mentions.sum(axis=1, dtype=np.float64)
        # Total mentions is the primary descending key. Daily trajectories
        # provide deterministic, permutation-invariant tie breaking.
        keys = tuple(
            -mentions[:, day] for day in range(int(n_days) - 1, -1, -1)
        ) + (-totals,)
        order = np.lexsort(keys)[: int(story_count)]
        ranked[: len(order)] = mentions[order]
    return ranked


def ranked_story_mask(
    data: Mapping[str, ArrayLike],
    *,
    n_days: int,
    story_count: int = DEFAULT_STORY_SUMMARY_COUNT,
) -> NDArray[np.float32]:
    """Mark rows occupied by selected stories rather than zero padding."""

    if (
        isinstance(story_count, (bool, np.bool_))
        or not isinstance(story_count, (int, np.integer))
        or story_count < 1
    ):
        raise ValueError("story_count must be a positive integer")
    mentions = validate_story_data(data, n_days=n_days)["mentions"]
    mask = np.zeros(int(story_count), dtype=np.float32)
    mask[: min(len(mentions), int(story_count))] = 1.0
    return mask


def make_story_summaries(
    *,
    n_days: int,
    story_count: int = DEFAULT_STORY_SUMMARY_COUNT,
) -> dict[str, Any]:
    """Return fixed-shape conditions used by daily-story models."""

    if (
        isinstance(n_days, (bool, np.bool_))
        or not isinstance(n_days, (int, np.integer))
        or n_days < 1
    ):
        raise ValueError("n_days must be a positive integer")
    if (
        isinstance(story_count, (bool, np.bool_))
        or not isinstance(story_count, (int, np.integer))
        or story_count < 1
    ):
        raise ValueError("story_count must be a positive integer")
    return {
        "mentions": lambda data: ranked_story_mentions(
            data,
            n_days=n_days,
            story_count=story_count,
        ),
        "story_mask": lambda data: ranked_story_mask(
            data,
            n_days=n_days,
            story_count=story_count,
        ),
    }


class StoryModel(Model):
    """A model whose native simulation is a population of daily series."""

    dataset = STORY_DATASET

    def validate_simulation(
        self,
        simulation: Mapping[str, ArrayLike],
        **context: Any,
    ) -> SimulationData:
        try:
            n_days = context["n_days"]
        except KeyError as exc:
            raise ValueError("story models require n_days") from exc
        return validate_story_data(simulation, n_days=n_days)

    def summarize(
        self,
        simulation: Mapping[str, ArrayLike],
        summaries: Mapping[str, Any],
        **context: Any,
    ) -> dict[str, NDArray[Any]]:
        data = self.validate_simulation(simulation, **context)
        if set(summaries) != set(STORY_SUMMARY_KEYS):
            raise ValueError(
                "story summaries must contain exactly "
                "'mentions' and 'story_mask'"
            )
        result = {
            name: np.atleast_1d(
                np.asarray(function(data), dtype=np.float32)
            )
            for name, function in summaries.items()
        }
        if any(not np.all(np.isfinite(value)) for value in result.values()):
            raise ValueError("story summary statistics must be finite")
        return result

    def _adapt_summary_variables(
        self,
        adapter: bf.Adapter,
        summary_names: Sequence[str],
    ) -> bf.Adapter:
        if set(summary_names) != set(STORY_SUMMARY_KEYS):
            raise ValueError(
                "story summaries must contain exactly "
                "'mentions' and 'story_mask'"
            )
        return (
            adapter.rename("mentions", "summary_variables")
            .rename("story_mask", "summary_mask")
        )

    def make_bayesflow_summary_network(
        self,
        **context: Any,
    ) -> Any:
        del context
        from .summary_network import StoryPopulationSummaryNetwork

        return StoryPopulationSummaryNetwork()

    @staticmethod
    def make_bayesflow_model_comparison_adapter(
        summaries: Mapping[str, Any],
    ) -> bf.Adapter:
        names = list(summaries)
        if set(names) != set(STORY_SUMMARY_KEYS):
            raise ValueError(
                "story summaries must contain exactly "
                "'mentions' and 'story_mask'"
            )
        return (
            bf.Adapter()
            .to_array(include=names)
            .convert_dtype("float64", "float32", include=names)
            .rename("mentions", "summary_variables")
            .rename("story_mask", "summary_mask")
        )


__all__ = [
    "DEFAULT_STORY_SUMMARY_COUNT",
    "STORY_DATASET",
    "STORY_KEYS",
    "STORY_SUMMARY_KEYS",
    "StoryData",
    "StoryModel",
    "make_story_summaries",
    "ranked_story_mask",
    "ranked_story_mentions",
    "validate_story_data",
]
