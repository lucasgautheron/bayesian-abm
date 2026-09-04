"""Base contract for models of daily MemeTracker story activity."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from base.model import Model, SimulationData
from base.summaries import compute_scalar_summaries


STORY_DATASET = "story_daily"
STORY_KEYS = ("mentions",)
DEFAULT_STORY_SUMMARY_COUNT = 1_000
StoryData = dict[str, NDArray[np.float32]]
StorySummaryStatistic = Callable[[NDArray[np.float32]], float]


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
    """Return up to ``story_count`` stories ranked by total mentions."""

    if (
        isinstance(story_count, (bool, np.bool_))
        or not isinstance(story_count, (int, np.integer))
        or story_count < 1
    ):
        raise ValueError("story_count must be a positive integer")

    mentions = validate_story_data(data, n_days=n_days)["mentions"]
    if not len(mentions):
        return mentions
    totals = mentions.sum(axis=1, dtype=np.float64)
    # Daily trajectories deterministically break equal-total ties.
    keys = tuple(
        -mentions[:, day] for day in range(int(n_days) - 1, -1, -1)
    ) + (-totals,)
    order = np.lexsort(keys)[: int(story_count)]
    return mentions[order]


def _selected_story_count(mentions: NDArray[np.float32]) -> float:
    return float(len(mentions))


def _total_mentions(mentions: NDArray[np.float32]) -> float:
    return float(mentions.sum(dtype=np.float64))


def _daily_total_stdev(mentions: NDArray[np.float32]) -> float:
    return float(mentions.sum(axis=0, dtype=np.float64).std())


def _mean_reporting_story_count(
    mentions: NDArray[np.float32],
) -> float:
    return float(np.count_nonzero(mentions > 0, axis=0).mean())


def _story_mentions_coefficient_of_variation(
    mentions: NDArray[np.float32],
) -> float:
    totals = mentions.sum(axis=1, dtype=np.float64)
    if not len(totals):
        return 0.0
    mean = totals.mean()
    return 0.0 if np.isclose(mean, 0.0) else float(totals.std() / mean)


def _mean_reporting_lifetime_days(
    mentions: NDArray[np.float32],
) -> float:
    if not len(mentions):
        return 0.0
    active = mentions > 0
    has_reports = active.any(axis=1)
    first = np.argmax(active, axis=1)
    last = mentions.shape[1] - 1 - np.argmax(active[:, ::-1], axis=1)
    return float(np.where(has_reports, last - first + 1, 0).mean())


def _mention_concentration(mentions: NDArray[np.float32]) -> float:
    totals = mentions.sum(axis=1, dtype=np.float64)
    total = totals.sum()
    if np.isclose(total, 0.0):
        return 0.0
    shares = totals / total
    return float(np.dot(shares, shares))


STORY_SUMMARY_STATISTICS: dict[str, StorySummaryStatistic] = {
    "selected_story_count": _selected_story_count,
    "total_mentions": _total_mentions,
    "daily_total_stdev": _daily_total_stdev,
    "mean_reporting_story_count": _mean_reporting_story_count,
    "story_mentions_coefficient_of_variation": (
        _story_mentions_coefficient_of_variation
    ),
    "mean_reporting_lifetime_days": _mean_reporting_lifetime_days,
    "mention_concentration": _mention_concentration,
}


def make_story_summaries(
    *,
    n_days: int,
    story_count: int = DEFAULT_STORY_SUMMARY_COUNT,
) -> dict[str, Any]:
    """Return scalar conditions used by daily-story models."""

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
    cached_data: Mapping[str, ArrayLike] | None = None
    cached_mentions = np.empty((0, int(n_days)), dtype=np.float32)

    def selected(data: Mapping[str, ArrayLike]) -> NDArray[np.float32]:
        nonlocal cached_data, cached_mentions
        if data is not cached_data:
            cached_data = data
            cached_mentions = ranked_story_mentions(
                data,
                n_days=n_days,
                story_count=story_count,
            )
        return cached_mentions

    return {
        name: lambda data, statistic=statistic: statistic(selected(data))
        for name, statistic in STORY_SUMMARY_STATISTICS.items()
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
        if set(summaries) != set(STORY_SUMMARY_STATISTICS):
            raise ValueError("story summaries do not match the registered set")
        return compute_scalar_summaries(
            data,
            summaries,
            label="story summary",
        )


__all__ = [
    "DEFAULT_STORY_SUMMARY_COUNT",
    "STORY_DATASET",
    "STORY_KEYS",
    "STORY_SUMMARY_STATISTICS",
    "StoryData",
    "StoryModel",
    "StorySummaryStatistic",
    "make_story_summaries",
    "ranked_story_mentions",
    "validate_story_data",
]
