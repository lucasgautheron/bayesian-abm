"""Summary statistics for daily MemeTracker story activity."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from base.summaries import select_summaries
from datasets.story_daily.schema import validate_story_data

DEFAULT_STORY_SUMMARY_COUNT = 1_000
StorySummaryStatistic = Callable[[NDArray[np.float32]], float]


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
    keys = tuple(
        -mentions[:, day] for day in range(int(n_days) - 1, -1, -1)
    ) + (-totals,)
    order = np.lexsort(keys)[: int(story_count)]
    return mentions[order]


def _mean_story_autocorrelation(mentions: NDArray[np.float32]) -> float:
    if not len(mentions) or mentions.shape[1] < 2:
        return 0.0
    series = mentions.astype(np.float64, copy=False)
    first = series[:, :-1]
    second = series[:, 1:]
    first = first - first.mean(axis=1, keepdims=True)
    second = second - second.mean(axis=1, keepdims=True)
    numerator = np.sum(first * second, axis=1)
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    correlations = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=~np.isclose(denominator, 0.0),
    )
    return float(correlations.mean())


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
    "total_mentions": _total_mentions,
    "daily_total_stdev": _daily_total_stdev,
    "mean_story_autocorrelation": _mean_story_autocorrelation,
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
    summary_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return selected scalar conditions used by daily-story models."""

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
    statistics = select_summaries(
        STORY_SUMMARY_STATISTICS,
        summary_names,
        label="story summary",
    )
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
        for name, statistic in statistics.items()
    }


__all__ = [
    "DEFAULT_STORY_SUMMARY_COUNT",
    "STORY_SUMMARY_STATISTICS",
    "StorySummaryStatistic",
    "make_story_summaries",
    "ranked_story_mentions",
]
