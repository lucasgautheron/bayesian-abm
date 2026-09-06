"""Data contract for daily MemeTracker story activity."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

STORY_DATASET = "story_daily"
STORY_KEYS = ("mentions",)
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
        raise ValueError("mentions must have shape (n_stories, n_days)")
    if not np.all(np.isfinite(mentions)):
        raise ValueError("mentions must be finite")
    if np.any(mentions < 0):
        raise ValueError("mentions must be non-negative")
    return {"mentions": mentions}


__all__ = [
    "STORY_DATASET",
    "STORY_KEYS",
    "StoryData",
    "validate_story_data",
]
