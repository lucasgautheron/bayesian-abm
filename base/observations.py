"""Dataset-specific loading of observed inference conditions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd

from base.model import ContactData
from base.summaries import (
    INTERVAL_SECONDS,
    compute_summaries,
    make_summaries,
)
from models import MODEL_REGISTRIES
from models.stories import (
    DEFAULT_STORY_SUMMARY_COUNT,
    make_story_summaries,
)


CONTACTS = "contacts"
STORY_DAILY = "story_daily"
DEFAULT_STORY_COUNT = DEFAULT_STORY_SUMMARY_COUNT
DATASETS = tuple(MODEL_REGISTRIES)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATHS = {
    CONTACTS: ROOT / "data" / "contacts" / "contacts.parquet",
    STORY_DAILY: (
        ROOT / "data" / "stories" / "processed" / "story_daily.parquet"
    ),
}

SummaryFunction = Callable[[Mapping[str, ArrayLike]], ArrayLike]
Summaries = Mapping[str, SummaryFunction]


@dataclass(frozen=True)
class Observations:
    """Inference-ready conditions and simulation context for one dataset."""

    dataset: str
    context: Mapping[str, int]
    summaries: Summaries
    conditions: Mapping[str, NDArray[Any]]
    observation_ids: NDArray[Any]

    @property
    def count(self) -> int:
        return len(self.observation_ids)


def load_contacts(path: Path) -> tuple[ContactData, int, int]:
    """Load contacts and normalize IDs and times to model conventions."""

    frame = pd.read_parquet(path, columns=["t", "i", "j"])
    if frame.empty:
        raise ValueError("the observed contact data is empty")
    if frame[["t", "i", "j"]].isna().any().any():
        raise ValueError("contact columns must not contain null values")

    times = frame["t"].to_numpy()
    if np.any(times % INTERVAL_SECONDS):
        raise ValueError("contact times must fall on 20-second boundaries")

    agent_ids = np.unique(
        np.concatenate(
            (frame["i"].to_numpy(), frame["j"].to_numpy())
        )
    )
    start = int(times.min())
    normalized_times = times - start + INTERVAL_SECONDS
    contacts = {
        "t": normalized_times.astype(np.int32),
        "i": np.searchsorted(
            agent_ids,
            frame["i"].to_numpy(),
        ).astype(np.int32),
        "j": np.searchsorted(
            agent_ids,
            frame["j"].to_numpy(),
        ).astype(np.int32),
    }
    n_steps = int(normalized_times.max() // INTERVAL_SECONDS)
    return contacts, len(agent_ids), n_steps


def contact_observations(path: Path) -> Observations:
    """Load one pairwise-contact dataset."""

    contacts, n_agents, n_steps = load_contacts(path)
    context = {"n_agents": n_agents, "n_steps": n_steps}
    summaries = make_summaries(**context)
    conditions = {
        name: values[None, ...]
        for name, values in compute_summaries(
            contacts,
            summaries,
        ).items()
    }
    return Observations(
        dataset=CONTACTS,
        context=context,
        summaries=summaries,
        conditions=conditions,
        observation_ids=np.asarray([CONTACTS]),
    )


def story_daily_frame_observations(
    frame: pd.DataFrame,
    *,
    story_count: int = DEFAULT_STORY_COUNT,
) -> Observations:
    """Convert a dense daily panel into one joint population condition."""

    required = {"story_id", "date", "mentions"}
    missing = required.difference(frame.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"story_daily data is missing columns: {names}")
    if frame.empty:
        raise ValueError("the observed story_daily data is empty")
    if frame[list(required)].isna().any().any():
        raise ValueError("story_daily columns must not contain null values")

    story_ids = frame["story_id"].to_numpy()
    dates = frame["date"].to_numpy()
    mentions = frame["mentions"].to_numpy()
    boundaries = np.flatnonzero(story_ids[1:] != story_ids[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(frame)]))
    block_ids = story_ids[starts]
    if len(np.unique(block_ids)) != len(block_ids):
        raise ValueError("each story_id must occupy one contiguous block")

    day_counts = ends - starts
    n_days = int(day_counts[0])
    if n_days < 1 or np.any(day_counts != n_days):
        raise ValueError("all stories must have the same number of daily rows")

    date_matrix = dates.reshape(len(block_ids), n_days)
    date_grid = date_matrix[0]
    if np.any(date_grid[1:] <= date_grid[:-1]):
        raise ValueError("story dates must be unique and increasing")
    if np.any(date_matrix != date_grid):
        raise ValueError("all stories must use the same date grid")

    mention_matrix = np.asarray(
        mentions.reshape(len(block_ids), n_days),
        dtype=np.float32,
    )
    if not np.all(np.isfinite(mention_matrix)):
        raise ValueError("story mentions must be finite")
    if np.any(mention_matrix < 0):
        raise ValueError("story mentions must be non-negative")

    context = {"n_days": n_days}
    summaries = make_story_summaries(
        **context,
        story_count=story_count,
    )
    story_data = {"mentions": mention_matrix}
    conditions = {
        name: np.atleast_1d(
            np.asarray(summary(story_data), dtype=np.float32)
        )[None, ...]
        for name, summary in summaries.items()
    }
    return Observations(
        dataset=STORY_DAILY,
        context=context,
        summaries=summaries,
        conditions=conditions,
        observation_ids=np.asarray([STORY_DAILY]),
    )


def story_daily_observations(path: Path) -> Observations:
    """Load and summarize the complete daily story population."""

    frame = pd.read_parquet(
        path,
        columns=["story_id", "date", "mentions"],
    )
    return story_daily_frame_observations(frame)


OBSERVATION_LOADERS = {
    CONTACTS: contact_observations,
    STORY_DAILY: story_daily_observations,
}


def load_observations(
    dataset: str,
    path: Path | None = None,
) -> Observations:
    """Load inference conditions using one dataset's native context."""

    try:
        loader = OBSERVATION_LOADERS[dataset]
    except KeyError as exc:
        choices = ", ".join(DATASETS)
        raise ValueError(
            f"unknown dataset {dataset!r}; available datasets: {choices}"
        ) from exc
    return loader(path or DEFAULT_DATA_PATHS[dataset])


__all__ = [
    "CONTACTS",
    "DATASETS",
    "DEFAULT_DATA_PATHS",
    "DEFAULT_STORY_COUNT",
    "OBSERVATION_LOADERS",
    "STORY_DAILY",
    "Observations",
    "contact_observations",
    "load_contacts",
    "load_observations",
    "story_daily_frame_observations",
    "story_daily_observations",
]
