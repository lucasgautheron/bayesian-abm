"""Dataset-specific loading of observed inference conditions."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd

from base.model import ContactData, INTERVAL_SECONDS
from base.summaries import compute_scalar_summaries
from datasets.contacts.summaries import compute_summaries, make_summaries
from datasets.story_daily.schema import STORY_DATASET
from datasets.story_daily.summaries import (
    DEFAULT_STORY_SUMMARY_COUNT,
    make_story_summaries,
)
from datasets.scientist_conventions.schema import (
    CULTURAL_BASELINE_YEAR,
    SCIENTIST_CONVENTIONS_DATASET,
)
from datasets.scientist_conventions.summaries import (
    make_scientist_summaries,
)


CONTACTS = "contacts"
STORY_DAILY = STORY_DATASET
SCIENTIST_CONVENTIONS = SCIENTIST_CONVENTIONS_DATASET
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATHS = {
    CONTACTS: ROOT / "data" / "contacts" / "contacts.parquet",
    STORY_DAILY: (
        ROOT / "data" / "stories" / "processed" / "story_daily.parquet"
    ),
    SCIENTIST_CONVENTIONS: ROOT / "data" / "scientist_conventions",
}

SummaryFunction = Callable[[Mapping[str, ArrayLike]], ArrayLike]
Summaries = Mapping[str, SummaryFunction]


@dataclass(frozen=True)
class Observations:
    """Inference-ready conditions and simulation context for one dataset."""

    dataset: str
    context: Mapping[str, Any]
    summaries: Summaries
    conditions: Mapping[str, NDArray[Any]]
    observation_ids: NDArray[Any]

    @property
    def count(self) -> int:
        return len(self.observation_ids)


def condition_batches(
    conditions: Mapping[str, ArrayLike],
    batch_size: int,
) -> Iterator[dict[str, NDArray[Any]]]:
    """Yield aligned observation conditions in bounded batches."""

    if batch_size < 1:
        raise ValueError("observation batch size must be positive")
    arrays = {
        name: np.asarray(values) for name, values in conditions.items()
    }
    counts = {values.shape[0] for values in arrays.values()}
    if len(counts) != 1:
        raise ValueError("condition arrays must share an observation axis")
    count = counts.pop()
    if count < 1:
        raise ValueError("at least one observation is required")
    for start in range(0, count, batch_size):
        yield {
            name: values[start : start + batch_size]
            for name, values in arrays.items()
        }


def load_contacts(path: Path) -> tuple[ContactData, int, int]:
    """Load contacts and normalize IDs and times to model conventions."""

    frame = pd.read_parquet(path, columns=["t", "i", "j"])
    if frame.empty:
        raise ValueError("the observed contact data is empty")
    if frame[["t", "i", "j"]].isna().any().any():
        raise ValueError("contact columns must not contain null values")

    times = frame["t"].to_numpy()
    if np.any(times % INTERVAL_SECONDS):
        raise ValueError(
            f"contact times must fall on {INTERVAL_SECONDS}-second boundaries"
        )

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
    story_count: int = DEFAULT_STORY_SUMMARY_COUNT,
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
        name: values[None, ...]
        for name, values in compute_scalar_summaries(
            story_data,
            summaries,
            label="story summary",
        ).items()
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


def _network_edges(
    frame: pd.DataFrame,
    *,
    n_scientists: int,
    directed: bool,
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float64]]:
    required = {"source", "target", "weight"}
    missing = required.difference(frame.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"network edge data is missing columns: {names}")
    if frame[list(required)].isna().any().any():
        raise ValueError("network edge columns must not contain null values")
    source = frame["source"].to_numpy(dtype=np.int32)
    target = frame["target"].to_numpy(dtype=np.int32)
    weight = frame["weight"].to_numpy(dtype=np.float64)
    if (
        np.any(source < 0)
        or np.any(source >= n_scientists)
        or np.any(target < 0)
        or np.any(target >= n_scientists)
    ):
        raise ValueError("network edge contains an unknown scientist ID")
    if np.any(source == target):
        raise ValueError("network edges must not contain self-loops")
    if not directed and np.any(source >= target):
        raise ValueError("coauthorship edges must use source < target")
    if not np.all(np.isfinite(weight)) or np.any(weight <= 0.0):
        raise ValueError("network weights must be finite and positive")
    pairs = np.stack((source, target), axis=1)
    if len(pairs) != len(np.unique(pairs, axis=0)):
        raise ValueError("network edges must be unique")
    return source, target, weight


def scientist_convention_frame_observations(
    scientists: pd.DataFrame,
    coauthorship: pd.DataFrame,
    citations: pd.DataFrame,
) -> Observations:
    """Convert scientist attributes and two edge lists into one observation."""

    area_columns = [f"area_share_{index}" for index in range(4)]
    required = {
        "scientist_id",
        "favorite_convention",
        "primary_area",
        "career_start_year",
        *area_columns,
    }
    missing = required.difference(scientists.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"scientist data is missing columns: {names}")
    if scientists.empty:
        raise ValueError("the observed scientist data is empty")
    nonnullable = list(required.difference({"favorite_convention"}))
    if scientists[nonnullable].isna().any().any():
        raise ValueError("scientist covariates must not contain null values")

    scientist_ids = scientists["scientist_id"].to_numpy()
    n_scientists = len(scientists)
    if not np.array_equal(scientist_ids, np.arange(n_scientists)):
        raise ValueError("scientist_id must be ordered densely from zero")
    favorite = scientists["favorite_convention"]
    observed_mask = favorite.notna().to_numpy(dtype=np.bool_)
    observed_values = favorite.loc[observed_mask].to_numpy()
    if not np.all(np.isin(observed_values, (-1, 1))):
        raise ValueError("favorite_convention must be -1, +1, or null")
    preference = favorite.fillna(1).to_numpy(dtype=np.int8)
    primary_area = scientists["primary_area"].to_numpy(dtype=np.int8)
    area_shares = scientists[area_columns].to_numpy(dtype=np.float64)
    career_start_year = scientists["career_start_year"].to_numpy(dtype=np.int16)

    co_source, co_target, co_weight = _network_edges(
        coauthorship,
        n_scientists=n_scientists,
        directed=False,
    )
    if "first_year" not in coauthorship:
        raise ValueError("coauthorship data is missing column: first_year")
    if coauthorship["first_year"].isna().any():
        raise ValueError("coauthorship first_year must not contain null values")
    first_year = coauthorship["first_year"].to_numpy(dtype=np.int16)
    from scipy.sparse import csr_matrix

    coauthorship_matrix = csr_matrix(
        (
            np.concatenate((co_weight, co_weight)),
            (
                np.concatenate((co_source, co_target)),
                np.concatenate((co_target, co_source)),
            ),
        ),
        shape=(n_scientists, n_scientists),
    )
    citation_source, citation_target, citation_weight = _network_edges(
        citations,
        n_scientists=n_scientists,
        directed=True,
    )
    citation_matrix = csr_matrix(
        (citation_weight, (citation_source, citation_target)),
        shape=(n_scientists, n_scientists),
    )

    first_coauthor = np.full(n_scientists, -1, dtype=np.int32)
    candidates: list[list[tuple[int, int]]] = [
        [] for _ in range(n_scientists)
    ]
    for source, target, year in zip(
        co_source,
        co_target,
        first_year,
        strict=True,
    ):
        candidates[int(source)].append((int(year), int(target)))
        candidates[int(target)].append((int(year), int(source)))
    for scientist_id, scientist_candidates in enumerate(candidates):
        if scientist_candidates:
            first_coauthor[scientist_id] = min(scientist_candidates)[1]

    context: dict[str, Any] = {
        "n_scientists": n_scientists,
        "primary_area": primary_area,
        "area_shares": area_shares,
        "career_start_year": career_start_year,
        "observed_mask": observed_mask,
        "coauthorship": coauthorship_matrix,
        "citations": citation_matrix,
        "first_coauthor": first_coauthor,
        "start_year": CULTURAL_BASELINE_YEAR,
        "end_year": int(career_start_year.max()),
    }
    summaries = make_scientist_summaries(**context)
    conditions = {
        name: values[None, ...]
        for name, values in compute_scalar_summaries(
            {"preference": preference},
            summaries,
            label="scientist summary",
        ).items()
    }
    return Observations(
        dataset=SCIENTIST_CONVENTIONS,
        context=context,
        summaries=summaries,
        conditions=conditions,
        observation_ids=np.asarray([SCIENTIST_CONVENTIONS]),
    )


def scientist_convention_observations(path: Path) -> Observations:
    """Load the scientist convention Parquet directory."""

    return scientist_convention_frame_observations(
        pd.read_parquet(path / "scientists.parquet"),
        pd.read_parquet(path / "coauthorship.parquet"),
        pd.read_parquet(path / "citations.parquet"),
    )


OBSERVATION_LOADERS = {
    CONTACTS: contact_observations,
    STORY_DAILY: story_daily_observations,
    SCIENTIST_CONVENTIONS: scientist_convention_observations,
}


def load_observations(
    dataset: str,
    path: Path | None = None,
) -> Observations:
    """Load inference conditions using one dataset's native context."""

    try:
        loader = OBSERVATION_LOADERS[dataset]
    except KeyError as exc:
        choices = ", ".join(OBSERVATION_LOADERS)
        raise ValueError(
            f"unknown dataset {dataset!r}; available datasets: {choices}"
        ) from exc
    return loader(path or DEFAULT_DATA_PATHS[dataset])


__all__ = [
    "CONTACTS",
    "DEFAULT_DATA_PATHS",
    "OBSERVATION_LOADERS",
    "SCIENTIST_CONVENTIONS",
    "STORY_DAILY",
    "Observations",
    "condition_batches",
    "contact_observations",
    "load_contacts",
    "load_observations",
    "scientist_convention_frame_observations",
    "scientist_convention_observations",
    "story_daily_frame_observations",
    "story_daily_observations",
]
