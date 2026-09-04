"""Reusable summary functions for temporal contact records."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from base.model import ContactData, INTERVAL_SECONDS, validate_contacts

HOUR_SECONDS = 60 * INTERVAL_SECONDS


SummaryFunction = Callable[[ContactData], ArrayLike]
Summaries = Mapping[str, SummaryFunction]
SummaryBuilder = Callable[[int, int], SummaryFunction]


def compute_scalar_summaries(
    data: Any,
    summaries: Mapping[str, Callable[[Any], ArrayLike]],
    *,
    label: str = "summary",
) -> dict[str, NDArray[Any]]:
    """Apply summary functions and enforce finite scalar outputs."""

    result = {
        name: np.atleast_1d(
            np.asarray(function(data), dtype=np.float32)
        )
        for name, function in summaries.items()
    }
    if any(value.size != 1 for value in result.values()):
        raise ValueError(f"{label} statistics must be scalar")
    if any(not np.all(np.isfinite(value)) for value in result.values()):
        raise ValueError(f"{label} statistics must be finite")
    return result


def compute_summaries(
    contacts: Mapping[str, ArrayLike],
    summaries: Summaries,
) -> dict[str, NDArray[Any]]:
    """Apply shared summary functions to native contact records."""

    return compute_scalar_summaries(
        validate_contacts(contacts),
        summaries,
    )


def _agent_sequence(agent_ids: Sequence[int]) -> NDArray:
    agents = np.asarray(agent_ids)
    if (
        agents.ndim != 1
        or len(agents) < 2
        or len(np.unique(agents)) != len(agents)
    ):
        raise ValueError(
            "agent_ids must contain at least two unique agents"
        )
    return agents


def _cumulative_neighbors(
    contacts: ContactData,
    agents: NDArray,
) -> list[set[int]]:
    positions = {int(agent): index for index, agent in enumerate(agents)}
    neighbors = [set() for _ in agents]
    for first, second in zip(contacts["i"], contacts["j"]):
        try:
            first_position = positions[int(first)]
            second_position = positions[int(second)]
        except KeyError as exc:
            raise ValueError(
                f"contact contains unknown agent ID {exc.args[0]}"
            ) from exc
        if first_position == second_position:
            raise ValueError("self-contacts are not valid network edges")
        neighbors[first_position].add(second_position)
        neighbors[second_position].add(first_position)
    return neighbors


def _validate_bin_range(
    start: int,
    end: int,
    bin_seconds: int = INTERVAL_SECONDS,
) -> int:
    if (
        start < INTERVAL_SECONDS
        or end < start
        or start % INTERVAL_SECONDS
        or end % INTERVAL_SECONDS
    ):
        raise ValueError(
            "start and end must be ordered positive "
            f"{INTERVAL_SECONDS}-second boundaries"
        )
    if bin_seconds < INTERVAL_SECONDS or bin_seconds % INTERVAL_SECONDS:
        raise ValueError(
            "bin_seconds must be a positive multiple of "
            f"{INTERVAL_SECONDS}"
        )
    return (end - start + INTERVAL_SECONDS) // bin_seconds


def _contacts_per_bin(
    contacts: ContactData,
    *,
    start: int,
    end: int,
    bin_count: int,
    bin_seconds: int = INTERVAL_SECONDS,
) -> NDArray[np.int64]:
    times = np.asarray(contacts["t"])
    if np.any(times < start) or np.any(times > end):
        raise ValueError("contact times fall outside the summary range")
    if bin_count < 1:
        return np.zeros(0, dtype=np.int64)
    indices = (times - start) // bin_seconds
    in_range = indices < bin_count
    return np.bincount(indices[in_range], minlength=bin_count)


def mean_contacts_per_bin(start: int, end: int) -> SummaryFunction:
    """Return mean concurrent contacts across fixed time bins."""

    bin_count = _validate_bin_range(start, end)

    def summary(contacts: ContactData) -> float:
        counts = _contacts_per_bin(
            contacts,
            start=start,
            end=end,
            bin_count=bin_count,
        )
        return float(counts.mean())

    return summary


def lag_one_contact_autocorrelation(
    start: int,
    end: int,
    *,
    bin_seconds: int = INTERVAL_SECONDS,
) -> SummaryFunction:
    """Return lag-one correlation of contact counts in fixed time bins."""

    bin_count = _validate_bin_range(start, end, bin_seconds)

    def summary(contacts: ContactData) -> float:
        counts = _contacts_per_bin(
            contacts,
            start=start,
            end=end,
            bin_count=bin_count,
            bin_seconds=bin_seconds,
        ).astype(np.float64)
        if len(counts) < 2:
            return 0.0
        first = counts[:-1] - counts[:-1].mean()
        second = counts[1:] - counts[1:].mean()
        denominator = np.linalg.norm(first) * np.linalg.norm(second)
        return (
            0.0
            if np.isclose(denominator, 0.0)
            else float(np.dot(first, second) / denominator)
        )

    return summary


def lag_one_hourly_contact_autocorrelation(
    start: int,
    end: int,
) -> SummaryFunction:
    """Return lag-one correlation of hourly contact counts.

    Incomplete trailing hours are dropped so every bin covers the same
    duration.
    """

    return lag_one_contact_autocorrelation(
        start,
        end,
        bin_seconds=HOUR_SECONDS,
    )


def integrated_contact_autocorrelation_time(
    start: int,
    end: int,
) -> SummaryFunction:
    """Return the IACT of per-minute contact counts.

    Uses ``τ = 1 + 2 ∑_{k=1}^{K} ρ(k)`` with Geyer's initial-positive
    sequence: ``K`` is the first lag where the ACF is non-positive.
    Constant or too-short series return 0.
    """

    bin_count = _validate_bin_range(start, end)

    def summary(contacts: ContactData) -> float:
        counts = _contacts_per_bin(
            contacts,
            start=start,
            end=end,
            bin_count=bin_count,
        ).astype(np.float64)
        if len(counts) < 2:
            return 0.0
        centered = counts - counts.mean()
        variance = float(np.dot(centered, centered))
        if np.isclose(variance, 0.0):
            return 0.0
        total = 0.0
        for lag in range(1, len(counts)):
            rho = float(np.dot(centered[:-lag], centered[lag:]) / variance)
            if rho <= 0.0:
                break
            total += rho
        return 1.0 + 2.0 * total

    return summary


def _occupied_pair_bins(
    contacts: ContactData,
    *,
    start: int,
    end: int,
) -> NDArray[np.int64]:
    """Return unique ``(i, j, bin)`` rows with ``i < j``."""

    times = np.asarray(contacts["t"])
    if np.any(times < start) or np.any(times > end):
        raise ValueError("contact times fall outside the summary range")
    first = np.minimum(contacts["i"], contacts["j"]).astype(np.int64)
    second = np.maximum(contacts["i"], contacts["j"]).astype(np.int64)
    if np.any(first == second):
        raise ValueError("self-contacts are not valid network edges")
    bins = (times - start) // INTERVAL_SECONDS
    if not first.size:
        return np.empty((0, 3), dtype=np.int64)
    return np.unique(np.stack((first, second, bins), axis=1), axis=0)


def mean_contact_run_duration(start: int, end: int) -> SummaryFunction:
    """Return mean consecutive minutes a pair stays in contact."""

    _validate_bin_range(start, end)

    def summary(contacts: ContactData) -> float:
        occupied = _occupied_pair_bins(contacts, start=start, end=end)
        if not occupied.size:
            return 0.0
        new_run = np.empty(len(occupied), dtype=np.bool_)
        new_run[0] = True
        new_run[1:] = (
            (occupied[1:, 0] != occupied[:-1, 0])
            | (occupied[1:, 1] != occupied[:-1, 1])
            | (occupied[1:, 2] != occupied[:-1, 2] + 1)
        )
        starts = np.flatnonzero(new_run)
        lengths = np.diff(np.append(starts, len(occupied)))
        return float(lengths.mean())

    return summary


def mean_pair_contact_duration(start: int, end: int) -> SummaryFunction:
    """Return mean total minutes of contact among pairs that ever meet."""

    _validate_bin_range(start, end)

    def summary(contacts: ContactData) -> float:
        occupied = _occupied_pair_bins(contacts, start=start, end=end)
        if not occupied.size:
            return 0.0
        _, counts = np.unique(occupied[:, :2], axis=0, return_counts=True)
        return float(counts.mean())

    return summary


def contact_time_coefficient_of_variation(
    agent_ids: Sequence[int],
) -> SummaryFunction:
    """Return relative heterogeneity in cumulative agent contact time."""

    agents = _agent_sequence(agent_ids)
    positions = {int(agent): index for index, agent in enumerate(agents)}

    def summary(contacts: ContactData) -> float:
        totals = np.zeros(len(agents), dtype=np.float64)
        endpoints = np.concatenate((contacts["i"], contacts["j"]))
        try:
            endpoint_positions = np.fromiter(
                (positions[int(agent)] for agent in endpoints),
                dtype=np.int64,
                count=len(endpoints),
            )
        except KeyError as exc:
            raise ValueError(
                f"contact contains unknown agent ID {exc.args[0]}"
            ) from exc
        np.add.at(totals, endpoint_positions, INTERVAL_SECONDS)
        mean = totals.mean()
        return 0.0 if np.isclose(mean, 0.0) else float(totals.std() / mean)

    return summary

def cumulative_network_giant_component(
    agent_ids: Sequence[int],
) -> SummaryFunction:
    """Return the fraction of agents in the largest connected component."""

    agents = _agent_sequence(agent_ids)

    def summary(contacts: ContactData) -> float:
        neighbors = _cumulative_neighbors(contacts, agents)
        seen = [False] * len(agents)
        giant = 0
        for start in range(len(agents)):
            if seen[start]:
                continue
            size = 0
            stack = [start]
            seen[start] = True
            while stack:
                node = stack.pop()
                size += 1
                for adjacent in neighbors[node]:
                    if not seen[adjacent]:
                        seen[adjacent] = True
                        stack.append(adjacent)
            giant = max(giant, size)
        return giant / len(agents)

    return summary


def cumulative_network_connectivity(
    agent_ids: Sequence[int],
) -> SummaryFunction:
    """Return the density of the cumulative binary contact network."""

    agents = _agent_sequence(agent_ids)

    def summary(contacts: ContactData) -> float:
        neighbors = _cumulative_neighbors(contacts, agents)
        degree_sum = sum(len(adjacent) for adjacent in neighbors)
        return degree_sum / (len(agents) * (len(agents) - 1))

    return summary


def cumulative_network_clustering(
    agent_ids: Sequence[int],
) -> SummaryFunction:
    """Return the mean local clustering of the cumulative contact network."""

    agents = _agent_sequence(agent_ids)

    def summary(contacts: ContactData) -> float:
        neighbors = _cumulative_neighbors(contacts, agents)
        coefficients = np.zeros(len(agents), dtype=np.float64)
        for agent, adjacent in enumerate(neighbors):
            degree = len(adjacent)
            if degree < 2:
                continue
            neighbor_edges = sum(
                second in neighbors[first]
                for first, second in combinations(adjacent, 2)
            )
            coefficients[agent] = (
                2.0 * neighbor_edges / (degree * (degree - 1))
            )
        return float(coefficients.mean())

    return summary


def cumulative_network_assortativity(
    agent_ids: Sequence[int],
) -> SummaryFunction:
    """Return degree assortativity of the cumulative contact network.

    Networks with no edges or no degree variation have an undefined Pearson
    correlation and are assigned zero to keep the summary finite.
    """

    agents = _agent_sequence(agent_ids)

    def summary(contacts: ContactData) -> float:
        neighbors = _cumulative_neighbors(contacts, agents)
        degrees = np.asarray(
            [len(adjacent) for adjacent in neighbors],
            dtype=np.float64,
        )
        degree_pairs = np.asarray(
            [
                (degrees[first], degrees[second])
                for first, adjacent in enumerate(neighbors)
                for second in adjacent
                if first < second
            ],
            dtype=np.float64,
        )
        if len(degree_pairs) == 0:
            return 0.0

        first, second = degree_pairs.T
        endpoint_mean = np.mean(np.concatenate((first, second)))
        covariance = np.mean(first * second) - endpoint_mean**2
        variance = (
            np.mean(np.concatenate((first**2, second**2)))
            - endpoint_mean**2
        )
        return 0.0 if np.isclose(variance, 0.0) else covariance / variance

    return summary


def _build_mean_contacts_per_bin(
    _n_agents: int,
    n_steps: int,
) -> SummaryFunction:
    return mean_contacts_per_bin(
        INTERVAL_SECONDS,
        n_steps * INTERVAL_SECONDS,
    )


def _build_lag_one_contact_autocorrelation(
    _n_agents: int,
    n_steps: int,
) -> SummaryFunction:
    return lag_one_contact_autocorrelation(
        INTERVAL_SECONDS,
        n_steps * INTERVAL_SECONDS,
    )


def _build_lag_one_hourly_contact_autocorrelation(
    _n_agents: int,
    n_steps: int,
) -> SummaryFunction:
    return lag_one_hourly_contact_autocorrelation(
        INTERVAL_SECONDS,
        n_steps * INTERVAL_SECONDS,
    )


def _build_integrated_contact_autocorrelation_time(
    _n_agents: int,
    n_steps: int,
) -> SummaryFunction:
    return integrated_contact_autocorrelation_time(
        INTERVAL_SECONDS,
        n_steps * INTERVAL_SECONDS,
    )


def _build_mean_contact_run_duration(
    _n_agents: int,
    n_steps: int,
) -> SummaryFunction:
    return mean_contact_run_duration(
        INTERVAL_SECONDS,
        n_steps * INTERVAL_SECONDS,
    )


def _build_mean_pair_contact_duration(
    _n_agents: int,
    n_steps: int,
) -> SummaryFunction:
    return mean_pair_contact_duration(
        INTERVAL_SECONDS,
        n_steps * INTERVAL_SECONDS,
    )


def _build_contact_time_coefficient_of_variation(
    n_agents: int,
    _n_steps: int,
) -> SummaryFunction:
    return contact_time_coefficient_of_variation(range(n_agents))


def _build_cumulative_network_giant_component(
    n_agents: int,
    _n_steps: int,
) -> SummaryFunction:
    return cumulative_network_giant_component(range(n_agents))


def _build_cumulative_network_connectivity(
    n_agents: int,
    _n_steps: int,
) -> SummaryFunction:
    return cumulative_network_connectivity(range(n_agents))


def _build_cumulative_network_clustering(
    n_agents: int,
    _n_steps: int,
) -> SummaryFunction:
    return cumulative_network_clustering(range(n_agents))


def _build_cumulative_network_assortativity(
    n_agents: int,
    _n_steps: int,
) -> SummaryFunction:
    return cumulative_network_assortativity(range(n_agents))


SUMMARY_BUILDERS: dict[str, SummaryBuilder] = {
    "mean_contacts_per_bin": _build_mean_contacts_per_bin,
    "lag_one_contact_autocorrelation": (
        _build_lag_one_contact_autocorrelation
    ),
    "lag_one_hourly_contact_autocorrelation": (
        _build_lag_one_hourly_contact_autocorrelation
    ),
    "integrated_contact_autocorrelation_time": (
        _build_integrated_contact_autocorrelation_time
    ),
    "mean_contact_run_duration": _build_mean_contact_run_duration,
    "mean_pair_contact_duration": _build_mean_pair_contact_duration,
    "contact_time_coefficient_of_variation": (
        _build_contact_time_coefficient_of_variation
    ),
    "cumulative_network_giant_component": (
        _build_cumulative_network_giant_component
    ),
    "cumulative_network_connectivity": (
        _build_cumulative_network_connectivity
    ),
    "cumulative_network_clustering": _build_cumulative_network_clustering,
    "cumulative_network_assortativity": (
        _build_cumulative_network_assortativity
    ),
}


def make_summaries(
    n_agents: int,
    n_steps: int,
) -> dict[str, SummaryFunction]:
    """Build every registered statistic for one simulation context."""

    if n_agents < 2:
        raise ValueError("n_agents must be at least 2")
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    return {
        name: builder(n_agents, n_steps)
        for name, builder in SUMMARY_BUILDERS.items()
    }


__all__ = [
    "HOUR_SECONDS",
    "INTERVAL_SECONDS",
    "SUMMARY_BUILDERS",
    "Summaries",
    "SummaryBuilder",
    "SummaryFunction",
    "compute_scalar_summaries",
    "compute_summaries",
    "contact_time_coefficient_of_variation",
    "cumulative_network_assortativity",
    "cumulative_network_clustering",
    "cumulative_network_connectivity",
    "cumulative_network_giant_component",
    "integrated_contact_autocorrelation_time",
    "lag_one_contact_autocorrelation",
    "lag_one_hourly_contact_autocorrelation",
    "make_summaries",
    "mean_contact_run_duration",
    "mean_contacts_per_bin",
    "mean_pair_contact_duration",
]
