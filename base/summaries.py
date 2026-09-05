"""Reusable summary functions for temporal contact records."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from base.model import ContactData, INTERVAL_SECONDS, validate_contacts

SummaryFunction = Callable[[ContactData], ArrayLike]
Summaries = Mapping[str, SummaryFunction]
SummaryBuilder = Callable[[int, int], SummaryFunction]


@dataclass
class _SummaryCache:
    contacts: ContactData
    adjacency: dict[bytes, NDArray[np.bool_]] = field(default_factory=dict)
    occupied: dict[tuple[int, int], NDArray[np.int64]] = field(
        default_factory=dict
    )
    bins: dict[tuple[int, int, int, int], NDArray[np.int64]] = field(
        default_factory=dict
    )


_SUMMARY_CACHE: ContextVar[_SummaryCache | None] = ContextVar(
    "_SUMMARY_CACHE",
    default=None,
)


def _cache_for(contacts: Mapping[str, ArrayLike]) -> _SummaryCache | None:
    cache = _SUMMARY_CACHE.get()
    if cache is not None and cache.contacts is contacts:
        return cache
    return None


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

    data = validate_contacts(contacts)
    token = _SUMMARY_CACHE.set(_SummaryCache(contacts=data))
    try:
        return compute_scalar_summaries(data, summaries)
    finally:
        _SUMMARY_CACHE.reset(token)


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


def _agent_positions(
    endpoint_ids: NDArray,
    agents: NDArray,
) -> NDArray[np.intp]:
    """Map contact endpoints onto agent-row indices."""

    ids = np.asarray(endpoint_ids)
    if ids.size == 0:
        return np.empty(0, dtype=np.intp)
    n_agents = int(agents.size)
    if (
        agents[0] == 0
        and agents[-1] == n_agents - 1
        and np.array_equal(agents, np.arange(n_agents))
    ):
        positions = np.asarray(ids)
        unknown = (positions < 0) | (positions >= n_agents)
        if np.any(unknown):
            raise ValueError(
                f"contact contains unknown agent ID {int(positions[unknown][0])}"
            )
        return np.asarray(positions, dtype=np.intp)
    sorted_index = np.argsort(agents, kind="stable")
    sorted_ids = agents[sorted_index]
    slots = np.searchsorted(sorted_ids, ids)
    in_range = slots < n_agents
    safe_slots = np.where(in_range, slots, 0)
    matched = in_range & (sorted_ids[safe_slots] == ids)
    if not np.all(matched):
        raise ValueError(
            f"contact contains unknown agent ID {int(ids[~matched][0])}"
        )
    return np.asarray(sorted_index[slots], dtype=np.intp)


def _adjacency_matrix(
    contacts: ContactData,
    agents: NDArray,
) -> NDArray[np.bool_]:
    """Return the undirected union graph as a symmetric adjacency matrix."""

    cache = _cache_for(contacts)
    agent_key = np.asarray(agents).tobytes()
    if cache is not None:
        cached = cache.adjacency.get(agent_key)
        if cached is not None:
            return cached

    n_agents = int(agents.size)
    adjacency = np.zeros((n_agents, n_agents), dtype=np.bool_)
    first = _agent_positions(contacts["i"], agents)
    second = _agent_positions(contacts["j"], agents)
    if first.size:
        if np.any(first == second):
            raise ValueError("self-contacts are not valid network edges")
        adjacency[first, second] = True
        adjacency[second, first] = True
    if cache is not None:
        cache.adjacency[agent_key] = adjacency
    return adjacency


def _mean_local_clustering(adjacency: NDArray[np.bool_]) -> float:
    """Return mean Watts–Strogatz local clustering from an adjacency matrix."""

    weights = adjacency.astype(np.float64, copy=False)
    degrees = weights.sum(axis=1)
    triangles = np.sum((weights @ weights) * weights, axis=1)
    coefficients = np.zeros(degrees.size, dtype=np.float64)
    connected = degrees >= 2.0
    coefficients[connected] = triangles[connected] / (
        degrees[connected] * (degrees[connected] - 1.0)
    )
    return float(coefficients.mean())


def _largest_component(adjacency: NDArray[np.bool_]) -> list[int]:
    """Return node indices in the largest connected component."""

    n_agents = int(adjacency.shape[0])
    seen = np.zeros(n_agents, dtype=np.bool_)
    largest: list[int] = []
    for start in range(n_agents):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        component = [start]
        while stack:
            node = stack.pop()
            for adjacent in np.flatnonzero(adjacency[node]):
                if not seen[adjacent]:
                    seen[adjacent] = True
                    stack.append(int(adjacent))
                    component.append(int(adjacent))
        if len(component) > len(largest):
            largest = component
    return largest


def _mean_shortest_path_length(adjacency: NDArray[np.bool_]) -> float:
    """Return mean geodesic length inside the largest component."""

    nodes = _largest_component(adjacency)
    n = len(nodes)
    if n < 2:
        return 0.0
    index = np.asarray(nodes, dtype=np.intp)
    subgraph = np.ascontiguousarray(adjacency[np.ix_(index, index)])
    total = 0.0
    for start in range(n):
        seen = np.zeros(n, dtype=np.bool_)
        layer = np.zeros(n, dtype=np.bool_)
        layer[start] = True
        seen[start] = True
        distance = 0
        while True:
            nxt = subgraph[layer].any(axis=0)
            nxt &= ~seen
            if not nxt.any():
                break
            distance += 1
            total += distance * int(nxt.sum())
            seen |= nxt
            layer = nxt
    return total / (n * (n - 1))


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
    cache = _cache_for(contacts)
    key = (start, end, bin_count, bin_seconds)
    if cache is not None:
        cached = cache.bins.get(key)
        if cached is not None:
            return cached
    times = np.asarray(contacts["t"])
    if np.any(times < start) or np.any(times > end):
        raise ValueError("contact times fall outside the summary range")
    if bin_count < 1:
        counts = np.zeros(0, dtype=np.int64)
    else:
        indices = (times - start) // bin_seconds
        in_range = indices < bin_count
        counts = np.bincount(indices[in_range], minlength=bin_count)
    if cache is not None:
        cache.bins[key] = counts
    return counts


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

    cache = _cache_for(contacts)
    key = (start, end)
    if cache is not None:
        cached = cache.occupied.get(key)
        if cached is not None:
            return cached
    times = np.asarray(contacts["t"])
    if np.any(times < start) or np.any(times > end):
        raise ValueError("contact times fall outside the summary range")
    first = np.minimum(contacts["i"], contacts["j"]).astype(np.int64, copy=False)
    second = np.maximum(contacts["i"], contacts["j"]).astype(np.int64, copy=False)
    if np.any(first == second):
        raise ValueError("self-contacts are not valid network edges")
    bins = (times - start) // INTERVAL_SECONDS
    if not first.size:
        occupied = np.empty((0, 3), dtype=np.int64)
    else:
        order = np.lexsort((bins, second, first))
        first = first[order]
        second = second[order]
        bins = bins[order]
        keep = np.empty(first.size, dtype=np.bool_)
        keep[0] = True
        keep[1:] = (
            (first[1:] != first[:-1])
            | (second[1:] != second[:-1])
            | (bins[1:] != bins[:-1])
        )
        occupied = np.stack((first[keep], second[keep], bins[keep]), axis=1)
    if cache is not None:
        cache.occupied[key] = occupied
    return occupied


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
        new_pair = np.empty(len(occupied), dtype=np.bool_)
        new_pair[0] = True
        new_pair[1:] = (occupied[1:, 0] != occupied[:-1, 0]) | (
            occupied[1:, 1] != occupied[:-1, 1]
        )
        starts = np.flatnonzero(new_pair)
        counts = np.diff(np.append(starts, len(occupied)))
        return float(counts.mean())

    return summary


def contact_time_coefficient_of_variation(
    agent_ids: Sequence[int],
) -> SummaryFunction:
    """Return relative heterogeneity in cumulative agent contact time."""

    agents = _agent_sequence(agent_ids)

    def summary(contacts: ContactData) -> float:
        first = _agent_positions(contacts["i"], agents)
        second = _agent_positions(contacts["j"], agents)
        totals = np.zeros(len(agents), dtype=np.float64)
        np.add.at(totals, first, INTERVAL_SECONDS)
        np.add.at(totals, second, INTERVAL_SECONDS)
        mean = totals.mean()
        return 0.0 if np.isclose(mean, 0.0) else float(totals.std() / mean)

    return summary


def cumulative_network_connectivity(
    agent_ids: Sequence[int],
) -> SummaryFunction:
    """Return the density of the cumulative binary contact network."""

    agents = _agent_sequence(agent_ids)

    def summary(contacts: ContactData) -> float:
        adjacency = _adjacency_matrix(contacts, agents)
        n_agents = int(adjacency.shape[0])
        return float(adjacency.sum()) / (n_agents * (n_agents - 1))

    return summary


def cumulative_network_clustering(
    agent_ids: Sequence[int],
) -> SummaryFunction:
    """Return the mean local clustering of the cumulative contact network."""

    agents = _agent_sequence(agent_ids)

    def summary(contacts: ContactData) -> float:
        return _mean_local_clustering(_adjacency_matrix(contacts, agents))

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
        adjacency = _adjacency_matrix(contacts, agents)
        degrees = adjacency.sum(axis=1).astype(np.float64)
        first_index, second_index = np.triu_indices(adjacency.shape[0], k=1)
        keep = adjacency[first_index, second_index]
        if not np.any(keep):
            return 0.0
        first = degrees[first_index[keep]]
        second = degrees[second_index[keep]]
        endpoints = np.concatenate((first, second))
        endpoint_mean = float(endpoints.mean())
        covariance = float(np.mean(first * second) - endpoint_mean**2)
        variance = float(np.mean(endpoints**2) - endpoint_mean**2)
        return 0.0 if np.isclose(variance, 0.0) else covariance / variance

    return summary


def cumulative_network_average_path_length(
    agent_ids: Sequence[int],
) -> SummaryFunction:
    """Return mean shortest-path length in the largest component.

    Isolated nodes and smaller components are ignored. Graphs whose largest
    component has fewer than two nodes have no pairs and return 0.
    """

    agents = _agent_sequence(agent_ids)

    def summary(contacts: ContactData) -> float:
        return _mean_shortest_path_length(_adjacency_matrix(contacts, agents))

    return summary


def cumulative_network_degree_coefficient_of_variation(
    agent_ids: Sequence[int],
) -> SummaryFunction:
    """Return the CV of unique-neighbor degrees, including isolates."""

    agents = _agent_sequence(agent_ids)

    def summary(contacts: ContactData) -> float:
        degrees = _adjacency_matrix(contacts, agents).sum(axis=1)
        mean = float(degrees.mean())
        return 0.0 if np.isclose(mean, 0.0) else float(degrees.std() / mean)

    return summary


def _build_mean_contacts_per_bin(
    _n_agents: int,
    n_steps: int,
) -> SummaryFunction:
    return mean_contacts_per_bin(
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


def _build_cumulative_network_average_path_length(
    n_agents: int,
    _n_steps: int,
) -> SummaryFunction:
    return cumulative_network_average_path_length(range(n_agents))


def _build_cumulative_network_degree_coefficient_of_variation(
    n_agents: int,
    _n_steps: int,
) -> SummaryFunction:
    return cumulative_network_degree_coefficient_of_variation(range(n_agents))


SUMMARY_BUILDERS: dict[str, SummaryBuilder] = {
    "mean_contacts_per_bin": _build_mean_contacts_per_bin,
    "integrated_contact_autocorrelation_time": (
        _build_integrated_contact_autocorrelation_time
    ),
    "mean_contact_run_duration": _build_mean_contact_run_duration,
    "mean_pair_contact_duration": _build_mean_pair_contact_duration,
    "contact_time_coefficient_of_variation": (
        _build_contact_time_coefficient_of_variation
    ),
    "cumulative_network_connectivity": (
        _build_cumulative_network_connectivity
    ),
    "cumulative_network_clustering": _build_cumulative_network_clustering,
    "cumulative_network_assortativity": (
        _build_cumulative_network_assortativity
    ),
    "cumulative_network_average_path_length": (
        _build_cumulative_network_average_path_length
    ),
    "cumulative_network_degree_coefficient_of_variation": (
        _build_cumulative_network_degree_coefficient_of_variation
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
    "INTERVAL_SECONDS",
    "SUMMARY_BUILDERS",
    "Summaries",
    "SummaryBuilder",
    "SummaryFunction",
    "compute_scalar_summaries",
    "compute_summaries",
    "contact_time_coefficient_of_variation",
    "cumulative_network_assortativity",
    "cumulative_network_average_path_length",
    "cumulative_network_clustering",
    "cumulative_network_connectivity",
    "cumulative_network_degree_coefficient_of_variation",
    "integrated_contact_autocorrelation_time",
    "make_summaries",
    "mean_contact_run_duration",
    "mean_contacts_per_bin",
    "mean_pair_contact_duration",
]
