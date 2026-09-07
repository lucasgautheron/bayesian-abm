"""Summary statistics for pairwise temporal-contact data."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from base.model import ContactData, INTERVAL_SECONDS, validate_contacts
from base.summaries import (
    Summaries,
    SummaryFunction,
    compute_scalar_summaries,
    validate_summary_names,
)

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
    return cache if cache is not None and cache.contacts is contacts else None


def compute_summaries(
    contacts: Mapping[str, ArrayLike],
    summaries: Summaries,
) -> dict[str, NDArray]:
    """Validate contact records and compute finite scalar summaries."""

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
        raise ValueError("agent_ids must contain at least two unique agents")
    return agents


def _agent_positions(
    endpoint_ids: NDArray,
    agents: NDArray,
) -> NDArray[np.intp]:
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
    cache = _cache_for(contacts)
    key = np.asarray(agents).tobytes()
    if cache is not None and key in cache.adjacency:
        return cache.adjacency[key]

    adjacency = np.zeros((len(agents), len(agents)), dtype=np.bool_)
    first = _agent_positions(contacts["i"], agents)
    second = _agent_positions(contacts["j"], agents)
    if first.size:
        if np.any(first == second):
            raise ValueError("self-contacts are not valid network edges")
        adjacency[first, second] = True
        adjacency[second, first] = True
    if cache is not None:
        cache.adjacency[key] = adjacency
    return adjacency


def _mean_local_clustering(adjacency: NDArray[np.bool_]) -> float:
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
    seen = np.zeros(len(adjacency), dtype=np.bool_)
    largest: list[int] = []
    for start in range(len(adjacency)):
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
    nodes = _largest_component(adjacency)
    n_nodes = len(nodes)
    if n_nodes < 2:
        return 0.0
    index = np.asarray(nodes, dtype=np.intp)
    subgraph = np.ascontiguousarray(adjacency[np.ix_(index, index)])
    total = 0.0
    for start in range(n_nodes):
        seen = np.zeros(n_nodes, dtype=np.bool_)
        layer = np.zeros(n_nodes, dtype=np.bool_)
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
    return total / (n_nodes * (n_nodes - 1))


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
    if cache is not None and key in cache.bins:
        return cache.bins[key]
    times = np.asarray(contacts["t"])
    if np.any(times < start) or np.any(times > end):
        raise ValueError("contact times fall outside the summary range")
    indices = (times - start) // bin_seconds
    counts = np.bincount(
        indices[indices < bin_count],
        minlength=bin_count,
    )
    if cache is not None:
        cache.bins[key] = counts
    return counts


def mean_contacts_per_bin(start: int, end: int) -> SummaryFunction:
    """Return mean concurrent contacts across fixed time bins."""

    bin_count = _validate_bin_range(start, end)

    def summary(contacts: ContactData) -> float:
        return float(
            _contacts_per_bin(
                contacts,
                start=start,
                end=end,
                bin_count=bin_count,
            ).mean()
        )

    return summary


def integrated_contact_autocorrelation_time(
    start: int,
    end: int,
) -> SummaryFunction:
    """Return the initial-positive-sequence IACT of contact counts."""

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
    cache = _cache_for(contacts)
    key = (start, end)
    if cache is not None and key in cache.occupied:
        return cache.occupied[key]
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
        first, second, bins = first[order], second[order], bins[order]
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
        return float(np.diff(np.append(starts, len(occupied))).mean())

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
        return float(np.diff(np.append(starts, len(occupied))).mean())

    return summary


def contact_time_coefficient_of_variation(
    agent_ids: Sequence[int],
) -> SummaryFunction:
    """Return relative heterogeneity in cumulative agent contact time."""

    agents = _agent_sequence(agent_ids)

    def summary(contacts: ContactData) -> float:
        totals = np.zeros(len(agents), dtype=np.float64)
        np.add.at(totals, _agent_positions(contacts["i"], agents), INTERVAL_SECONDS)
        np.add.at(totals, _agent_positions(contacts["j"], agents), INTERVAL_SECONDS)
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
        return float(adjacency.sum()) / (len(agents) * (len(agents) - 1))

    return summary


def cumulative_network_clustering(
    agent_ids: Sequence[int],
) -> SummaryFunction:
    """Return mean local clustering of the cumulative contact network."""

    agents = _agent_sequence(agent_ids)
    return lambda contacts: _mean_local_clustering(
        _adjacency_matrix(contacts, agents)
    )


def cumulative_network_assortativity(
    agent_ids: Sequence[int],
) -> SummaryFunction:
    """Return degree assortativity of the cumulative contact network."""

    agents = _agent_sequence(agent_ids)

    def summary(contacts: ContactData) -> float:
        adjacency = _adjacency_matrix(contacts, agents)
        degrees = adjacency.sum(axis=1).astype(np.float64)
        first_index, second_index = np.triu_indices(len(agents), k=1)
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
    """Return mean shortest-path length in the largest component."""

    agents = _agent_sequence(agent_ids)
    return lambda contacts: _mean_shortest_path_length(
        _adjacency_matrix(contacts, agents)
    )


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


def _time_factory(
    factory: Callable[[int, int], SummaryFunction],
) -> SummaryBuilder:
    return lambda _n_agents, n_steps: factory(
        INTERVAL_SECONDS,
        n_steps * INTERVAL_SECONDS,
    )


def _agent_factory(
    factory: Callable[[Sequence[int]], SummaryFunction],
) -> SummaryBuilder:
    return lambda n_agents, _n_steps: factory(range(n_agents))


SUMMARY_BUILDERS: dict[str, SummaryBuilder] = {
    "mean_contacts_per_bin": _time_factory(mean_contacts_per_bin),
    "integrated_contact_autocorrelation_time": _time_factory(
        integrated_contact_autocorrelation_time
    ),
    "mean_contact_run_duration": _time_factory(mean_contact_run_duration),
    "mean_pair_contact_duration": _time_factory(mean_pair_contact_duration),
    "contact_time_coefficient_of_variation": _agent_factory(
        contact_time_coefficient_of_variation
    ),
    "cumulative_network_connectivity": _agent_factory(
        cumulative_network_connectivity
    ),
    "cumulative_network_clustering": _agent_factory(
        cumulative_network_clustering
    ),
    "cumulative_network_assortativity": _agent_factory(
        cumulative_network_assortativity
    ),
    "cumulative_network_average_path_length": _agent_factory(
        cumulative_network_average_path_length
    ),
    "cumulative_network_degree_coefficient_of_variation": _agent_factory(
        cumulative_network_degree_coefficient_of_variation
    ),
}


def make_summaries(
    n_agents: int,
    n_steps: int,
    *,
    summary_names: Sequence[str] | None = None,
) -> dict[str, SummaryFunction]:
    """Build selected registered contact statistics for a simulation context."""

    if n_agents < 2:
        raise ValueError("n_agents must be at least 2")
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    selected_names = (
        tuple(SUMMARY_BUILDERS)
        if summary_names is None
        else validate_summary_names(
            summary_names,
            SUMMARY_BUILDERS,
            label="contact summary",
        )
    )
    return {
        name: SUMMARY_BUILDERS[name](n_agents, n_steps)
        for name in selected_names
    }


__all__ = [
    "SUMMARY_BUILDERS",
    "SummaryBuilder",
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
