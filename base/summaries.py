"""Reusable summary functions for temporal contact records."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import wraps
from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from base.model import ContactData, INTERVAL_SECONDS, validate_contacts


SummaryFunction = Callable[[ContactData], ArrayLike]
Summaries = Mapping[str, SummaryFunction]
SummaryBuilder = Callable[[int, int], SummaryFunction]


def compute_summaries(
    contacts: Mapping[str, ArrayLike],
    summaries: Summaries,
) -> dict[str, NDArray[Any]]:
    """Apply shared summary functions to native contact records."""

    contacts = validate_contacts(contacts)
    result = {
        name: np.atleast_1d(
            np.asarray(function(contacts), dtype=np.float32)
        )
        for name, function in summaries.items()
    }
    if any(not np.all(np.isfinite(value)) for value in result.values()):
        raise ValueError("summary statistics must be finite")
    return result


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


def sorted_summary(function: SummaryFunction) -> SummaryFunction:
    """Make a one-dimensional per-agent summary permutation invariant.

    Sorting preserves the empirical distribution while removing dependence on
    the order or labels of agents. It assumes a fixed population size; for
    variable populations, use quantiles or a BayesFlow set summary network.
    """

    @wraps(function)
    def sorted_function(contacts: ContactData) -> NDArray:
        values = np.asarray(function(contacts))
        if values.ndim != 1:
            raise ValueError("sorted summaries must be one-dimensional")
        return np.sort(values)

    return sorted_function


def contacts_per_bin(start: int, end: int) -> SummaryFunction:
    """Count contacts at each inclusive 20-second interval boundary."""

    if (
        start < INTERVAL_SECONDS
        or end < start
        or start % INTERVAL_SECONDS
        or end % INTERVAL_SECONDS
    ):
        raise ValueError(
            "start and end must be ordered positive 20-second boundaries"
        )

    bin_count = (end - start) // INTERVAL_SECONDS + 1

    def summary(contacts: ContactData) -> NDArray[np.int64]:
        times = np.asarray(contacts["t"])
        if np.any(times < start) or np.any(times > end):
            raise ValueError("contact times fall outside the summary range")
        indices = (times - start) // INTERVAL_SECONDS
        return np.bincount(indices, minlength=bin_count)

    return summary


def cumulative_contact_times(
    agent_ids: Sequence[int],
) -> SummaryFunction:
    """Return the sorted cumulative pair-contact time for every agent."""

    agents = _agent_sequence(agent_ids)
    positions = {int(agent): index for index, agent in enumerate(agents)}

    def per_agent(contacts: ContactData) -> NDArray[np.int64]:
        totals = np.zeros(len(agents), dtype=np.int64)
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
        return totals

    return sorted_summary(per_agent)


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


def _build_contacts_per_bin(
    _n_agents: int,
    n_steps: int,
) -> SummaryFunction:
    return contacts_per_bin(
        INTERVAL_SECONDS,
        n_steps * INTERVAL_SECONDS,
    )


def _build_cumulative_contact_times(
    n_agents: int,
    _n_steps: int,
) -> SummaryFunction:
    return cumulative_contact_times(range(n_agents))


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
    "contacts_per_bin": _build_contacts_per_bin,
    "cumulative_contact_times": _build_cumulative_contact_times,
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
    "INTERVAL_SECONDS",
    "SUMMARY_BUILDERS",
    "Summaries",
    "SummaryBuilder",
    "SummaryFunction",
    "compute_summaries",
    "contacts_per_bin",
    "cumulative_contact_times",
    "cumulative_network_assortativity",
    "cumulative_network_clustering",
    "cumulative_network_connectivity",
    "make_summaries",
    "sorted_summary",
]
