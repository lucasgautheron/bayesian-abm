"""Random walks over a latent network with probabilistic conversations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pymc as pm
from numpy.typing import ArrayLike, NDArray

from base.model import INTERVAL_SECONDS
from .base import ContactModel


def network_neighbors(
    network_edges: ArrayLike,
    n_agents: int,
) -> tuple[NDArray[np.int32], ...]:
    """Build neighbor arrays from upper-triangle Bernoulli edge indicators."""

    expected_edges = n_agents * (n_agents - 1) // 2
    edges = np.asarray(network_edges)
    if edges.shape != (expected_edges,):
        raise ValueError(
            "network_edges must contain one value per unordered agent pair"
        )
    if not np.all((edges == 0) | (edges == 1)):
        raise ValueError("network_edges must contain only zeroes and ones")

    neighbors: list[list[int]] = [[] for _ in range(n_agents)]
    first, second = np.triu_indices(n_agents, k=1)
    for left, right in zip(first[edges.astype(bool)], second[edges.astype(bool)]):
        neighbors[int(left)].append(int(right))
        neighbors[int(right)].append(int(left))
    return tuple(np.asarray(values, dtype=np.int32) for values in neighbors)


def simulate_walk_conversations(
    rng: np.random.Generator,
    *,
    neighbors: Sequence[NDArray[np.int32]],
    n_steps: int,
    p_move: float,
    p_conversation: float,
) -> dict[str, NDArray[np.int32]]:
    """Simulate walkers and return deduplicated undirected contacts."""

    n_agents = len(neighbors)
    walkers = np.arange(n_agents, dtype=np.int32)
    positions = walkers.copy()
    times: list[NDArray[np.int32]] = []
    first_agents: list[NDArray[np.int32]] = []
    second_agents: list[NDArray[np.int32]] = []

    for step in range(n_steps):
        # Each participant's walker independently stays or explores a neighbor.
        moving = np.flatnonzero(rng.random(n_agents) < p_move)
        for walker in moving:
            candidates = neighbors[int(positions[walker])]
            if candidates.size:
                positions[walker] = rng.choice(candidates)

        # Eligible walkers independently initiate one-step conversations.
        eligible = np.flatnonzero(positions != walkers)
        if eligible.size:
            successful = eligible[
                rng.random(eligible.size) < p_conversation
            ]
            if successful.size:
                pairs = np.column_stack(
                    (successful, positions[successful])
                ).astype(np.int32)
                pairs.sort(axis=1)
                pairs = np.unique(pairs, axis=0)
                count = len(pairs)
                times.append(
                    np.full(
                        count,
                        (step + 1) * INTERVAL_SECONDS,
                        dtype=np.int32,
                    )
                )
                first_agents.append(pairs[:, 0])
                second_agents.append(pairs[:, 1])

    if not times:
        empty = np.empty(0, dtype=np.int32)
        return {"t": empty, "i": empty, "j": empty}
    return {
        "t": np.concatenate(times),
        "i": np.concatenate(first_agents),
        "j": np.concatenate(second_agents),
    }


class RandomWalkModel(ContactModel):
    """Participants explore a latent random network and start conversations."""

    name = "random_walk"
    inference_variables = ("p_edge", "p_move", "p_conversation")
    parameter_units = {
        "p_move": "probability per minute",
        "p_conversation": "probability per minute",
    }

    def build_prior(self, **context: Any) -> pm.Model:
        n_agents = int(context["n_agents"])
        if n_agents < 2:
            raise ValueError("n_agents must be at least 2")
        n_edges = n_agents * (n_agents - 1) // 2

        with pm.Model() as prior:
            p_edge = pm.Beta("p_edge", alpha=2.0, beta=8.0)
            pm.Beta("p_move", alpha=2.0, beta=2.0)
            pm.Beta("p_conversation", alpha=1.0, beta=9.0)
            pm.Bernoulli("network_edges", p=p_edge, shape=n_edges)
        return prior

    def simulate(
        self,
        parameters: Mapping[str, NDArray[Any]],
        rng: np.random.Generator,
        **context: Any,
    ) -> Mapping[str, ArrayLike]:
        n_agents = int(context["n_agents"])
        n_steps = int(context["n_steps"])
        if n_agents < 2:
            raise ValueError("n_agents must be at least 2")
        if n_steps < 1:
            raise ValueError("n_steps must be positive")

        p_move = float(parameters["p_move"])
        p_conversation = float(parameters["p_conversation"])
        if not 0.0 <= p_move <= 1.0:
            raise ValueError("p_move must be between zero and one")
        if not 0.0 <= p_conversation <= 1.0:
            raise ValueError("p_conversation must be between zero and one")

        neighbors = network_neighbors(parameters["network_edges"], n_agents)
        return simulate_walk_conversations(
            rng,
            neighbors=neighbors,
            n_steps=n_steps,
            p_move=p_move,
            p_conversation=p_conversation,
        )


__all__ = [
    "RandomWalkModel",
    "network_neighbors",
    "simulate_walk_conversations",
]
