"""Conversation model mixing friendship and reputation partner choice."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pymc as pm
from numpy.typing import ArrayLike, NDArray

from base.model import INTERVAL_SECONDS
from .base import ContactModel
from .reputation import (
    duration_in_steps,
    partner_probabilities,
    per_step_probability,
)


def zero_truncated_poisson(
    rng: np.random.Generator,
    rate: float,
) -> int:
    """Draw a positive Poisson variate efficiently, including near zero."""

    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("rate must be finite and positive")

    if rate >= 1.0:
        draw = int(rng.poisson(rate))
        while draw == 0:
            draw = int(rng.poisson(rate))
        return draw

    # Invert the conditional CDF when rejection would be slow. The first
    # conditional mass is P(X=1 | X>0) = rate / (exp(rate) - 1).
    target = float(rng.random())
    draw = 1
    probability = rate / np.expm1(rate)
    cumulative = probability
    while target > cumulative:
        draw += 1
        probability *= rate / draw
        if probability == 0.0:
            break
        cumulative += probability
    return draw


def draw_friend_network(
    rng: np.random.Generator,
    *,
    n_agents: int,
    phi: float,
    block_strength: float,
) -> tuple[int, NDArray[np.int32], NDArray[np.bool_]]:
    """Draw community identities and a simple undirected friend network."""

    if (
        isinstance(n_agents, (bool, np.bool_))
        or not isinstance(n_agents, (int, np.integer))
        or n_agents < 2
    ):
        raise ValueError("n_agents must be an integer of at least 2")
    n_agents = int(n_agents)
    if not np.isfinite(phi) or phi <= 0:
        raise ValueError("phi must be finite and positive")
    if not np.isfinite(block_strength) or not 0 <= block_strength <= 1:
        raise ValueError("block_strength must be between 0 and 1")

    n_communities = zero_truncated_poisson(rng, phi)
    communities = rng.integers(
        n_communities,
        size=n_agents,
        dtype=np.int32,
    )

    adjacency = np.zeros((n_agents, n_agents), dtype=np.bool_)
    first, second = np.triu_indices(n_agents, k=1)
    within = communities[first] == communities[second]
    within_first = first[within]
    within_second = second[within]
    edges = rng.random(within_first.size) < block_strength
    adjacency[within_first[edges], within_second[edges]] = True
    adjacency[within_second[edges], within_first[edges]] = True
    return n_communities, communities, adjacency


def choose_partner(
    rng: np.random.Generator,
    reputations: NDArray[np.floating],
    candidates: Sequence[int],
    friends: Sequence[int],
    p_friend: float,
) -> int | None:
    """Choose by friendship or reputation, allowing friendship to fail."""

    if not 0 <= p_friend <= 1:
        raise ValueError("p_friend must be between 0 and 1")

    candidate_list = list(candidates)
    if not candidate_list:
        raise ValueError("at least one candidate is required")

    candidate_set = set(candidate_list)
    available_friends = sorted(candidate_set.intersection(friends))
    if rng.random() < p_friend:
        if not available_friends:
            return None
        return int(rng.choice(available_friends))

    probabilities = partner_probabilities(reputations, candidate_list)
    return int(rng.choice(candidate_list, p=probabilities))


def _scalar_parameter(
    parameters: Mapping[str, NDArray[Any]],
    name: str,
) -> float:
    """Read a finite scalar simulation parameter."""

    value = np.asarray(parameters[name])
    if value.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


class FriendModel(ContactModel):
    """Agents prefer friends sometimes and reputable partners otherwise."""

    name = "friend_conversation"
    inference_variables = (
        "p_minute",
        "mean_duration_minutes",
        "reputation_sigma",
        "phi",
        "block_strength",
        "p_friend",
    )

    def build_prior(self, **context: Any) -> pm.Model:
        n_agents = context.get("n_agents")
        if (
            isinstance(n_agents, (bool, np.bool_))
            or not isinstance(n_agents, (int, np.integer))
            or n_agents < 2
        ):
            raise ValueError("n_agents must be an integer of at least 2")

        with pm.Model() as prior:
            pm.Beta("p_minute", alpha=1, beta=9)
            pm.LogNormal(
                "mean_duration_minutes",
                mu=np.log(5.0),
                sigma=0.5,
            )
            reputation_sigma = pm.Exponential(
                "reputation_sigma",
                lam=1.0,
            )
            pm.Normal(
                "reputation",
                mu=0,
                sigma=reputation_sigma,
                shape=int(n_agents),
            )
            pm.Exponential("phi", lam=0.1)
            pm.Uniform("block_strength", lower=0.0, upper=1.0)
            pm.Uniform("p_friend", lower=0.0, upper=1.0)
        return prior

    def simulate(
        self,
        parameters: Mapping[str, NDArray[Any]],
        rng: np.random.Generator,
        **context: Any,
    ) -> Mapping[str, ArrayLike]:
        n_agents = context.get("n_agents")
        n_steps = context.get("n_steps")
        if (
            isinstance(n_agents, (bool, np.bool_))
            or not isinstance(n_agents, (int, np.integer))
            or n_agents < 2
        ):
            raise ValueError("n_agents must be an integer of at least 2")
        if (
            isinstance(n_steps, (bool, np.bool_))
            or not isinstance(n_steps, (int, np.integer))
            or n_steps < 1
        ):
            raise ValueError("n_steps must be a positive integer")
        n_agents = int(n_agents)
        n_steps = int(n_steps)

        reputations = np.asarray(parameters["reputation"], dtype=float)
        if reputations.shape != (n_agents,) or not np.all(
            np.isfinite(reputations)
        ):
            raise ValueError(
                "reputation must be a finite vector with length n_agents"
            )

        p_minute = _scalar_parameter(parameters, "p_minute")
        mean_duration = _scalar_parameter(
            parameters,
            "mean_duration_minutes",
        )
        phi = _scalar_parameter(parameters, "phi")
        block_strength = _scalar_parameter(parameters, "block_strength")
        p_friend = _scalar_parameter(parameters, "p_friend")

        if not 0 <= p_minute <= 1:
            raise ValueError("p_minute must be between 0 and 1")
        if mean_duration <= 0:
            raise ValueError("mean_duration_minutes must be positive")
        if not 0 <= p_friend <= 1:
            raise ValueError("p_friend must be between 0 and 1")

        _, _, adjacency = draw_friend_network(
            rng,
            n_agents=n_agents,
            phi=phi,
            block_strength=block_strength,
        )
        p_step = per_step_probability(p_minute)

        # Conversations are (initiator, partner, exclusive end step).
        conversations: list[tuple[int, int, int]] = []
        times: list[int] = []
        first_agents: list[int] = []
        second_agents: list[int] = []

        for step in range(n_steps):
            conversations = [
                conversation
                for conversation in conversations
                if conversation[2] > step
            ]
            busy = {
                agent
                for first, second, _ in conversations
                for agent in (first, second)
            }
            available = set(range(n_agents)).difference(busy)

            for initiator in rng.permutation(tuple(available)):
                initiator = int(initiator)
                if initiator not in available or len(available) < 2:
                    continue
                if rng.random() >= p_step:
                    continue

                candidates = sorted(available.difference({initiator}))
                friends = np.flatnonzero(adjacency[initiator])
                partner = choose_partner(
                    rng,
                    reputations,
                    candidates,
                    friends,
                    p_friend,
                )
                if partner is None:
                    continue
                duration = duration_in_steps(
                    rng,
                    mean_duration,
                )
                conversations.append(
                    (initiator, partner, step + duration)
                )
                available.remove(initiator)
                available.remove(partner)

            time = (step + 1) * INTERVAL_SECONDS
            for first, second, _ in conversations:
                times.append(time)
                first_agents.append(first)
                second_agents.append(second)

        return {
            "t": np.asarray(times, dtype=np.int32),
            "i": np.asarray(first_agents, dtype=np.int32),
            "j": np.asarray(second_agents, dtype=np.int32),
        }


__all__ = [
    "FriendModel",
    "choose_partner",
    "draw_friend_network",
    "zero_truncated_poisson",
]
