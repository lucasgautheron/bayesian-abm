"""Conversation model with reputation-weighted partner choice."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pymc as pm
from numpy.typing import ArrayLike, NDArray

from base.model import INTERVAL_SECONDS
from .base import ContactModel


STEPS_PER_MINUTE = 60 // INTERVAL_SECONDS


def per_step_probability(p_minute: float) -> float:
    """Convert a per-minute probability to one simulation step."""

    return 1.0 - (1.0 - p_minute) ** (1.0 / STEPS_PER_MINUTE)


def partner_probabilities(
    reputations: NDArray[np.floating],
    candidates: Sequence[int],
) -> NDArray[np.float64]:
    """Return stable softmax probabilities over candidate reputations."""

    logits = np.asarray(reputations)[np.asarray(candidates)]
    weights = np.exp(logits - logits.max())
    return weights / weights.sum()


def duration_in_steps(
    rng: np.random.Generator,
    mean_minutes: float,
) -> int:
    """Draw an exponential duration and round it up to simulation steps."""

    duration_minutes = rng.exponential(scale=mean_minutes)
    return max(1, int(np.ceil(duration_minutes * STEPS_PER_MINUTE)))


def simulate_exclusive_conversations(
    rng: np.random.Generator,
    *,
    n_agents: int,
    n_steps: int,
    p_step: float,
    mean_duration: float,
    choose_partner,
) -> dict[str, NDArray[np.int32]]:
    """Run exclusive conversations and emit one contact row per active step."""

    busy_until = np.zeros(n_agents, dtype=np.int64)
    conversations: list[tuple[int, int, int]] = []
    times: list[NDArray[np.int32]] = []
    first_agents: list[NDArray[np.int32]] = []
    second_agents: list[NDArray[np.int32]] = []

    for step in range(n_steps):
        if conversations:
            conversations = [
                conversation
                for conversation in conversations
                if conversation[2] > step
            ]

        available = np.flatnonzero(busy_until <= step)
        n_free = int(available.size)
        if n_free >= 2:
            for initiator in rng.permutation(available):
                initiator = int(initiator)
                if busy_until[initiator] > step:
                    continue
                if n_free < 2:
                    break
                if rng.random() >= p_step:
                    continue

                candidates = np.flatnonzero(busy_until <= step)
                candidates = candidates[candidates != initiator]
                partner = choose_partner(initiator, candidates)
                if partner is None:
                    continue
                partner = int(partner)
                duration = duration_in_steps(rng, mean_duration)
                end = step + duration
                busy_until[initiator] = end
                busy_until[partner] = end
                conversations.append((initiator, partner, end))
                n_free -= 2

        if conversations:
            time = np.int32((step + 1) * INTERVAL_SECONDS)
            times.append(np.full(len(conversations), time, dtype=np.int32))
            first_agents.append(
                np.fromiter(
                    (first for first, _, _ in conversations),
                    dtype=np.int32,
                    count=len(conversations),
                )
            )
            second_agents.append(
                np.fromiter(
                    (second for _, second, _ in conversations),
                    dtype=np.int32,
                    count=len(conversations),
                )
            )

    if not times:
        empty = np.empty(0, dtype=np.int32)
        return {"t": empty, "i": empty, "j": empty}
    return {
        "t": np.concatenate(times),
        "i": np.concatenate(first_agents),
        "j": np.concatenate(second_agents),
    }


class ReputationConversationModel(ContactModel):
    """Agents start conversations and prefer reputable available partners."""

    name = "reputation_conversation"
    inference_variables = (
        "p_minute",
        "mean_duration_minutes",
        "reputation_sigma",
    )
    parameter_units = {
        "p_minute": "probability per minute",
        "mean_duration_minutes": "minutes",
    }

    def build_prior(self, **context: Any) -> pm.Model:
        n_agents = context["n_agents"]
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
                shape=n_agents,
            )
        return prior

    def simulate(
        self,
        parameters: Mapping[str, NDArray[Any]],
        rng: np.random.Generator,
        **context: Any,
    ) -> Mapping[str, ArrayLike]:
        n_agents = int(context["n_agents"])
        n_steps = int(context["n_steps"])
        reputations = np.asarray(parameters["reputation"])
        p_step = per_step_probability(parameters["p_minute"])
        mean_duration = parameters["mean_duration_minutes"]
        weights = np.exp(reputations - np.max(reputations))

        def choose_partner(initiator: int, candidates: NDArray[np.intp]) -> int:
            del initiator
            partner_weights = weights[candidates]
            return int(
                rng.choice(
                    candidates,
                    p=partner_weights / partner_weights.sum(),
                )
            )

        return simulate_exclusive_conversations(
            rng,
            n_agents=n_agents,
            n_steps=n_steps,
            p_step=p_step,
            mean_duration=mean_duration,
            choose_partner=choose_partner,
        )


__all__ = [
    "ReputationConversationModel",
    "duration_in_steps",
    "partner_probabilities",
    "per_step_probability",
    "simulate_exclusive_conversations",
]
