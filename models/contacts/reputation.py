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
    """Convert a per-minute probability to one 20-second step."""

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
    """Draw an exponential duration and round it up to 20-second steps."""

    duration_minutes = rng.exponential(scale=mean_minutes)
    return max(1, int(np.ceil(duration_minutes * STEPS_PER_MINUTE)))


class ReputationConversationModel(ContactModel):
    """Agents start conversations and prefer reputable available partners."""

    name = "reputation_conversation"
    inference_variables = (
        "p_minute",
        "mean_duration_minutes",
        "reputation_sigma",
    )

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
        n_agents = context["n_agents"]
        n_steps = context["n_steps"]
        if n_agents < 2 or n_steps < 1:
            raise ValueError("n_agents must be at least 2 and n_steps positive")

        reputations = np.asarray(parameters["reputation"])
        p_step = per_step_probability(parameters["p_minute"])
        mean_duration = parameters["mean_duration_minutes"]

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
                probabilities = partner_probabilities(
                    reputations,
                    candidates,
                )
                partner = int(rng.choice(candidates, p=probabilities))
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
    "ReputationConversationModel",
    "duration_in_steps",
    "partner_probabilities",
    "per_step_probability",
]
