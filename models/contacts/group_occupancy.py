"""Latent groups with activity-weighted Markov pair occupancy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pymc as pm
from numpy.typing import ArrayLike, NDArray

from base.model import INTERVAL_SECONDS

from .base import ContactModel


def n_groups(group_rate: float, n_agents: int) -> int:
    """Return the number of groups implied by the group-rate prior."""

    return max(1, min(int(n_agents), int(round(float(group_rate)))))


def end_probability(mean_duration_minutes: float) -> float:
    """Return the per-minute end probability from a mean on-run length."""

    if np.isinf(mean_duration_minutes):
        return 0.0
    return float(-np.expm1(-1.0 / float(mean_duration_minutes)))


def start_probabilities(
    communities: ArrayLike,
    activities: ArrayLike,
    start_rate: float,
    between_ratio: float,
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float64]]:
    """Return pair indices and activity-weighted start probabilities."""

    labels = np.asarray(communities)
    rates = np.asarray(activities, dtype=np.float64)
    first, second = np.triu_indices(labels.size, k=1)
    first = np.asarray(first, dtype=np.int32)
    second = np.asarray(second, dtype=np.int32)
    pair_rate = np.where(
        labels[first] == labels[second],
        start_rate,
        between_ratio * start_rate,
    )
    hazard = pair_rate * rates[first] * rates[second]
    probabilities = np.clip(-np.expm1(-hazard), 0.0, 1.0)
    return first, second, probabilities


class GroupOccupancyModel(ContactModel):
    """Pairs occupy independently; between-group starts are rarer."""

    name = "group_occupancy"
    inference_variables = (
        "group_rate",
        "start_rate",
        "mean_duration_minutes",
        "activity_sigma",
        "between_ratio",
    )
    parameter_units = {
        "group_rate": "groups",
        "start_rate": "per minute",
        "mean_duration_minutes": "minutes",
    }

    def build_prior(self, **context: Any) -> pm.Model:
        del context
        with pm.Model() as prior:
            pm.LogNormal("group_rate", mu=np.log(6.0), sigma=0.75)
            pm.LogNormal("start_rate", mu=np.log(0.001), sigma=1.0)
            pm.LogNormal(
                "mean_duration_minutes",
                mu=np.log(3.0),
                sigma=0.5,
            )
            pm.HalfNormal("activity_sigma", sigma=1.0)
            pm.Beta("between_ratio", alpha=1.0, beta=20.0)
        return prior

    def simulate(
        self,
        parameters: Mapping[str, NDArray[Any]],
        rng: np.random.Generator,
        **context: Any,
    ) -> Mapping[str, ArrayLike]:
        n_agents = int(context["n_agents"])
        n_steps = int(context["n_steps"])
        group_rate = float(parameters["group_rate"])
        start_rate = float(parameters["start_rate"])
        mean_duration = float(parameters["mean_duration_minutes"])
        activity_sigma = float(parameters["activity_sigma"])
        between_ratio = float(parameters["between_ratio"])

        n_communities = n_groups(group_rate, n_agents)
        communities = rng.integers(n_communities, size=n_agents, dtype=np.int32)
        activities = np.asarray(
            rng.lognormal(0.0, activity_sigma, size=n_agents),
            dtype=np.float64,
        )
        first, second, p_start = start_probabilities(
            communities,
            activities,
            start_rate,
            between_ratio,
        )
        q_end = end_probability(mean_duration)
        p_stay = 1.0 - q_end
        n_pairs = int(first.size)
        if not n_pairs:
            empty = np.empty(0, dtype=np.int32)
            return {"t": empty, "i": empty, "j": empty}

        occupied = rng.random(n_pairs) < p_start
        times: list[NDArray[np.int32]] = []
        first_agents: list[NDArray[np.int32]] = []
        second_agents: list[NDArray[np.int32]] = []
        for step in range(n_steps):
            active = occupied
            n_active = int(active.sum())
            if n_active:
                times.append(
                    np.full(
                        n_active,
                        np.int32((step + 1) * INTERVAL_SECONDS),
                        dtype=np.int32,
                    )
                )
                first_agents.append(first[active])
                second_agents.append(second[active])
            if step + 1 == n_steps:
                break
            stay = rng.random(n_pairs) < p_stay
            start = rng.random(n_pairs) < p_start
            occupied = np.where(active, stay, start)

        if not times:
            empty = np.empty(0, dtype=np.int32)
            return {"t": empty, "i": empty, "j": empty}
        return {
            "t": np.concatenate(times),
            "i": np.concatenate(first_agents),
            "j": np.concatenate(second_agents),
        }


__all__ = [
    "GroupOccupancyModel",
    "end_probability",
    "n_groups",
    "start_probabilities",
]
