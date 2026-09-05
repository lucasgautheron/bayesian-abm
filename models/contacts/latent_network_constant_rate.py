"""Weighted SBM affinities; Poisson starts at a constant activity-and-affinity rate."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pymc as pm
from numpy.typing import ArrayLike, NDArray

from base.model import INTERVAL_SECONDS

from .base import ContactModel
from .group_occupancy import end_probability, n_groups
from .latent_network import (
    PARETO_ALPHA,
    PARETO_MINIMUM,
    affinity_means,
    draw_affinities,
    draw_weighted_indices,
)


def _empty_contacts() -> dict[str, NDArray[np.int32]]:
    empty = np.empty(0, dtype=np.int32)
    return {"t": empty, "i": empty, "j": empty}


class LatentNetworkConstantRateModel(ContactModel):
    """Static Beta affinities; Poisson starts with a constant start rate."""

    name = "latent_network_constant_rate"
    inference_variables = (
        "group_rate",
        "start_rate",
        "mean_duration_minutes",
        "activity_sigma",
        "mu_within",
        "mu_ratio",
        "eta",
    )

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
            pm.Beta("mu_within", alpha=2.0, beta=5.0)
            pm.Beta("mu_ratio", alpha=1.0, beta=20.0)
            pm.Pareto("eta", alpha=PARETO_ALPHA, m=PARETO_MINIMUM)
        return prior

    def simulate(
        self,
        parameters: Mapping[str, NDArray[Any]],
        rng: np.random.Generator,
        **context: Any,
    ) -> Mapping[str, ArrayLike]:
        n_agents = int(context["n_agents"])
        n_steps = int(context["n_steps"])
        if n_agents < 2 or n_steps < 1:
            return _empty_contacts()

        group_rate = float(parameters["group_rate"])
        start_rate = float(parameters["start_rate"])
        mean_duration = float(parameters["mean_duration_minutes"])
        activity_sigma = float(parameters["activity_sigma"])
        mu_within = float(parameters["mu_within"])
        mu_ratio = float(parameters["mu_ratio"])
        eta = float(parameters["eta"])

        n_communities = n_groups(group_rate, n_agents)
        communities = rng.integers(n_communities, size=n_agents, dtype=np.int32)
        activities = np.asarray(
            rng.lognormal(0.0, activity_sigma, size=n_agents),
            dtype=np.float64,
        )
        first, second, means = affinity_means(
            communities,
            mu_within,
            mu_ratio,
        )
        affinities = draw_affinities(rng, means, eta)
        pair_products = activities[first] * activities[second]
        hazards = affinities * pair_products
        occupied = np.zeros(first.size, dtype=np.bool_)
        p_stay = 1.0 - end_probability(mean_duration)

        def _start_conversations(blocked: NDArray[np.bool_]) -> None:
            nonlocal occupied
            if start_rate <= 0.0:
                return
            mass = np.where(blocked, 0.0, hazards)
            intensity = float(mass.sum())
            if intensity <= 0.0:
                return
            n_idle = int(blocked.size - blocked.sum())
            n_new = min(int(rng.poisson(start_rate * intensity)), n_idle)
            if n_new <= 0:
                return
            occupied[draw_weighted_indices(rng, mass, n_new)] = True

        _start_conversations(occupied)

        times: list[NDArray[np.int32]] = []
        first_agents: list[NDArray[np.int32]] = []
        second_agents: list[NDArray[np.int32]] = []
        for step in range(n_steps):
            active = occupied
            n_active = int(active.sum())
            if n_active:
                time = np.int32((step + 1) * INTERVAL_SECONDS)
                times.append(np.full(n_active, time, dtype=np.int32))
                first_agents.append(first[active])
                second_agents.append(second[active])
            if step + 1 == n_steps:
                break
            blocked = occupied.copy()
            if n_active:
                occupied[active] = rng.random(n_active) < p_stay
            _start_conversations(blocked)

        if not times:
            return _empty_contacts()
        return {
            "t": np.concatenate(times),
            "i": np.concatenate(first_agents),
            "j": np.concatenate(second_agents),
        }


__all__ = ["LatentNetworkConstantRateModel"]
