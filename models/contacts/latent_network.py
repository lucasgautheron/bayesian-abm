"""Weighted SBM affinities with OU-modulated Poisson conversation starts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pymc as pm
from numpy.typing import ArrayLike, NDArray

from .base import ContactModel
from .latent_network_gravity import (
    LENGTHSCALE_MEAN_MINUTES,
    PARETO_ALPHA,
    PARETO_MINIMUM,
    affinity_means,
    draw_affinities,
    draw_log_ou_path,
    empty_contacts,
    n_groups,
    simulate_weighted_conversations,
)


class LatentNetworkModel(ContactModel):
    """Static Beta pair affinities with OU-modulated conversation starts."""

    name = "latent_network"
    inference_variables = (
        "group_rate",
        "start_rate",
        "lengthscale",
        "mean_duration_minutes",
        "mu_within",
        "mu_ratio",
        "eta",
    )
    parameter_units = {
        "group_rate": "groups",
        "start_rate": "per minute",
        "lengthscale": "minutes",
        "mean_duration_minutes": "minutes",
    }

    def build_prior(self, **context: Any) -> pm.Model:
        del context
        with pm.Model() as prior:
            pm.LogNormal("group_rate", mu=np.log(6.0), sigma=0.75)
            pm.LogNormal("start_rate", mu=np.log(0.001), sigma=1.0)
            pm.Exponential(
                "lengthscale",
                lam=1.0 / LENGTHSCALE_MEAN_MINUTES,
            )
            pm.LogNormal(
                "mean_duration_minutes",
                mu=np.log(3.0),
                sigma=0.5,
            )
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
            return empty_contacts()

        group_rate = float(parameters["group_rate"])
        start_rate = float(parameters["start_rate"])
        lengthscale = float(parameters["lengthscale"])
        mean_duration = float(parameters["mean_duration_minutes"])
        mu_within = float(parameters["mu_within"])
        mu_ratio = float(parameters["mu_ratio"])
        eta = float(parameters["eta"])

        n_communities = n_groups(group_rate, n_agents)
        communities = rng.integers(n_communities, size=n_agents, dtype=np.int32)
        first, second, means = affinity_means(
            communities,
            mu_within,
            mu_ratio,
        )
        hazards = draw_affinities(rng, means, eta)
        rates = start_rate * np.exp(
            draw_log_ou_path(rng, n_steps, lengthscale)
        )
        return simulate_weighted_conversations(
            rng,
            first=first,
            second=second,
            hazards=hazards,
            rates=rates,
            mean_duration_minutes=mean_duration,
        )


__all__ = ["LatentNetworkModel"]
