"""Agent gravity with a uniform pair baseline and OU-modulated starts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pymc as pm
from numpy.typing import ArrayLike, NDArray

from .base import ContactModel
from .latent_network_gravity import (
    LENGTHSCALE_MEAN_MINUTES,
    draw_log_ou_path,
    empty_contacts,
    simulate_weighted_conversations,
)


class GravityModel(ContactModel):
    """Agent-activity gravity with no latent pair-affinity structure."""

    name = "gravity"
    inference_variables = (
        "start_rate",
        "lengthscale",
        "mean_duration_minutes",
        "activity_sigma",
    )
    parameter_units = {
        "start_rate": "per minute",
        "lengthscale": "minutes",
        "mean_duration_minutes": "minutes",
    }

    def build_prior(self, **context: Any) -> pm.Model:
        del context
        with pm.Model() as prior:
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
            pm.HalfNormal("activity_sigma", sigma=1.0)
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

        start_rate = float(parameters["start_rate"])
        lengthscale = float(parameters["lengthscale"])
        mean_duration = float(parameters["mean_duration_minutes"])
        activity_sigma = float(parameters["activity_sigma"])

        activities = np.asarray(
            rng.lognormal(0.0, activity_sigma, size=n_agents),
            dtype=np.float64,
        )
        first, second = np.triu_indices(n_agents, k=1)
        first = np.asarray(first, dtype=np.int32)
        second = np.asarray(second, dtype=np.int32)
        hazards = activities[first] * activities[second]
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


__all__ = ["GravityModel"]
