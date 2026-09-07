"""Independent stories with heavy-tailed daily mention rates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pymc as pm
from numpy.typing import ArrayLike, NDArray

from .base import StoryModel


def daily_death_probability(lifetime_scale: float) -> float:
    """Return the daily geometric death probability from a mean lifetime."""

    if np.isinf(lifetime_scale):
        return 0.0
    return float(-np.expm1(-1.0 / float(lifetime_scale)))


class LatentRateModel(StoryModel):
    """Stories mention independently at a heterogeneous daily rate until death."""

    name = "latent_rate"
    inference_variables = (
        "story_rate",
        "rate_mu",
        "rate_sigma",
        "lifetime_scale",
    )
    parameter_units = {
        "story_rate": "stories per day",
        "rate_mu": "log(mentions per day)",
        "lifetime_scale": "days",
    }

    def build_prior(self, **context: Any) -> pm.Model:
        del context
        with pm.Model() as prior:
            pm.LogNormal(
                "story_rate",
                mu=np.log(20.0),
                sigma=1.0,
            )
            pm.Normal("rate_mu", mu=0.0, sigma=1.0)
            pm.HalfNormal("rate_sigma", sigma=1.0)
            pm.LogNormal(
                "lifetime_scale",
                mu=np.log(150.0),
                sigma=0.5,
            )
        return prior

    def simulate(
        self,
        parameters: Mapping[str, NDArray[Any]],
        rng: np.random.Generator,
        **context: Any,
    ) -> Mapping[str, ArrayLike]:
        n_days = int(context["n_days"])
        story_rate = float(parameters["story_rate"])
        rate_mu = float(parameters["rate_mu"])
        rate_sigma = float(parameters["rate_sigma"])
        lifetime_scale = float(parameters["lifetime_scale"])
        death_probability = daily_death_probability(lifetime_scale)

        births = np.asarray(
            rng.poisson(story_rate, size=n_days),
            dtype=np.int64,
        )
        birth_days = np.repeat(
            np.arange(n_days, dtype=np.int64),
            births,
        )
        n_stories = int(len(birth_days))
        mentions = np.zeros((n_stories, n_days), dtype=np.int64)
        if not n_stories:
            return {"mentions": mentions}

        rates = np.asarray(
            rng.lognormal(rate_mu, rate_sigma, size=n_stories),
            dtype=np.float64,
        )
        if death_probability:
            lifetimes = np.asarray(
                rng.geometric(death_probability, size=n_stories),
                dtype=np.int64,
            )
        else:
            lifetimes = np.full(n_stories, n_days, dtype=np.int64)
        death_days = np.minimum(birth_days + lifetimes, n_days)
        days = np.arange(n_days, dtype=np.int64)
        alive = (days >= birth_days[:, None]) & (days < death_days[:, None])
        draws = rng.poisson(rates[:, None], size=(n_stories, n_days))
        mentions[alive] = draws[alive]
        return {"mentions": mentions}


__all__ = [
    "LatentRateModel",
    "daily_death_probability",
]
