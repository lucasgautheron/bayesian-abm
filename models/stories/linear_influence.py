"""Aggregate generative adaptation of the Linear Influence Model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pymc as pm
from numpy.typing import ArrayLike, NDArray

from .base import StoryModel


def exponential_lag_kernel(
    decay: float,
    *,
    lag_count: int,
) -> NDArray[np.float64]:
    """Return a normalized exponential kernel for subsequent-day influence."""

    weights = np.exp(-float(decay) * np.arange(int(lag_count), dtype=np.float64))
    return weights / weights.sum()


def linear_influence_intensity(
    past_mentions: ArrayLike,
    kernel: ArrayLike,
    *,
    branching_ratio: float,
    background_rate: float = 0.0,
) -> float:
    """Return next-day intensity from chronological prior mention counts."""

    mentions = np.asarray(past_mentions, dtype=np.float64)
    weights = np.asarray(kernel, dtype=np.float64)
    lag_count = min(len(mentions), len(weights))
    if lag_count == 0:
        return 0.0
    used = weights[:lag_count]
    return float(
        background_rate
        + branching_ratio
        * np.dot(mentions[-lag_count:][::-1], used)
        / used.sum()
    )


class LinearInfluenceModel(StoryModel):
    """Stories generate mentions through story-specific linear influence.

    A shared exponential kernel scores the full mention history. Each story
    draws its own seed rate and subcritical branching ratio. A shared
    background rate keeps stories from going permanently silent.
    """

    name = "linear_influence"
    inference_variables = (
        "story_rate",
        "background_rate",
        "seed_shape",
        "seed_scale",
        "branching_alpha",
        "branching_beta",
        "influence_decay",
    )
    parameter_units = {
        "story_rate": "stories per day",
        "background_rate": "mentions per day",
        "seed_scale": "mentions per day",
        "influence_decay": "per day",
    }

    def build_prior(self, **context: Any) -> pm.Model:
        del context
        with pm.Model() as prior:
            pm.LogNormal(
                "story_rate",
                mu=np.log(20.0),
                sigma=1.0,
            )
            pm.LogNormal(
                "background_rate",
                mu=np.log(2.0),
                sigma=1.0,
            )
            pm.Exponential("seed_shape", lam=1.0)
            pm.Exponential("seed_scale", lam=1.0)
            pm.Exponential("branching_alpha", lam=1.0)
            pm.Exponential("branching_beta", lam=1.0)
            pm.HalfNormal("influence_decay", sigma=1.0)
        return prior

    def simulate(
        self,
        parameters: Mapping[str, NDArray[Any]],
        rng: np.random.Generator,
        **context: Any,
    ) -> Mapping[str, ArrayLike]:
        n_days = int(context["n_days"])
        story_rate = float(parameters["story_rate"])
        background_rate = float(parameters["background_rate"])
        seed_shape = float(parameters["seed_shape"])
        seed_scale = float(parameters["seed_scale"])
        branching_alpha = float(parameters["branching_alpha"])
        branching_beta = float(parameters["branching_beta"])
        influence_decay = float(parameters["influence_decay"])
        kernel = exponential_lag_kernel(influence_decay, lag_count=n_days)
        births = np.asarray(
            rng.poisson(story_rate, size=n_days),
            dtype=np.int64,
        )
        birth_days = np.repeat(
            np.arange(n_days, dtype=np.int64),
            births,
        )
        mentions = np.zeros((len(birth_days), n_days), dtype=np.int64)
        if not len(birth_days):
            return {"mentions": mentions}

        seed_rates = np.asarray(
            rng.gamma(seed_shape, seed_scale, size=len(birth_days)),
            dtype=np.float64,
        )
        branching_ratios = np.minimum(
            np.asarray(
                rng.beta(
                    branching_alpha,
                    branching_beta,
                    size=len(birth_days),
                ),
                dtype=np.float64,
            ),
            np.nextafter(1.0, 0.0),
        )
        mentions[np.arange(len(birth_days)), birth_days] = (
            1
            + np.asarray(
                rng.poisson(seed_rates),
                dtype=np.int64,
            )
        )

        active_count = int(births[0])
        for day in range(1, n_days):
            if active_count:
                past = mentions[:active_count, :day]
                weights = kernel[: past.shape[1]]
                intensities = background_rate + (
                    branching_ratios[:active_count]
                    * past[:, ::-1].dot(weights)
                    / weights.sum()
                )
                mentions[:active_count, day] = rng.poisson(intensities)
            active_count += int(births[day])

        return {"mentions": mentions}


__all__ = [
    "LinearInfluenceModel",
    "exponential_lag_kernel",
    "linear_influence_intensity",
]
