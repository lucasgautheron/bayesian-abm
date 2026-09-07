"""SIRD story diffusion with Weibull interest and exponential forgetting."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import numpy as np
import pymc as pm
from numpy.typing import ArrayLike, NDArray

from .base import StoryModel


SOURCE_POPULATION = 1_000_000


def weibull_mean_interest_duration(scale: float, shape: float) -> float:
    """Return the mean duration of a Weibull interest period."""

    try:
        return float(scale) * math.gamma(1.0 + 1.0 / float(shape))
    except OverflowError:
        return math.inf


def weibull_recovery_probability(
    infection_age: int,
    *,
    scale: float,
    shape: float,
) -> float:
    """Return recovery probability over the next day conditional on interest."""

    start = np.power(float(infection_age) / float(scale), float(shape))
    stop = np.power((float(infection_age) + 1.0) / float(scale), float(shape))
    if not np.isfinite(stop):
        return 1.0
    integrated_hazard = max(0.0, float(stop - start))
    return float(-np.expm1(-integrated_hazard))


def sir_infection_probability(
    infected: int | NDArray[np.integer],
    *,
    population: int,
    reproduction_number: float | NDArray[np.floating],
    mean_interest_duration: float,
) -> float | NDArray[np.float64]:
    """Return one susceptible source's infection probability for one day."""

    infected_array = np.asarray(infected)
    reproduction = np.asarray(reproduction_number, dtype=np.float64)
    scalar = infected_array.ndim == 0
    probabilities = np.zeros(infected_array.shape, dtype=np.float64)
    if infected_array.size and not np.isinf(mean_interest_duration):
        infected_values = infected_array.astype(np.int64, copy=False)
        reproduction = np.broadcast_to(reproduction, infected_values.shape)
        active = (infected_values > 0) & (reproduction > 0)
        if np.any(active):
            force = (
                reproduction[active]
                * infected_values[active]
                / (population * float(mean_interest_duration))
            )
            probabilities[active] = -np.expm1(-force)
    if scalar:
        return float(probabilities)
    return probabilities


class GeneralizedSIRModel(StoryModel):
    """Stories spread while infected, keep reporting while recovered, then die.

    Recovered sources still mention the story. Dead sources do not. Infection
    uses a shared reproduction number; forgetting is a constant daily hazard.
    """

    name = "generalized_sir"
    inference_variables = (
        "story_rate",
        "reproduction_number",
        "interest_scale",
        "interest_shape",
        "report_alpha",
        "report_beta",
        "report_ratio",
        "forget_scale",
    )
    parameter_units = {
        "story_rate": "stories per day",
        "reproduction_number": "secondary sources per source",
        "interest_scale": "days",
        "forget_scale": "days",
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
                "reproduction_number",
                mu=np.log(2.0),
                sigma=0.75,
            )
            pm.LogNormal(
                "interest_scale",
                mu=np.log(7.0),
                sigma=1.0,
            )
            pm.Exponential("interest_shape", lam=1.0)
            pm.Exponential("report_alpha", lam=1.0)
            pm.Exponential("report_beta", lam=1.0)
            pm.Uniform("report_ratio", lower=0.0, upper=1.0)
            pm.LogNormal(
                "forget_scale",
                mu=np.log(80.0),
                sigma=0.75,
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
        reproduction_number = float(parameters["reproduction_number"])
        interest_scale = float(parameters["interest_scale"])
        interest_shape = float(parameters["interest_shape"])
        report_alpha = float(parameters["report_alpha"])
        report_beta = float(parameters["report_beta"])
        report_ratio = float(parameters["report_ratio"])
        forget_scale = float(parameters["forget_scale"])
        death_probability = (
            0.0
            if np.isinf(forget_scale)
            else float(-np.expm1(-1.0 / forget_scale))
        )
        mean_duration = weibull_mean_interest_duration(
            interest_scale,
            interest_shape,
        )
        daily_recovery_probabilities = np.asarray(
            [
                weibull_recovery_probability(
                    age,
                    scale=interest_scale,
                    shape=interest_shape,
                )
                for age in range(n_days)
            ],
            dtype=np.float64,
        )

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

        report_infected = np.asarray(
            rng.beta(report_alpha, report_beta, size=n_stories),
            dtype=np.float64,
        )
        report_recovered = report_ratio * report_infected
        can_report_recovered = report_recovered > 0
        susceptible = np.full(
            n_stories,
            SOURCE_POPULATION - 1,
            dtype=np.int64,
        )
        infected = np.zeros(n_stories, dtype=np.int64)
        recovered = np.zeros(n_stories, dtype=np.int64)
        cohorts = np.zeros((n_stories, n_days), dtype=np.int64)
        cursor = 0
        finite_duration = not np.isinf(mean_duration)
        infect_scale = (
            reproduction_number / (SOURCE_POPULATION * mean_duration)
            if finite_duration and reproduction_number
            else 0.0
        )

        for day, n_born in enumerate(births.tolist()):
            if n_born:
                stop = cursor + n_born
                infected[cursor:stop] = 1
                cohorts[cursor:stop, day] = 1
                cursor = stop

            infected_idx = np.flatnonzero(infected)
            if infected_idx.size:
                mentions[infected_idx, day] += rng.binomial(
                    infected[infected_idx],
                    report_infected[infected_idx],
                )
            recovered_idx = np.flatnonzero(
                (recovered > 0) & can_report_recovered
            )
            if recovered_idx.size:
                mentions[recovered_idx, day] += rng.binomial(
                    recovered[recovered_idx],
                    report_recovered[recovered_idx],
                )
            if day + 1 == n_days:
                break

            spreading_idx = infected_idx
            if spreading_idx.size:
                susceptible_counts = susceptible[spreading_idx]
                new_infections = np.zeros(spreading_idx.size, dtype=np.int64)
                can_infect = susceptible_counts > 0
                if can_infect.any() and infect_scale:
                    force = infect_scale * infected[spreading_idx][can_infect]
                    new_infections[can_infect] = rng.binomial(
                        susceptible_counts[can_infect],
                        -np.expm1(-force),
                    )
                susceptible[spreading_idx] -= new_infections

                start = int(birth_days[spreading_idx].min())
                width = day + 1 - start
                if spreading_idx.size * width <= 200_000:
                    live_cohorts = cohorts[spreading_idx, start : day + 1]
                    recoveries = rng.binomial(
                        live_cohorts,
                        daily_recovery_probabilities[day - np.arange(start, day + 1)],
                    )
                    newly_recovered = recoveries.sum(axis=1, dtype=np.int64)
                    cohorts[spreading_idx, start : day + 1] = (
                        live_cohorts - recoveries
                    )
                else:
                    newly_recovered = np.zeros(
                        spreading_idx.size,
                        dtype=np.int64,
                    )
                    for age_day in range(start, day + 1):
                        column = cohorts[spreading_idx, age_day]
                        if not column.any():
                            continue
                        recovered_here = rng.binomial(
                            column,
                            daily_recovery_probabilities[day - age_day],
                        )
                        newly_recovered += recovered_here
                        cohorts[spreading_idx, age_day] = (
                            column - recovered_here
                        )
                cohorts[spreading_idx, day + 1] = new_infections
                infected[spreading_idx] += new_infections - newly_recovered
                recovered[spreading_idx] += newly_recovered

            if death_probability:
                dying_idx = np.flatnonzero(recovered > 0)
                if dying_idx.size:
                    recovered[dying_idx] -= rng.binomial(
                        recovered[dying_idx],
                        death_probability,
                    )

        return {"mentions": mentions}


__all__ = [
    "SOURCE_POPULATION",
    "GeneralizedSIRModel",
    "sir_infection_probability",
    "weibull_mean_interest_duration",
    "weibull_recovery_probability",
]
