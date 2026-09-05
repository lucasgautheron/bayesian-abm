"""Weighted SBM affinities and agent gravity; OU-modulated Poisson starts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pymc as pm
from numpy.typing import ArrayLike, NDArray

from base.model import INTERVAL_SECONDS

from .base import ContactModel
from .group_occupancy import end_probability, n_groups


GP_LOG_SIGMA = 1.0
LENGTHSCALE_MEAN_MINUTES = 60.0
MEAN_CLIP = 1.0e-6
PARETO_ALPHA = 1.5
PARETO_MINIMUM = 1.0


def affinity_means(
    communities: ArrayLike,
    mu_within: float,
    mu_ratio: float,
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float64]]:
    """Return pair indices and Beta means from group membership."""

    labels = np.asarray(communities)
    first, second = np.triu_indices(labels.size, k=1)
    first = np.asarray(first, dtype=np.int32)
    second = np.asarray(second, dtype=np.int32)
    within = np.clip(float(mu_within), MEAN_CLIP, 1.0 - MEAN_CLIP)
    between = np.clip(within * float(mu_ratio), MEAN_CLIP, 1.0 - MEAN_CLIP)
    means = np.where(labels[first] == labels[second], within, between)
    return first, second, np.asarray(means, dtype=np.float64)


def draw_affinities(
    rng: np.random.Generator,
    means: ArrayLike,
    eta: float,
) -> NDArray[np.float64]:
    """Draw pair affinities from Beta(mean * eta, (1 - mean) * eta)."""

    mu = np.asarray(means, dtype=np.float64)
    concentration = max(float(eta), MEAN_CLIP)
    return np.asarray(
        rng.beta(mu * concentration, (1.0 - mu) * concentration),
        dtype=np.float64,
    )


def draw_log_ou_path(
    rng: np.random.Generator,
    n_steps: int,
    lengthscale: float,
    sigma: float = GP_LOG_SIGMA,
) -> NDArray[np.float64]:
    """Draw a zero-mean OU path on the minute grid."""

    path = np.empty(n_steps, dtype=np.float64)
    if n_steps < 1:
        return path
    path[0] = rng.normal(0.0, sigma)
    if n_steps == 1:
        return path
    phi = float(np.exp(-1.0 / lengthscale))
    innovation_sd = sigma * np.sqrt(max(0.0, 1.0 - phi * phi))
    noise = rng.normal(0.0, innovation_sd, size=n_steps - 1)
    for step in range(1, n_steps):
        path[step] = phi * path[step - 1] + noise[step - 1]
    return path


def pair_product_sum(activities: NDArray[np.float64]) -> float:
    """Return ``sum_{i<j} α_i α_j``."""

    total = float(activities.sum())
    return 0.5 * (total * total - float(np.dot(activities, activities)))


def draw_weighted_indices(
    rng: np.random.Generator,
    weights: ArrayLike,
    n_draw: int,
) -> NDArray[np.intp]:
    """Sample ``n_draw`` indices with probability proportional to ``weights``."""

    mass = np.asarray(weights, dtype=np.float64)
    if n_draw <= 0 or mass.size == 0:
        return np.empty(0, dtype=np.intp)
    cdf = np.cumsum(mass)
    total = float(cdf[-1])
    if total <= 0.0:
        return np.empty(0, dtype=np.intp)
    slots = np.searchsorted(cdf, rng.random(n_draw) * total, side="right")
    np.minimum(slots, mass.size - 1, out=slots)
    return slots


def empty_contacts() -> dict[str, NDArray[np.int32]]:
    """Return an empty contact table with the required dtypes."""

    empty = np.empty(0, dtype=np.int32)
    return {"t": empty, "i": empty, "j": empty}


def simulate_weighted_conversations(
    rng: np.random.Generator,
    *,
    first: NDArray[np.int32],
    second: NDArray[np.int32],
    hazards: NDArray[np.float64],
    rates: ArrayLike,
    mean_duration_minutes: float,
) -> dict[str, NDArray[np.int32]]:
    """Simulate pair occupancy from weighted Poisson conversation starts."""

    step_rates = np.asarray(rates, dtype=np.float64)
    if step_rates.ndim != 1:
        raise ValueError("rates must be one-dimensional")
    if len(first) != len(second) or len(first) != len(hazards):
        raise ValueError("pair indices and hazards must have equal lengths")
    if not len(step_rates) or not len(first):
        return empty_contacts()

    occupied = np.zeros(first.size, dtype=np.bool_)
    p_stay = 1.0 - end_probability(mean_duration_minutes)

    def start_conversations(
        rate: float,
        blocked: NDArray[np.bool_],
    ) -> None:
        nonlocal occupied
        if rate <= 0.0:
            return
        mass = np.where(blocked, 0.0, hazards)
        intensity = float(mass.sum())
        if intensity <= 0.0:
            return
        n_idle = int(blocked.size - blocked.sum())
        n_new = min(int(rng.poisson(rate * intensity)), n_idle)
        if n_new <= 0:
            return
        occupied[draw_weighted_indices(rng, mass, n_new)] = True

    start_conversations(float(step_rates[0]), occupied)

    times: list[NDArray[np.int32]] = []
    first_agents: list[NDArray[np.int32]] = []
    second_agents: list[NDArray[np.int32]] = []
    for step, rate in enumerate(step_rates):
        active = occupied
        n_active = int(active.sum())
        if n_active:
            time = np.int32((step + 1) * INTERVAL_SECONDS)
            times.append(np.full(n_active, time, dtype=np.int32))
            first_agents.append(first[active])
            second_agents.append(second[active])
        if step + 1 == len(step_rates):
            break
        blocked = occupied.copy()
        if n_active:
            occupied[active] = rng.random(n_active) < p_stay
        start_conversations(float(step_rates[step + 1]), blocked)

    if not times:
        return empty_contacts()
    return {
        "t": np.concatenate(times),
        "i": np.concatenate(first_agents),
        "j": np.concatenate(second_agents),
    }


class LatentNetworkGravityModel(ContactModel):
    """Static Beta affinities and agent gravity with OU-modulated starts."""

    name = "latent_network_gravity"
    inference_variables = (
        "group_rate",
        "start_rate",
        "lengthscale",
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
        hazards = affinities * activities[first] * activities[second]
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


__all__ = [
    "GP_LOG_SIGMA",
    "LENGTHSCALE_MEAN_MINUTES",
    "MEAN_CLIP",
    "PARETO_ALPHA",
    "PARETO_MINIMUM",
    "LatentNetworkGravityModel",
    "affinity_means",
    "draw_affinities",
    "draw_log_ou_path",
    "draw_weighted_indices",
    "empty_contacts",
    "pair_product_sum",
    "simulate_weighted_conversations",
]
