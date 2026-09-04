"""SBM relationship network with activity-weighted occupancy."""

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


def start_probability(
    pair_products: ArrayLike,
    start_rate: float,
) -> NDArray[np.float64]:
    """Return start probabilities from stored pair activity products."""

    products = np.asarray(pair_products, dtype=np.float64)
    return np.clip(-np.expm1(-start_rate * products), 0.0, 1.0)


def pair_start_probability(
    activities_i: ArrayLike,
    activities_j: ArrayLike,
    start_rate: float,
) -> NDArray[np.float64]:
    """Return activity-weighted start probabilities for linked pairs."""

    first = np.asarray(activities_i, dtype=np.float64)
    second = np.asarray(activities_j, dtype=np.float64)
    return start_probability(first * second, start_rate)


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


def edge_probabilities(
    communities: ArrayLike,
    p_edge: float,
    between_ratio: float,
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float64]]:
    """Return pair indices and SBM edge probabilities."""

    labels = np.asarray(communities)
    first, second = np.triu_indices(labels.size, k=1)
    first = np.asarray(first, dtype=np.int32)
    second = np.asarray(second, dtype=np.int32)
    probabilities = np.where(
        labels[first] == labels[second],
        p_edge,
        between_ratio * p_edge,
    )
    return first, second, np.asarray(probabilities, dtype=np.float64)


def draw_sbm_edges(
    rng: np.random.Generator,
    communities: ArrayLike,
    p_edge: float,
    between_ratio: float,
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Draw an undirected SBM and return its upper-triangle edges."""

    first, second, probabilities = edge_probabilities(
        communities,
        p_edge,
        between_ratio,
    )
    keep = rng.random(first.size) < probabilities
    return first[keep], second[keep]


class LatentNetworkModel(ContactModel):
    """Occupancy on a static SBM with a log-OU start-rate path."""

    name = "latent_network"
    inference_variables = (
        "group_rate",
        "start_rate",
        "lengthscale",
        "mean_duration_minutes",
        "activity_sigma",
        "between_ratio",
        "p_edge",
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
            pm.Beta("between_ratio", alpha=1.0, beta=20.0)
            pm.Beta("p_edge", alpha=2.0, beta=5.0)
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
        lengthscale = float(parameters["lengthscale"])
        mean_duration = float(parameters["mean_duration_minutes"])
        activity_sigma = float(parameters["activity_sigma"])
        between_ratio = float(parameters["between_ratio"])
        p_edge = float(parameters["p_edge"])

        n_communities = n_groups(group_rate, n_agents)
        communities = rng.integers(n_communities, size=n_agents, dtype=np.int32)
        activities = np.asarray(
            rng.lognormal(0.0, activity_sigma, size=n_agents),
            dtype=np.float64,
        )
        first, second = draw_sbm_edges(
            rng,
            communities,
            p_edge,
            between_ratio,
        )
        if not first.size:
            empty = np.empty(0, dtype=np.int32)
            return {"t": empty, "i": empty, "j": empty}

        pair_products = activities[first] * activities[second]
        rates = start_rate * np.exp(
            draw_log_ou_path(rng, n_steps, lengthscale)
        )
        p_start = start_probability(pair_products, rates[0])
        q_end = end_probability(mean_duration)
        p_stay = 1.0 - q_end
        occupied = rng.random(first.size) < p_start

        times: list[NDArray[np.int32]] = []
        first_agents: list[NDArray[np.int32]] = []
        second_agents: list[NDArray[np.int32]] = []
        for step in range(n_steps):
            time = np.int32((step + 1) * INTERVAL_SECONDS)
            active = occupied
            n_active = int(active.sum())
            if n_active:
                times.append(np.full(n_active, time, dtype=np.int32))
                first_agents.append(first[active])
                second_agents.append(second[active])
            if step + 1 == n_steps:
                break
            p_start = start_probability(pair_products, rates[step + 1])
            stay = rng.random(occupied.size) < p_stay
            start = rng.random(occupied.size) < p_start
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
    "GP_LOG_SIGMA",
    "LENGTHSCALE_MEAN_MINUTES",
    "LatentNetworkModel",
    "draw_log_ou_path",
    "draw_sbm_edges",
    "edge_probabilities",
    "pair_start_probability",
    "start_probability",
]
