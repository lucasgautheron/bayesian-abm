"""Spatial random walks with pairwise conversations that pause movement."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pymc as pm
from numpy.typing import ArrayLike, NDArray

from base.model import INTERVAL_SECONDS

from .base import ContactModel


def empty_contacts() -> dict[str, NDArray[np.int32]]:
    """Return an empty contact table with the required dtypes."""

    empty = np.empty(0, dtype=np.int32)
    return {"t": empty, "i": empty, "j": empty}


def toroidal_pair_distances_squared(
    positions: ArrayLike,
    first: ArrayLike,
    second: ArrayLike,
) -> NDArray[np.float64]:
    """Return squared minimum-image distances for pairs on the unit torus."""

    coordinates = np.asarray(positions, dtype=np.float64)
    first_indices = np.asarray(first, dtype=np.intp)
    second_indices = np.asarray(second, dtype=np.intp)
    dx = np.abs(
        coordinates[first_indices, 0] - coordinates[second_indices, 0]
    )
    dx = np.minimum(dx, 1.0 - dx)
    dy = np.abs(
        coordinates[first_indices, 1] - coordinates[second_indices, 1]
    )
    dy = np.minimum(dy, 1.0 - dy)
    return dx * dx + dy * dy


def paused_agents(
    active: ArrayLike,
    first: ArrayLike,
    second: ArrayLike,
    n_agents: int,
) -> NDArray[np.bool_]:
    """Return which agents participate in at least one active conversation."""

    occupied = np.asarray(active, dtype=np.bool_)
    first_indices = np.asarray(first, dtype=np.intp)
    second_indices = np.asarray(second, dtype=np.intp)
    paused = np.zeros(n_agents, dtype=np.bool_)
    paused[first_indices[occupied]] = True
    paused[second_indices[occupied]] = True
    return paused


def move_unpaused(
    rng: np.random.Generator,
    positions: ArrayLike,
    paused: ArrayLike,
    movement_scale: float,
) -> NDArray[np.float64]:
    """Move unpaused agents by wrapped isotropic Gaussian increments."""

    moved = np.asarray(positions, dtype=np.float64).copy()
    stationary = np.asarray(paused, dtype=np.bool_)
    free = ~stationary
    n_free = int(free.sum())
    if n_free:
        increments = rng.normal(
            0.0,
            movement_scale,
            size=(n_free, 2),
        )
        moved[free] = (moved[free] + increments) % 1.0
    return moved


def update_conversations(
    rng: np.random.Generator,
    active: ArrayLike,
    nearby: ArrayLike,
    stay_probability: float,
    start_probability: float,
) -> NDArray[np.bool_]:
    """End old edges and start eligible new edges independently."""

    blocked = np.asarray(active, dtype=np.bool_)
    close = np.asarray(nearby, dtype=np.bool_)
    updated = np.zeros_like(blocked)

    occupied_indices = np.flatnonzero(blocked)
    if occupied_indices.size:
        updated[occupied_indices] = (
            rng.random(occupied_indices.size) < stay_probability
        )

    # Edges active at the start cannot end and restart in the same minute.
    eligible_indices = np.flatnonzero(~blocked & close)
    if eligible_indices.size:
        updated[eligible_indices] = (
            rng.random(eligible_indices.size) < start_probability
        )
    return updated


class SpatialConversationModel(ContactModel):
    """Random walkers pause while one or more pair conversations are active."""

    name = "spatial_conversation"
    inference_variables = (
        "interaction_radius",
        "movement_scale",
        "start_probability",
        "mean_duration_minutes",
    )
    parameter_units = {
        "interaction_radius": "unit-torus distance",
        "movement_scale": "unit-torus distance per minute",
        "mean_duration_minutes": "minutes",
    }

    def build_prior(self, **context: Any) -> pm.Model:
        del context
        with pm.Model() as prior:
            pm.Beta("interaction_radius", alpha=1.0, beta=9.0)
            pm.LogNormal(
                "movement_scale",
                mu=np.log(0.03),
                sigma=1.0,
            )
            pm.Beta("start_probability", alpha=1.0, beta=4.0)
            pm.LogNormal(
                "mean_duration_minutes",
                mu=np.log(3.0),
                sigma=0.75,
            )
        return prior

    def simulate(
        self,
        parameters: Mapping[str, NDArray[Any]],
        rng: np.random.Generator,
        **context: Any,
    ) -> Mapping[str, ArrayLike]:
        n_agents = int(context["n_agents"])
        n_steps = int(context["n_steps"])
        if n_agents < 0:
            raise ValueError("n_agents must be non-negative")
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        if n_agents < 2 or n_steps == 0:
            return empty_contacts()

        interaction_radius = float(parameters["interaction_radius"])
        movement_scale = float(parameters["movement_scale"])
        start_probability = float(parameters["start_probability"])
        mean_duration = float(parameters["mean_duration_minutes"])
        if not 0.0 <= interaction_radius <= 1.0:
            raise ValueError("interaction_radius must be between 0 and 1")
        if not np.isfinite(movement_scale) or movement_scale < 0.0:
            raise ValueError("movement_scale must be finite and non-negative")
        if not 0.0 <= start_probability <= 1.0:
            raise ValueError("start_probability must be between 0 and 1")
        if not np.isfinite(mean_duration) or mean_duration <= 0.0:
            raise ValueError(
                "mean_duration_minutes must be finite and positive"
            )

        first, second = np.triu_indices(n_agents, k=1)
        first = np.asarray(first, dtype=np.int32)
        second = np.asarray(second, dtype=np.int32)
        positions = np.asarray(
            rng.random((n_agents, 2)),
            dtype=np.float64,
        )
        active = np.zeros(first.size, dtype=np.bool_)
        stay_probability = float(np.exp(-1.0 / mean_duration))
        radius_squared = interaction_radius * interaction_radius

        times: list[NDArray[np.int32]] = []
        first_agents: list[NDArray[np.int32]] = []
        second_agents: list[NDArray[np.int32]] = []
        for step in range(n_steps):
            # Anyone talking at the start of the minute remains stationary.
            paused = paused_agents(active, first, second, n_agents)
            positions = move_unpaused(
                rng,
                positions,
                paused,
                movement_scale,
            )

            nearby = (
                toroidal_pair_distances_squared(positions, first, second)
                <= radius_squared
            )
            active = update_conversations(
                rng,
                active,
                nearby,
                stay_probability,
                start_probability,
            )

            n_active = int(active.sum())
            if n_active:
                time = np.int32((step + 1) * INTERVAL_SECONDS)
                times.append(np.full(n_active, time, dtype=np.int32))
                first_agents.append(first[active])
                second_agents.append(second[active])

        if not times:
            return empty_contacts()
        return {
            "t": np.concatenate(times),
            "i": np.concatenate(first_agents),
            "j": np.concatenate(second_agents),
        }


__all__ = [
    "SpatialConversationModel",
    "empty_contacts",
    "move_unpaused",
    "paused_agents",
    "toroidal_pair_distances_squared",
    "update_conversations",
]
