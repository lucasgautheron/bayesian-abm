"""Registered models for pairwise temporal-contact data."""

from base.model import Model

from .base import CONTACT_DATASET, ContactModel
from .gravity import GravityModel
from .latent_network import LatentNetworkModel
from .latent_network_gravity import LatentNetworkGravityModel
from .latent_network_gravity_constant_rate import (
    LatentNetworkGravityConstantRateModel,
)


MODEL_CLASSES: tuple[type[Model], ...] = (
    GravityModel,
    LatentNetworkModel,
    LatentNetworkGravityModel,
    LatentNetworkGravityConstantRateModel,
)
MODEL_REGISTRY: dict[str, type[Model]] = {
    model.name: model for model in MODEL_CLASSES
}

if len(MODEL_REGISTRY) != len(MODEL_CLASSES):
    raise ValueError("registered contact models must have unique names")
if any(model.dataset != CONTACT_DATASET for model in MODEL_CLASSES):
    raise ValueError("contact models must declare the contacts dataset")


__all__ = [
    "CONTACT_DATASET",
    "ContactModel",
    "GravityModel",
    "LatentNetworkGravityConstantRateModel",
    "LatentNetworkGravityModel",
    "LatentNetworkModel",
    "MODEL_CLASSES",
    "MODEL_REGISTRY",
]
