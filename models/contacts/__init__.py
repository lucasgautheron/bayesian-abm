"""Registered models for pairwise temporal-contact data."""

from base.model import Model

from .base import CONTACT_DATASET, ContactModel
from .friend import FriendModel
from .reputation import ReputationConversationModel


MODEL_CLASSES: tuple[type[Model], ...] = (
    ReputationConversationModel,
    FriendModel,
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
    "FriendModel",
    "MODEL_CLASSES",
    "MODEL_REGISTRY",
    "ReputationConversationModel",
]
