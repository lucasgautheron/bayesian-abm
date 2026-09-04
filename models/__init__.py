"""Registered contact simulation models."""

from base.model import Model

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
    raise ValueError("registered models must have unique names")


__all__ = [
    "FriendModel",
    "MODEL_CLASSES",
    "MODEL_REGISTRY",
    "ReputationConversationModel",
]
