"""Registered contact simulation models."""

from base.abm import Model

from .reputation import ReputationConversationModel


MODEL_CLASSES: tuple[type[Model], ...] = (
    ReputationConversationModel,
)
MODEL_REGISTRY: dict[str, type[Model]] = {
    model.name: model for model in MODEL_CLASSES
}

if len(MODEL_REGISTRY) != len(MODEL_CLASSES):
    raise ValueError("registered models must have unique names")


__all__ = [
    "MODEL_CLASSES",
    "MODEL_REGISTRY",
    "ReputationConversationModel",
]
