"""Models for scientists' binary convention preferences."""

from base.model import Model
from datasets.scientist_conventions.schema import (
    SCIENTIST_CONVENTIONS_DATASET as _SCIENTIST_CONVENTIONS_DATASET,
)

from .base import ScientistConventionModel
from .global_transmission import GlobalTransmissionModel
from .local_transmission import LocalTransmissionModel
from .strategic import StrategicConventionModel


MODEL_CLASSES: tuple[type[Model], ...] = (
    StrategicConventionModel,
    GlobalTransmissionModel,
    LocalTransmissionModel,
)
MODEL_REGISTRY: dict[str, type[Model]] = {
    model.name: model for model in MODEL_CLASSES
}
if len(MODEL_REGISTRY) != len(MODEL_CLASSES):
    raise ValueError("registered scientist models must have unique names")
if any(
    model.dataset != _SCIENTIST_CONVENTIONS_DATASET
    for model in MODEL_CLASSES
):
    raise ValueError("scientist models must declare the scientist dataset")


__all__ = [
    "GlobalTransmissionModel",
    "LocalTransmissionModel",
    "MODEL_CLASSES",
    "MODEL_REGISTRY",
    "ScientistConventionModel",
    "StrategicConventionModel",
]
