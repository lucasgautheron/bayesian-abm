"""Dataset-specific simulation-model registries."""

from base.model import Model

from .contacts import (
    GravityModel,
    GroupOccupancyModel,
    LatentNetworkGravityConstantRateModel,
    LatentNetworkGravityModel,
    LatentNetworkModel,
    ReputationConversationModel,
    SpatialConversationModel,
)
from .contacts import MODEL_CLASSES as CONTACT_MODEL_CLASSES
from .contacts import MODEL_REGISTRY as CONTACT_MODEL_REGISTRY
from .stories import (
    GeneralizedSIRModel,
    LatentRateModel,
    LimitedAttentionModel,
    LinearInfluenceModel,
    StoryCompetitionModel,
)
from .stories import MODEL_CLASSES as STORY_MODEL_CLASSES
from .stories import MODEL_REGISTRY as STORY_MODEL_REGISTRY


MODEL_REGISTRIES: dict[str, dict[str, type[Model]]] = {
    "contacts": CONTACT_MODEL_REGISTRY,
    "story_daily": STORY_MODEL_REGISTRY,
}

MODEL_CLASSES: tuple[type[Model], ...] = (
    *CONTACT_MODEL_CLASSES,
    *STORY_MODEL_CLASSES,
)
MODEL_REGISTRY: dict[str, type[Model]] = {
    model.name: model for model in MODEL_CLASSES
}
if len(MODEL_REGISTRY) != len(MODEL_CLASSES):
    raise ValueError("registered models must have globally unique names")


def model_registry(dataset: str) -> dict[str, type[Model]]:
    """Return the model registry belonging to one observed dataset."""

    try:
        return MODEL_REGISTRIES[dataset]
    except KeyError as exc:
        choices = ", ".join(sorted(MODEL_REGISTRIES))
        raise ValueError(
            f"unknown dataset {dataset!r}; available datasets: {choices}"
        ) from exc


def resolve_model(name: str) -> Model:
    """Instantiate a model by its globally unique name."""

    try:
        return MODEL_REGISTRY[name]()
    except KeyError as exc:
        choices = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"unknown model {name!r}; available models: {choices}"
        ) from exc


__all__ = [
    "CONTACT_MODEL_CLASSES",
    "CONTACT_MODEL_REGISTRY",
    "GeneralizedSIRModel",
    "GravityModel",
    "GroupOccupancyModel",
    "LatentNetworkGravityConstantRateModel",
    "LatentNetworkGravityModel",
    "LatentNetworkModel",
    "LatentRateModel",
    "LinearInfluenceModel",
    "LimitedAttentionModel",
    "MODEL_CLASSES",
    "MODEL_REGISTRY",
    "MODEL_REGISTRIES",
    "ReputationConversationModel",
    "SpatialConversationModel",
    "STORY_MODEL_CLASSES",
    "STORY_MODEL_REGISTRY",
    "StoryCompetitionModel",
    "model_registry",
    "resolve_model",
]
