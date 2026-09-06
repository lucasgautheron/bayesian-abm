"""Models for daily MemeTracker story activity."""

from base.model import Model
from datasets.story_daily.schema import STORY_DATASET as _STORY_DATASET

from .base import StoryModel
from .competition import StoryCompetitionModel
from .generalized_sir import GeneralizedSIRModel
from .latent_rate import LatentRateModel
from .linear_influence import LinearInfluenceModel
from .limited_attention import LimitedAttentionModel


MODEL_CLASSES: tuple[type[Model], ...] = (
    StoryCompetitionModel,
    LimitedAttentionModel,
    LinearInfluenceModel,
    GeneralizedSIRModel,
    LatentRateModel,
)
MODEL_REGISTRY: dict[str, type[Model]] = {
    model.name: model for model in MODEL_CLASSES
}
if len(MODEL_REGISTRY) != len(MODEL_CLASSES):
    raise ValueError("registered story models must have unique names")
if any(model.dataset != _STORY_DATASET for model in MODEL_CLASSES):
    raise ValueError("story models must declare the story_daily dataset")


__all__ = [
    "GeneralizedSIRModel",
    "LatentRateModel",
    "LinearInfluenceModel",
    "LimitedAttentionModel",
    "MODEL_CLASSES",
    "MODEL_REGISTRY",
    "StoryCompetitionModel",
    "StoryModel",
]
