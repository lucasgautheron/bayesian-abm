"""Models for daily MemeTracker story activity."""

from base.model import Model

from .base import (
    DEFAULT_STORY_SUMMARY_COUNT,
    STORY_DATASET,
    STORY_SUMMARY_KEYS,
    StoryData,
    StoryModel,
    make_story_summaries,
    ranked_story_mask,
    ranked_story_mentions,
    validate_story_data,
)
from .competition import StoryCompetitionModel


MODEL_CLASSES: tuple[type[Model], ...] = (StoryCompetitionModel,)
MODEL_REGISTRY: dict[str, type[Model]] = {
    model.name: model for model in MODEL_CLASSES
}
if len(MODEL_REGISTRY) != len(MODEL_CLASSES):
    raise ValueError("registered story models must have unique names")
if any(model.dataset != STORY_DATASET for model in MODEL_CLASSES):
    raise ValueError("story models must declare the story_daily dataset")


__all__ = [
    "DEFAULT_STORY_SUMMARY_COUNT",
    "MODEL_CLASSES",
    "MODEL_REGISTRY",
    "STORY_DATASET",
    "STORY_SUMMARY_KEYS",
    "StoryCompetitionModel",
    "StoryData",
    "StoryModel",
    "make_story_summaries",
    "ranked_story_mask",
    "ranked_story_mentions",
    "validate_story_data",
]
