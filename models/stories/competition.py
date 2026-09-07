"""Joint story-population model with age- and appeal-weighted reports."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pymc as pm
from numpy.typing import ArrayLike, NDArray

from .base import StoryModel


def story_choice_probabilities(
    ages: ArrayLike,
    appeals: ArrayLike,
    cumulative_mentions: ArrayLike,
    *,
    beta_age: float,
    beta_appeal: float,
    reinforcement: float,
    n_days: int,
) -> NDArray[np.float64]:
    """Return stable age-, appeal-, and reinforcement-weighted probabilities."""

    age_values = np.asarray(ages, dtype=np.float64)
    appeal_values = np.asarray(appeals, dtype=np.float64)
    mention_values = np.asarray(cumulative_mentions, dtype=np.float64)
    normalized_ages = age_values / max(n_days - 1, 1)
    logits = (
        beta_age * normalized_ages
        + beta_appeal * appeal_values
        + reinforcement * np.log1p(mention_values)
    )
    weights = np.exp(logits - logits.max())
    return weights / weights.sum()


class StoryCompetitionModel(StoryModel):
    """Stories arrive over time and compete for a shared report stream."""

    name = "story_competition"
    inference_variables = (
        "story_rate",
        "report_rate",
        "beta_age",
        "beta_appeal",
        "reinforcement",
    )
    parameter_units = {
        "story_rate": "stories per day",
        "report_rate": "mentions per day",
    }

    def build_prior(self, **context: Any) -> pm.Model:
        del context
        with pm.Model() as prior:
            pm.LogNormal(
                "story_rate",
                mu=np.log(5.0),
                sigma=0.75,
            )
            pm.LogNormal(
                "report_rate",
                mu=np.log(1_000.0),
                sigma=1.5,
            )
            pm.Normal("beta_age", mu=0.0, sigma=1.0)
            pm.HalfNormal("beta_appeal", sigma=1.0)
            pm.HalfNormal("reinforcement", sigma=1.0)
        return prior

    def simulate(
        self,
        parameters: Mapping[str, NDArray[Any]],
        rng: np.random.Generator,
        **context: Any,
    ) -> Mapping[str, ArrayLike]:
        n_days = int(context["n_days"])
        story_rate = parameters["story_rate"]
        report_rate = parameters["report_rate"]
        beta_age = parameters["beta_age"]
        beta_appeal = parameters["beta_appeal"]
        reinforcement = parameters["reinforcement"]

        births = np.asarray(
            rng.poisson(story_rate, size=n_days),
            dtype=np.int64,
        )
        birth_days = np.repeat(
            np.arange(n_days, dtype=np.int64),
            births,
        )
        appeals = np.asarray(
            rng.normal(loc=0.0, scale=1.0, size=len(birth_days)),
            dtype=np.float64,
        )
        mentions = np.zeros((len(birth_days), n_days), dtype=np.int64)
        cumulative_mentions = np.zeros(len(birth_days), dtype=np.int64)

        active_count = 0
        for day, birth_count in enumerate(births):
            active_count += int(birth_count)
            if active_count == 0:
                continue

            report_count = int(rng.poisson(report_rate))
            probabilities = story_choice_probabilities(
                day - birth_days[:active_count],
                appeals[:active_count],
                cumulative_mentions[:active_count],
                beta_age=beta_age,
                beta_appeal=beta_appeal,
                reinforcement=reinforcement,
                n_days=n_days,
            )
            daily_mentions = rng.multinomial(
                report_count,
                probabilities,
            )
            mentions[:active_count, day] = daily_mentions
            cumulative_mentions[:active_count] += daily_mentions

        return {"mentions": mentions}


__all__ = [
    "StoryCompetitionModel",
    "story_choice_probabilities",
]
