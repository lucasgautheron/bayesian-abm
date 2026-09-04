"""Joint story-population model with age- and appeal-weighted reports."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pymc as pm
from numpy.typing import ArrayLike, NDArray

from .base import StoryModel


def _positive_integer(value: Any, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def story_choice_probabilities(
    ages: ArrayLike,
    appeals: ArrayLike,
    *,
    beta_age: float,
    beta_appeal: float,
    n_days: int,
) -> NDArray[np.float64]:
    """Return stable softmax probabilities for active stories."""

    n_days = _positive_integer(n_days, "n_days")
    age_values = np.asarray(ages, dtype=np.float64)
    appeal_values = np.asarray(appeals, dtype=np.float64)
    if (
        age_values.ndim != 1
        or appeal_values.ndim != 1
        or age_values.shape != appeal_values.shape
        or len(age_values) == 0
    ):
        raise ValueError(
            "ages and appeals must be non-empty one-dimensional arrays "
            "with equal lengths"
        )
    if (
        not np.all(np.isfinite(age_values))
        or np.any(age_values < 0)
        or not np.all(np.isfinite(appeal_values))
        or not np.isfinite(beta_age)
        or not np.isfinite(beta_appeal)
    ):
        raise ValueError("softmax inputs must be finite and ages non-negative")

    normalized_ages = age_values / max(n_days - 1, 1)
    logits = beta_age * normalized_ages + beta_appeal * appeal_values
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
    )

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
        return prior

    def simulate(
        self,
        parameters: Mapping[str, NDArray[Any]],
        rng: np.random.Generator,
        **context: Any,
    ) -> Mapping[str, ArrayLike]:
        try:
            n_days = _positive_integer(context["n_days"], "n_days")
        except KeyError as exc:
            raise ValueError("story models require n_days") from exc

        story_rate = parameters["story_rate"]
        report_rate = parameters["report_rate"]
        beta_age = parameters["beta_age"]
        beta_appeal = parameters["beta_appeal"]
        if story_rate <= 0 or report_rate <= 0:
            raise ValueError("story_rate and report_rate must be positive")
        if beta_appeal < 0:
            raise ValueError("beta_appeal must be non-negative")

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

        active_count = 0
        for day, birth_count in enumerate(births):
            active_count += int(birth_count)
            if active_count == 0:
                continue

            report_count = int(rng.poisson(report_rate))
            probabilities = story_choice_probabilities(
                day - birth_days[:active_count],
                appeals[:active_count],
                beta_age=beta_age,
                beta_appeal=beta_appeal,
                n_days=n_days,
            )
            mentions[:active_count, day] = rng.multinomial(
                report_count,
                probabilities,
            )

        return {"mentions": mentions}


__all__ = [
    "StoryCompetitionModel",
    "story_choice_probabilities",
]
