"""Learned hierarchical summaries for populations of story time series."""

from __future__ import annotations

from typing import Any

import bayesflow as bf
import keras
from bayesflow.utils.serialization import serializable, serialize


@serializable("bayesian_modelling.networks")
class StoryPopulationSummaryNetwork(bf.networks.SummaryNetwork):
    """Encode each story over time, then aggregate stories invariantly."""

    def __init__(
        self,
        *,
        time_summary_dim: int = 16,
        population_summary_dim: int = 32,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if time_summary_dim < 1 or population_summary_dim < 1:
            raise ValueError("summary dimensions must be positive")

        self.time_summary_dim = int(time_summary_dim)
        self.population_summary_dim = int(population_summary_dim)
        self.summary_dim = self.population_summary_dim
        self.time_encoder = keras.layers.TimeDistributed(
            bf.networks.TimeSeriesNetwork(
                summary_dim=self.time_summary_dim,
            ),
            name="story_time_encoder",
        )
        self.population_encoder = bf.networks.DeepSet(
            summary_dim=self.population_summary_dim,
            name="story_set_encoder",
        )

    def call(
        self,
        x: Any,
        training: bool = False,
        mask: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Return one learned summary vector per story population."""

        del kwargs
        time_series = keras.ops.expand_dims(x, axis=-1)
        embeddings = self.time_encoder(time_series, training=training)

        if mask is None:
            valid = keras.ops.ones_like(embeddings[..., 0])
        else:
            valid = keras.ops.cast(mask, embeddings.dtype)
        valid = keras.ops.expand_dims(valid, axis=-1)

        # The BayesFlow DeepSet does not consume a padding mask directly.
        # Zero invalid embeddings and expose validity as an explicit feature.
        set_features = keras.ops.concatenate(
            (embeddings * valid, valid),
            axis=-1,
        )
        return self.population_encoder(set_features, training=training)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        return config | serialize(
            {
                "time_summary_dim": self.time_summary_dim,
                "population_summary_dim": self.population_summary_dim,
            }
        )


__all__ = ["StoryPopulationSummaryNetwork"]
