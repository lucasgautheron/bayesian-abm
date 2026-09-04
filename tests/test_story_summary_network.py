from __future__ import annotations

import unittest

import numpy as np


try:
    import bayesflow as bf
    import keras

    HAS_NETWORK_BACKEND = (
        hasattr(bf, "networks")
        and hasattr(bf.networks, "TimeSeriesNetwork")
        and hasattr(keras, "ops")
    )
except ImportError:
    HAS_NETWORK_BACKEND = False


@unittest.skipUnless(
    HAS_NETWORK_BACKEND,
    "BayesFlow and a Keras backend are required",
)
class StoryPopulationSummaryNetworkTests(unittest.TestCase):
    def test_output_is_fixed_size_and_story_permutation_invariant(self) -> None:
        from models.stories.summary_network import (
            StoryPopulationSummaryNetwork,
        )

        network = StoryPopulationSummaryNetwork(
            time_summary_dim=4,
            population_summary_dim=6,
        )
        values = np.random.default_rng(4).normal(size=(2, 3, 12))
        mask = np.asarray([[1, 1, 0], [1, 1, 1]], dtype=np.float32)

        first = np.asarray(network(values, mask=mask, training=False))
        permutation = [2, 0, 1]
        second = np.asarray(
            network(
                values[:, permutation],
                mask=mask[:, permutation],
                training=False,
            )
        )

        self.assertEqual(first.shape, (2, 6))
        self.assertTrue(np.all(np.isfinite(first)))
        np.testing.assert_allclose(first, second, atol=1e-5)

    def test_summary_dimensions_must_be_positive(self) -> None:
        from models.stories.summary_network import (
            StoryPopulationSummaryNetwork,
        )

        with self.assertRaisesRegex(ValueError, "positive"):
            StoryPopulationSummaryNetwork(time_summary_dim=0)


if __name__ == "__main__":
    unittest.main()
