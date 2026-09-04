from __future__ import annotations

import sys
import unittest
from types import ModuleType

import numpy as np

sys.modules.setdefault("bayesflow", ModuleType("bayesflow"))
sys.modules.setdefault("pymc", ModuleType("pymc"))

from base.summaries import (
    SUMMARY_BUILDERS,
    contacts_per_bin,
    cumulative_contact_times,
    cumulative_network_assortativity,
    cumulative_network_clustering,
    cumulative_network_connectivity,
    make_summaries,
    sorted_summary,
)


def contacts(
    times: list[int],
    first: list[int],
    second: list[int],
) -> dict[str, np.ndarray]:
    return {
        "t": np.asarray(times, dtype=np.int32),
        "i": np.asarray(first, dtype=np.int32),
        "j": np.asarray(second, dtype=np.int32),
    }


class TemporalSummaryTests(unittest.TestCase):
    def test_builds_every_registered_summary(self) -> None:
        summaries = make_summaries(n_agents=4, n_steps=3)

        self.assertEqual(set(summaries), set(SUMMARY_BUILDERS))

    def test_summary_context_must_be_valid(self) -> None:
        with self.assertRaises(ValueError):
            make_summaries(n_agents=1, n_steps=3)
        with self.assertRaises(ValueError):
            make_summaries(n_agents=4, n_steps=0)

    def test_counts_contacts_and_preserves_empty_bins(self) -> None:
        summary = contacts_per_bin(start=20, end=80)

        result = summary(contacts([20, 20, 60], [0, 1, 0], [1, 2, 2]))

        np.testing.assert_array_equal(result, [2, 0, 1, 0])

    def test_empty_contacts_produce_all_zeroes(self) -> None:
        summary = contacts_per_bin(start=20, end=60)

        result = summary(contacts([], [], []))

        np.testing.assert_array_equal(result, [0, 0, 0])

    def test_bounds_must_be_aligned(self) -> None:
        with self.assertRaises(ValueError):
            contacts_per_bin(start=10, end=60)
        with self.assertRaises(ValueError):
            contacts_per_bin(start=80, end=60)


class AgentDistributionTests(unittest.TestCase):
    def test_counts_both_endpoints_and_isolated_agents(self) -> None:
        summary = cumulative_contact_times([10, 20, 30, 40])
        data = contacts(
            [20, 20, 40, 60],
            [10, 10, 10, 20],
            [20, 20, 30, 30],
        )

        result = summary(data)

        np.testing.assert_array_equal(result, [0, 40, 60, 60])

    def test_agent_relabeling_does_not_change_distribution(self) -> None:
        original = contacts(
            [20, 20, 40, 60],
            [10, 10, 10, 20],
            [20, 20, 30, 30],
        )
        relabeling = {10: 103, 20: 101, 30: 104, 40: 102}
        relabeled = {
            "t": original["t"],
            "i": np.asarray(
                [relabeling[int(agent)] for agent in original["i"]],
                dtype=np.int32,
            ),
            "j": np.asarray(
                [relabeling[int(agent)] for agent in original["j"]],
                dtype=np.int32,
            ),
        }

        first = cumulative_contact_times([10, 20, 30, 40])(original)
        second = cumulative_contact_times([101, 102, 103, 104])(
            relabeled
        )

        np.testing.assert_array_equal(first, second)

    def test_unknown_agents_are_rejected(self) -> None:
        summary = cumulative_contact_times([0, 1])

        with self.assertRaisesRegex(ValueError, "unknown agent"):
            summary(contacts([20], [0], [2]))

    def test_sorted_summary_requires_a_vector(self) -> None:
        summary = sorted_summary(
            lambda _: np.asarray([[2, 1], [0, 3]])
        )

        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            summary(contacts([], [], []))


class NetworkSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent_ids = [0, 1, 2, 3, 4]
        self.data = contacts(
            [20, 40, 60, 80, 100],
            [0, 1, 2, 2, 0],
            [1, 2, 0, 3, 1],
        )

    def test_connectivity_and_clustering(self) -> None:
        connectivity = cumulative_network_connectivity(self.agent_ids)
        clustering = cumulative_network_clustering(self.agent_ids)
        assortativity = cumulative_network_assortativity(self.agent_ids)

        self.assertAlmostEqual(connectivity(self.data), 0.4)
        self.assertAlmostEqual(clustering(self.data), 7.0 / 15.0)
        self.assertAlmostEqual(assortativity(self.data), -5.0 / 7.0)

    def test_empty_network_has_zero_summaries(self) -> None:
        empty = contacts([], [], [])

        for factory in (
            cumulative_network_connectivity,
            cumulative_network_clustering,
            cumulative_network_assortativity,
        ):
            result = factory(self.agent_ids)(empty)
            self.assertEqual(result, 0.0)
            self.assertEqual(np.asarray(result).shape, ())
            self.assertTrue(np.issubdtype(np.asarray(result).dtype, np.floating))

    def test_assortativity_is_zero_without_degree_variation(self) -> None:
        triangle = contacts(
            [20, 40, 60],
            [0, 1, 2],
            [1, 2, 0],
        )

        result = cumulative_network_assortativity([0, 1, 2])(triangle)

        self.assertEqual(result, 0.0)

    def test_statistics_ignore_row_order_and_agent_labels(self) -> None:
        order = np.asarray([4, 2, 0, 3, 1])
        reordered = {
            key: values[order] for key, values in self.data.items()
        }
        relabeling = {0: 14, 1: 10, 2: 12, 3: 11, 4: 13}
        relabeled = {
            "t": self.data["t"],
            "i": np.asarray(
                [relabeling[int(agent)] for agent in self.data["i"]],
                dtype=np.int32,
            ),
            "j": np.asarray(
                [relabeling[int(agent)] for agent in self.data["j"]],
                dtype=np.int32,
            ),
        }

        for factory in (
            cumulative_network_connectivity,
            cumulative_network_clustering,
            cumulative_network_assortativity,
        ):
            expected = factory(self.agent_ids)(self.data)
            self.assertEqual(factory(self.agent_ids)(reordered), expected)
            self.assertEqual(
                factory(list(relabeling.values()))(relabeled),
                expected,
            )

    def test_invalid_agents_and_unknown_endpoints_are_rejected(self) -> None:
        for factory in (
            cumulative_network_connectivity,
            cumulative_network_clustering,
            cumulative_network_assortativity,
        ):
            with self.assertRaisesRegex(ValueError, "at least two unique"):
                factory([0])
            with self.assertRaisesRegex(ValueError, "at least two unique"):
                factory([0, 0])
            with self.assertRaisesRegex(ValueError, "unknown agent"):
                factory([0, 1])(contacts([20], [0], [2]))


if __name__ == "__main__":
    unittest.main()
