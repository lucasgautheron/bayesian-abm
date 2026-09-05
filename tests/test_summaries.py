from __future__ import annotations

import sys
import unittest
from types import ModuleType

import numpy as np

sys.modules.setdefault("bayesflow", ModuleType("bayesflow"))
sys.modules.setdefault("pymc", ModuleType("pymc"))

from base.model import INTERVAL_SECONDS
from base.summaries import (
    SUMMARY_BUILDERS,
    compute_summaries,
    contact_time_coefficient_of_variation,
    cumulative_network_assortativity,
    cumulative_network_average_path_length,
    cumulative_network_clustering,
    cumulative_network_connectivity,
    cumulative_network_degree_coefficient_of_variation,
    integrated_contact_autocorrelation_time,
    make_summaries,
    mean_contact_run_duration,
    mean_contacts_per_bin,
    mean_pair_contact_duration,
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

    def test_mean_contacts_includes_empty_bins(self) -> None:
        data = contacts(
            [
                INTERVAL_SECONDS,
                INTERVAL_SECONDS,
                3 * INTERVAL_SECONDS,
            ],
            [0, 1, 0],
            [1, 2, 2],
        )

        mean = mean_contacts_per_bin(
            start=INTERVAL_SECONDS,
            end=4 * INTERVAL_SECONDS,
        )(data)

        self.assertAlmostEqual(mean, np.mean([2, 0, 1, 0]))

    def test_integrated_autocorrelation_time_is_hand_computed(self) -> None:
        data = contacts(
            [
                INTERVAL_SECONDS,
                INTERVAL_SECONDS,
                INTERVAL_SECONDS,
                2 * INTERVAL_SECONDS,
                2 * INTERVAL_SECONDS,
                3 * INTERVAL_SECONDS,
            ],
            [0, 0, 1, 0, 1, 0],
            [1, 2, 2, 1, 2, 1],
        )
        reordered = {
            "t": data["t"][::-1],
            "i": data["j"][::-1],
            "j": data["i"][::-1],
        }
        bounds = dict(start=INTERVAL_SECONDS, end=4 * INTERVAL_SECONDS)
        counts = np.asarray([3.0, 2.0, 1.0, 0.0])
        centered = counts - counts.mean()
        variance = float(np.dot(centered, centered))
        rho_one = float(np.dot(centered[:-1], centered[1:]) / variance)
        expected = 1.0 + 2.0 * rho_one

        result = integrated_contact_autocorrelation_time(**bounds)(data)
        self.assertAlmostEqual(result, expected)
        self.assertAlmostEqual(expected, 1.5)
        self.assertEqual(
            integrated_contact_autocorrelation_time(**bounds)(reordered),
            result,
        )
        self.assertEqual(np.asarray(result).shape, ())
        self.assertTrue(np.issubdtype(np.asarray(result).dtype, np.floating))

    def test_empty_contacts_produce_zero_activity_summaries(self) -> None:
        empty = contacts([], [], [])

        for factory in (
            mean_contacts_per_bin,
            integrated_contact_autocorrelation_time,
            mean_contact_run_duration,
            mean_pair_contact_duration,
        ):
            self.assertEqual(
                factory(
                    start=INTERVAL_SECONDS,
                    end=3 * INTERVAL_SECONDS,
                )(empty),
                0.0,
            )

    def test_mean_run_and_pair_durations_are_hand_computed(self) -> None:
        data = contacts(
            [
                INTERVAL_SECONDS,
                2 * INTERVAL_SECONDS,
                3 * INTERVAL_SECONDS,
                5 * INTERVAL_SECONDS,
                6 * INTERVAL_SECONDS,
            ],
            [0, 0, 1, 0, 0],
            [1, 1, 2, 1, 1],
        )
        bounds = dict(start=INTERVAL_SECONDS, end=6 * INTERVAL_SECONDS)

        self.assertAlmostEqual(mean_contact_run_duration(**bounds)(data), 5.0 / 3.0)
        self.assertAlmostEqual(mean_pair_contact_duration(**bounds)(data), 2.5)

    def test_duration_summaries_ignore_row_order_and_orientation(self) -> None:
        data = contacts(
            [
                INTERVAL_SECONDS,
                2 * INTERVAL_SECONDS,
                4 * INTERVAL_SECONDS,
            ],
            [0, 1, 0],
            [1, 0, 1],
        )
        reordered = {
            "t": data["t"][::-1],
            "i": data["j"][::-1],
            "j": data["i"][::-1],
        }
        bounds = dict(start=INTERVAL_SECONDS, end=4 * INTERVAL_SECONDS)

        for factory in (mean_contact_run_duration, mean_pair_contact_duration):
            self.assertEqual(factory(**bounds)(data), factory(**bounds)(reordered))

    def test_duration_summaries_reject_out_of_range_and_self_contacts(
        self,
    ) -> None:
        for factory in (mean_contact_run_duration, mean_pair_contact_duration):
            summary = factory(
                start=INTERVAL_SECONDS,
                end=3 * INTERVAL_SECONDS,
            )
            with self.assertRaises(ValueError):
                summary(contacts([4 * INTERVAL_SECONDS], [0], [1]))
            with self.assertRaises(ValueError):
                summary(contacts([INTERVAL_SECONDS], [0], [0]))

    def test_bounds_must_be_aligned(self) -> None:
        for factory in (
            mean_contacts_per_bin,
            integrated_contact_autocorrelation_time,
            mean_contact_run_duration,
            mean_pair_contact_duration,
        ):
            with self.assertRaises(ValueError):
                factory(
                    start=INTERVAL_SECONDS // 2,
                    end=3 * INTERVAL_SECONDS,
                )
            with self.assertRaises(ValueError):
                factory(
                    start=4 * INTERVAL_SECONDS,
                    end=3 * INTERVAL_SECONDS,
                )
        iact = integrated_contact_autocorrelation_time(
            start=INTERVAL_SECONDS,
            end=3 * INTERVAL_SECONDS,
        )
        with self.assertRaises(ValueError):
            iact(contacts([4 * INTERVAL_SECONDS], [0], [1]))


class AgentDistributionTests(unittest.TestCase):
    def test_contact_time_variation_includes_isolated_agents(self) -> None:
        summary = contact_time_coefficient_of_variation([10, 20, 30, 40])
        data = contacts(
            [
                INTERVAL_SECONDS,
                INTERVAL_SECONDS,
                2 * INTERVAL_SECONDS,
                3 * INTERVAL_SECONDS,
            ],
            [10, 10, 10, 20],
            [20, 20, 30, 30],
        )

        result = summary(data)

        totals = np.asarray(
            [
                3 * INTERVAL_SECONDS,
                3 * INTERVAL_SECONDS,
                2 * INTERVAL_SECONDS,
                0,
            ]
        )
        self.assertAlmostEqual(result, totals.std() / totals.mean())

    def test_agent_relabeling_does_not_change_distribution(self) -> None:
        original = contacts(
            [
                INTERVAL_SECONDS,
                INTERVAL_SECONDS,
                2 * INTERVAL_SECONDS,
                3 * INTERVAL_SECONDS,
            ],
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

        first = contact_time_coefficient_of_variation(
            [10, 20, 30, 40]
        )(original)
        second = contact_time_coefficient_of_variation(
            [101, 102, 103, 104]
        )(relabeled)

        self.assertEqual(first, second)

    def test_unknown_agents_are_rejected(self) -> None:
        summary = contact_time_coefficient_of_variation([0, 1])

        with self.assertRaisesRegex(ValueError, "unknown agent"):
            summary(contacts([INTERVAL_SECONDS], [0], [2]))

    def test_empty_contacts_have_zero_contact_time_variation(self) -> None:
        summary = contact_time_coefficient_of_variation([0, 1])

        self.assertEqual(summary(contacts([], [], [])), 0.0)


class NetworkSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent_ids = [0, 1, 2, 3, 4]
        self.data = contacts(
            [
                INTERVAL_SECONDS,
                2 * INTERVAL_SECONDS,
                3 * INTERVAL_SECONDS,
                4 * INTERVAL_SECONDS,
                5 * INTERVAL_SECONDS,
            ],
            [0, 1, 2, 2, 0],
            [1, 2, 0, 3, 1],
        )

    def test_connectivity_and_clustering(self) -> None:
        connectivity = cumulative_network_connectivity(self.agent_ids)
        clustering = cumulative_network_clustering(self.agent_ids)
        assortativity = cumulative_network_assortativity(self.agent_ids)
        path_length = cumulative_network_average_path_length(self.agent_ids)
        degree_cv = cumulative_network_degree_coefficient_of_variation(
            self.agent_ids
        )

        self.assertAlmostEqual(connectivity(self.data), 0.4)
        self.assertAlmostEqual(clustering(self.data), 7.0 / 15.0)
        self.assertAlmostEqual(assortativity(self.data), -5.0 / 7.0)
        self.assertAlmostEqual(path_length(self.data), 4.0 / 3.0)
        self.assertAlmostEqual(
            degree_cv(self.data),
            float(np.std([2, 2, 3, 1, 0]) / np.mean([2, 2, 3, 1, 0])),
        )
        batched = compute_summaries(
            self.data,
            make_summaries(n_agents=5, n_steps=5),
        )
        self.assertAlmostEqual(
            float(batched["cumulative_network_average_path_length"][0]),
            4.0 / 3.0,
        )
        self.assertAlmostEqual(
            float(batched["cumulative_network_degree_coefficient_of_variation"][0]),
            float(np.std([2, 2, 3, 1, 0]) / np.mean([2, 2, 3, 1, 0])),
        )

    def test_empty_network_has_zero_summaries(self) -> None:
        empty = contacts([], [], [])

        for factory in (
            cumulative_network_connectivity,
            cumulative_network_clustering,
            cumulative_network_assortativity,
            cumulative_network_average_path_length,
            cumulative_network_degree_coefficient_of_variation,
        ):
            result = factory(self.agent_ids)(empty)
            self.assertEqual(result, 0.0)
            self.assertEqual(np.asarray(result).shape, ())
            self.assertTrue(np.issubdtype(np.asarray(result).dtype, np.floating))

    def test_assortativity_is_zero_without_degree_variation(self) -> None:
        triangle = contacts(
            [
                INTERVAL_SECONDS,
                2 * INTERVAL_SECONDS,
                3 * INTERVAL_SECONDS,
            ],
            [0, 1, 2],
            [1, 2, 0],
        )

        result = cumulative_network_assortativity([0, 1, 2])(triangle)

        self.assertEqual(result, 0.0)
        self.assertEqual(
            cumulative_network_degree_coefficient_of_variation([0, 1, 2])(
                triangle
            ),
            0.0,
        )

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
            cumulative_network_average_path_length,
            cumulative_network_degree_coefficient_of_variation,
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
            cumulative_network_average_path_length,
            cumulative_network_degree_coefficient_of_variation,
        ):
            with self.assertRaisesRegex(ValueError, "at least two unique"):
                factory([0])
            with self.assertRaisesRegex(ValueError, "at least two unique"):
                factory([0, 0])
            with self.assertRaisesRegex(ValueError, "unknown agent"):
                factory([0, 1])(
                    contacts([INTERVAL_SECONDS], [0], [2])
                )
            with self.assertRaisesRegex(ValueError, "self-contacts"):
                factory([0, 1])(
                    contacts([INTERVAL_SECONDS], [0], [0])
                )


if __name__ == "__main__":
    unittest.main()
