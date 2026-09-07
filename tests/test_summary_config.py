from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from base.summary_config import (
    RECOMMENDED_SUMMARY_NAMES,
    SummaryConfigurationError,
    available_summary_names,
    format_summary_configuration_error,
    load_summary_names,
    recommended_summary_names,
)
from base.summaries import select_summaries


class SummaryConfigurationTests(unittest.TestCase):
    def test_loads_ordered_multiline_selection(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "summary.ini"
            path.write_text(
                "[contacts]\n"
                "enabled =\n"
                "    cumulative_network_connectivity\n"
                "    mean_contacts_per_bin\n",
                encoding="utf-8",
            )

            selected = load_summary_names("contacts", path)

        self.assertEqual(
            selected,
            (
                "cumulative_network_connectivity",
                "mean_contacts_per_bin",
            ),
        )

    def test_accepts_a_comma_separated_enabled_list(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "summary.ini"
            path.write_text(
                "[story_daily]\n"
                "enabled = mention_concentration, total_mentions\n",
                encoding="utf-8",
            )

            selected = load_summary_names("story_daily", path)

        self.assertEqual(
            selected,
            ("mention_concentration", "total_mentions"),
        )

    def test_missing_file_has_cursor_guidance_and_available_names(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "summary.ini"

            with self.assertRaises(SummaryConfigurationError) as raised:
                load_summary_names("contacts", path)

        message = str(raised.exception)
        self.assertIn("/configure-summary-stats", message)
        self.assertIn("[contacts]", message)
        self.assertIn("Recommended starting set:", message)
        self.assertIn("Available statistics:", message)
        self.assertIn("mean_contacts_per_bin", message)
        self.assertIn(
            "enabled =\n        <explicitly chosen registered statistic>",
            message,
        )

    def test_formats_only_the_first_error_line_in_red(self) -> None:
        error = SummaryConfigurationError(
            "Summary statistics have not been configured.\n"
            "Ask Cursor to choose summary statistics."
        )

        colored = format_summary_configuration_error(error, color=True)
        plain = format_summary_configuration_error(error, color=False)

        self.assertEqual(
            colored,
            "\033[31mSummary statistics have not been configured.\033[0m\n"
            "Ask Cursor to choose summary statistics.",
        )
        self.assertEqual(plain, str(error))

    def test_rejects_missing_or_invalid_dataset_selection(self) -> None:
        cases = {
            "missing section": "[story_daily]\nenabled = total_mentions\n",
            "missing option": "[contacts]\nother = value\n",
            "empty selection": "[contacts]\nenabled =\n",
            "duplicate selection": (
                "[contacts]\n"
                "enabled = mean_contacts_per_bin, mean_contacts_per_bin\n"
            ),
            "unknown selection": "[contacts]\nenabled = not_registered\n",
            "malformed ini": "[contacts\nenabled = value\n",
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "summary.ini"
            for label, contents in cases.items():
                with self.subTest(label=label):
                    path.write_text(contents, encoding="utf-8")
                    with self.assertRaises(SummaryConfigurationError):
                        load_summary_names("contacts", path)

    def test_lists_each_dataset_registry(self) -> None:
        self.assertIn(
            "mean_contacts_per_bin",
            available_summary_names("contacts"),
        )
        self.assertIn(
            "total_mentions",
            available_summary_names("story_daily"),
        )
        self.assertIn(
            "citation_coupling",
            available_summary_names("scientist_conventions"),
        )

    def test_recommended_sets_are_valid_ordered_registry_subsets(self) -> None:
        expected = {
            "contacts": (
                "cumulative_network_connectivity",
                "cumulative_network_clustering",
                "cumulative_network_degree_variance",
            ),
            "story_daily": (
                "total_mentions",
                "mention_concentration",
                "mean_story_autocorrelation",
            ),
            "scientist_conventions": (
                "coauthorship_coupling",
                "citation_coupling",
                "field_theory_hep",
            ),
        }

        self.assertEqual(RECOMMENDED_SUMMARY_NAMES, expected)
        for dataset, names in expected.items():
            with self.subTest(dataset=dataset):
                self.assertEqual(recommended_summary_names(dataset), names)
                self.assertTrue(
                    set(names).issubset(available_summary_names(dataset))
                )

    def test_all_registered_statistics_remain_valid_without_an_upper_limit(
        self,
    ) -> None:
        names = available_summary_names("contacts")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "summary.ini"
            path.write_text(
                "[contacts]\nenabled = " + ", ".join(names) + "\n",
                encoding="utf-8",
            )

            selected = load_summary_names("contacts", path)

        self.assertEqual(selected, names)

    def test_stale_degree_cv_name_reports_the_variance_replacement(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "summary.ini"
            path.write_text(
                "[contacts]\n"
                "enabled = "
                "cumulative_network_degree_coefficient_of_variation\n",
                encoding="utf-8",
            )

            with self.assertRaises(SummaryConfigurationError) as raised:
                load_summary_names("contacts", path)

        message = str(raised.exception)
        self.assertIn(
            "cumulative_network_degree_coefficient_of_variation",
            message,
        )
        self.assertIn("cumulative_network_degree_variance", message)

    def test_selects_functions_in_requested_order(self) -> None:
        first = lambda data: data
        second = lambda data: data

        selected = select_summaries(
            {"first": first, "second": second},
            ("second", "first"),
        )

        self.assertEqual(tuple(selected), ("second", "first"))
        self.assertIs(selected["first"], first)
        with self.assertRaisesRegex(ValueError, "at least one"):
            select_summaries({"first": first}, ())
        with self.assertRaisesRegex(ValueError, "unknown"):
            select_summaries({"first": first}, ("second",))


if __name__ == "__main__":
    unittest.main()
