from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pyarrow.parquet as pq

from data.convert_contacts import DAY_SECONDS, INTERVAL_SECONDS, convert


class ConvertContactsTests(unittest.TestCase):
    def test_first_day_only_stops_before_following_day(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "contacts.dat"
            output = root / "contacts.parquet"
            source.write_text(
                "20 1 2\n"
                "40 2 3\n"
                f"{DAY_SECONDS + 20} 3 4\n",
                encoding="utf-8",
            )

            stats = convert(source, output, first_day_only=True)
            table = pq.read_table(output)

            self.assertEqual(stats.rows, 2)
            self.assertEqual(stats.people, 3)
            self.assertEqual(stats.max_t, INTERVAL_SECONDS)
            self.assertEqual(
                table.column("t").to_pylist(),
                [INTERVAL_SECONDS, INTERVAL_SECONDS],
            )

    def test_default_conversion_keeps_all_days(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "contacts.dat"
            output = root / "contacts.parquet"
            source.write_text(
                "20 1 2\n"
                f"{DAY_SECONDS + 20} 3 4\n",
                encoding="utf-8",
            )

            stats = convert(source, output)

            self.assertEqual(stats.rows, 2)
            self.assertEqual(stats.max_t, DAY_SECONDS + INTERVAL_SECONDS)

    def test_aggregates_source_intervals_and_undirected_pairs(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "contacts.dat"
            output = root / "contacts.parquet"
            source.write_text(
                "20 1 2\n"
                "40 1 2\n"
                "40 2 1\n"
                "60 2 3\n"
                "80 1 2\n",
                encoding="utf-8",
            )

            stats = convert(source, output)
            table = pq.read_table(output)

            self.assertEqual(stats.rows, 3)
            self.assertEqual(table.column("t").to_pylist(), [60, 60, 120])
            self.assertEqual(table.column("i").to_pylist(), [1, 2, 1])
            self.assertEqual(table.column("j").to_pylist(), [2, 3, 2])
            self.assertEqual(
                table.schema.metadata[b"interval_seconds"],
                b"60",
            )


if __name__ == "__main__":
    unittest.main()
