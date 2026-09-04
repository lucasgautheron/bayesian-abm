from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pyarrow.parquet as pq

from scripts.convert_contacts import DAY_SECONDS, convert


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
            self.assertEqual(stats.max_t, 40)
            self.assertEqual(table.column("t").to_pylist(), [20, 40])

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
            self.assertEqual(stats.max_t, DAY_SECONDS + 20)


if __name__ == "__main__":
    unittest.main()
