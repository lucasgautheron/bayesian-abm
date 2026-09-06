from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd
import pyarrow.parquet as pq

from data.convert_scientist_conventions import (
    COAUTHORSHIP_SCHEMA,
    CITATIONS_SCHEMA,
    SCIENTISTS_SCHEMA,
    convert,
)


def write_source(root: Path) -> Path:
    source = root / "source"
    database = source / "inspire-harvest" / "database"
    data = source / "data"
    database.mkdir(parents=True)
    data.mkdir()

    pd.DataFrame(
        {
            "article_id": [10, 20, 30],
            "arxiv": ["1234.1", "hep-th/0001", "9999.9"],
            "categories": [
                ["Phenomenology-HEP"],
                ["Theory-HEP", "Astrophysics"],
                ["Gravitation and Cosmology"],
            ],
            "date_created": ["2000-01-01", "2001-01-01", "2002-01-01"],
        }
    ).to_parquet(database / "articles.parquet")
    pd.DataFrame(
        {
            "article_id": [10, 10, 20, 20, 30, 30],
            "bai": ["a", "b", "b", "c", "a", "c"],
        }
    ).to_parquet(database / "articles_authors.parquet")
    pd.DataFrame({"cites": [30], "cited": [20]}).to_parquet(
        database / "articles_references.parquet"
    )
    pd.DataFrame(
        {
            "arxiv": ["1234.1", "hep-th0001"],
            "mostly_minus": [0, 2],
            "mostly_plus": [1, 0],
        }
    ).to_parquet(data / "signature.parquet")
    preferences = pd.DataFrame(
        {"signature": [1, -1]},
        index=pd.Index(["a", "c"], name="bai"),
    )
    preferences.to_parquet(data / "authors_signatures.parquet")
    return source


class ConvertScientistConventionsTests(unittest.TestCase):
    def test_builds_normalized_network_tables(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_source(root)
            output = root / "output"

            stats = convert(source, output)

            self.assertEqual(stats.scientists, 3)
            self.assertEqual(stats.observed_preferences, 2)
            self.assertEqual(stats.coauthorship_edges, 3)
            self.assertEqual(stats.citation_edges, 1)

            scientists = pq.read_table(output / "scientists.parquet")
            coauthorship = pq.read_table(output / "coauthorship.parquet")
            citations = pq.read_table(output / "citations.parquet")
            self.assertEqual(scientists.schema, SCIENTISTS_SCHEMA)
            self.assertEqual(coauthorship.schema, COAUTHORSHIP_SCHEMA)
            self.assertEqual(citations.schema, CITATIONS_SCHEMA)

            scientist_frame = scientists.to_pandas()
            self.assertEqual(
                scientist_frame["source_author_id"].tolist(),
                ["a", "b", "c"],
            )
            self.assertEqual(scientist_frame["favorite_convention"].iloc[0], 1)
            self.assertTrue(
                pd.isna(scientist_frame["favorite_convention"].iloc[1])
            )
            self.assertEqual(scientist_frame["favorite_convention"].iloc[2], -1)
            self.assertEqual(
                scientist_frame["career_start_year"].tolist(),
                [2000, 2000, 2001],
            )
            self.assertEqual(
                scientist_frame["area_share_0"].tolist(),
                [0.5, 0.5, 0.0],
            )
            self.assertEqual(
                scientist_frame["area_share_3"].tolist(),
                [0.0, 0.5, 0.5],
            )

            self.assertEqual(
                coauthorship.to_pandas()[["source", "target"]].values.tolist(),
                [[0, 1], [0, 2], [1, 2]],
            )
            self.assertEqual(
                coauthorship.column("weight").to_pylist(),
                [1.0, 1.0, 1.0],
            )
            self.assertEqual(
                citations.to_pandas().to_dict(orient="records"),
                [{"source": 0, "target": 2, "weight": 0.25}],
            )

    def test_rejects_missing_source_files(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "missing source"):
                convert(Path(directory), Path(directory) / "output")


if __name__ == "__main__":
    unittest.main()
