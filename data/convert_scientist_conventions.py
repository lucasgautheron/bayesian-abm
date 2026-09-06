#!/usr/bin/env python3
"""Convert the convention paper's DataLad sources to network Parquet files."""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


RESEARCH_AREAS = (
    "Phenomenology-HEP",
    "Theory-HEP",
    "Gravitation and Cosmology",
    "Astrophysics",
)
MAX_AUTHORS_PER_PAPER = 100

SCIENTISTS_SCHEMA = pa.schema(
    [
        pa.field("scientist_id", pa.int32(), nullable=False),
        pa.field("source_author_id", pa.string(), nullable=False),
        pa.field("favorite_convention", pa.int8(), nullable=True),
        pa.field("primary_area", pa.int8(), nullable=False),
        pa.field("career_start_year", pa.int16(), nullable=False),
        *[
            pa.field(f"area_share_{area_id}", pa.float32(), nullable=False)
            for area_id in range(len(RESEARCH_AREAS))
        ],
    ],
    metadata={
        b"description": b"Scientists and metric-signature preferences",
        b"favorite_convention": b"-1 mostly-minus; +1 mostly-plus; null unknown",
        b"research_areas": "|".join(RESEARCH_AREAS).encode(),
    },
)
COAUTHORSHIP_SCHEMA = pa.schema(
    [
        pa.field("source", pa.int32(), nullable=False),
        pa.field("target", pa.int32(), nullable=False),
        pa.field("weight", pa.float64(), nullable=False),
        pa.field("first_year", pa.int16(), nullable=False),
    ],
    metadata={
        b"description": b"Undirected weighted coauthorship network",
        b"weight": b"sum of 1/(number of authors on paper - 1)",
    },
)
CITATIONS_SCHEMA = pa.schema(
    [
        pa.field("source", pa.int32(), nullable=False),
        pa.field("target", pa.int32(), nullable=False),
        pa.field("weight", pa.float64(), nullable=False),
    ],
    metadata={
        b"description": b"Directed weighted scientist citation network",
        b"weight": b"sum of 1/(citing paper authors * cited paper authors)",
    },
)


@dataclass(frozen=True)
class ConversionStats:
    scientists: int
    observed_preferences: int
    coauthorship_edges: int
    citation_edges: int


def format_arxiv_id(value: object) -> str:
    """Return the identifier format used by the INSPIRE article table."""

    text = str(value)
    match = re.fullmatch(r"([A-Za-z-]*)([\d.]+)", text)
    if match is None:
        raise ValueError(f"invalid arXiv identifier: {text!r}")
    prefix, number = match.groups()
    return number if not prefix else f"{prefix}/{number}"


def _source_paths(source: Path) -> dict[str, Path]:
    paths = {
        "signatures": source / "data" / "signature.parquet",
        "author_preferences": source / "data" / "authors_signatures.parquet",
        "articles": source / "inspire-harvest" / "database" / "articles.parquet",
        "authorships": (
            source / "inspire-harvest" / "database" / "articles_authors.parquet"
        ),
        "references": (
            source / "inspire-harvest" / "database" / "articles_references.parquet"
        ),
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"missing source Parquet files: {names}")
    return paths


def _author_preference_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "source_author_id" in frame:
        author_ids = frame["source_author_id"]
    elif "bai" in frame:
        author_ids = frame["bai"]
    elif frame.index.name in {"bai", "source_author_id"}:
        author_ids = frame.index.to_series(index=frame.index)
    else:
        raise ValueError(
            "authors_signatures.parquet must identify authors with a "
            "'bai' column or index"
        )
    if "signature" not in frame:
        raise ValueError("authors_signatures.parquet is missing 'signature'")
    result = pd.DataFrame(
        {
            "source_author_id": author_ids.astype(str).to_numpy(),
            "favorite_convention": frame["signature"].to_numpy(),
        }
    )
    if result["source_author_id"].duplicated().any():
        raise ValueError("authors_signatures.parquet contains duplicate authors")
    if not result["favorite_convention"].isin([-1, 1]).all():
        raise ValueError("author signatures must be -1 or +1")
    return result


def _article_areas(categories: object) -> np.ndarray:
    if not isinstance(categories, (list, tuple, np.ndarray)):
        return np.zeros(len(RESEARCH_AREAS), dtype=np.float64)
    values = set(categories)
    return np.asarray([area in values for area in RESEARCH_AREAS], dtype=np.float64)


def _write_table(frame: pd.DataFrame, schema: pa.Schema, output: Path) -> None:
    temporary = output.with_name(f".{output.name}.tmp")
    table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
    try:
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def convert(source: Path, output_directory: Path) -> ConversionStats:
    """Build scientist, coauthorship, and citation Parquet tables."""

    source = source.resolve()
    output_directory = output_directory.resolve()
    paths = _source_paths(source)
    output_directory.mkdir(parents=True, exist_ok=True)

    articles = pd.read_parquet(
        paths["articles"],
        columns=["article_id", "arxiv", "categories", "date_created"],
    )
    authorships = pd.read_parquet(
        paths["authorships"],
        columns=["article_id", "bai"],
    ).dropna(subset=["article_id", "bai"])
    references = pd.read_parquet(
        paths["references"],
        columns=["cites", "cited"],
    ).dropna(subset=["cites", "cited"])
    signatures = pd.read_parquet(paths["signatures"])

    required_signature_columns = {"arxiv", "mostly_minus", "mostly_plus"}
    missing_signature_columns = required_signature_columns.difference(
        signatures.columns
    )
    if missing_signature_columns:
        names = ", ".join(sorted(missing_signature_columns))
        raise ValueError(f"signature.parquet is missing columns: {names}")

    articles = articles.dropna(subset=["article_id"]).copy()
    articles["article_id"] = articles["article_id"].astype(np.int64)
    authorships = authorships.copy()
    authorships["article_id"] = authorships["article_id"].astype(np.int64)
    authorships["source_author_id"] = authorships["bai"].astype(str)
    references = references.astype({"cites": np.int64, "cited": np.int64})

    article_arxiv = articles["arxiv"].fillna("").astype(str)
    article_lookup = dict(zip(article_arxiv, articles["article_id"], strict=False))
    labeled_article_ids: set[int] = set()
    informative = signatures[
        signatures["mostly_minus"] != signatures["mostly_plus"]
    ]
    for raw_id in informative["arxiv"]:
        normalized = format_arxiv_id(raw_id)
        article_id = article_lookup.get(normalized)
        if article_id is not None:
            labeled_article_ids.add(int(article_id))
    if not labeled_article_ids:
        raise ValueError("no convention-labelled papers matched INSPIRE articles")

    selected_authors = set(
        authorships.loc[
            authorships["article_id"].isin(labeled_article_ids),
            "source_author_id",
        ]
    )
    if not selected_authors:
        raise ValueError("no scientists occur on convention-labelled papers")

    article_author_counts = (
        authorships.groupby("article_id")["source_author_id"].size().astype(int)
    )
    articles = articles.copy()
    year_text = articles["date_created"].fillna("").astype(str).str[:4]
    valid_year = year_text.str.fullmatch(r"\d{4}")
    articles = articles.loc[valid_year].copy()
    articles["year"] = year_text.loc[valid_year].astype(np.int16)
    article_year = dict(
        zip(articles["article_id"], articles["year"], strict=False)
    )
    article_categories = {
        int(row.article_id): _article_areas(row.categories)
        for row in articles.itertuples()
    }

    selected_authorships = authorships[
        authorships["source_author_id"].isin(selected_authors)
        & authorships["article_id"].isin(article_year)
    ].copy()
    if selected_authorships.empty:
        raise ValueError("selected scientists have no dated publications")

    area_totals = {
        author: np.zeros(len(RESEARCH_AREAS), dtype=np.float64)
        for author in selected_authors
    }
    publication_counts = {author: 0 for author in selected_authors}
    career_start = {author: np.iinfo(np.int16).max for author in selected_authors}
    for row in selected_authorships.itertuples():
        author = row.source_author_id
        article_id = int(row.article_id)
        area_totals[author] += article_categories[article_id]
        publication_counts[author] += 1
        career_start[author] = min(career_start[author], int(article_year[article_id]))

    unusable = [
        author
        for author in selected_authors
        if np.isclose(area_totals[author].sum(), 0.0)
        or career_start[author] == np.iinfo(np.int16).max
    ]
    if unusable:
        raise ValueError(
            f"{len(unusable)} selected scientists lack dated research-area data"
        )

    ordered_authors = sorted(selected_authors)
    scientist_ids = {
        author: scientist_id
        for scientist_id, author in enumerate(ordered_authors)
    }
    preferences = _author_preference_frame(paths["author_preferences"])
    preference_lookup = dict(
        zip(
            preferences["source_author_id"],
            preferences["favorite_convention"],
            strict=False,
        )
    )
    scientist_rows: list[dict[str, object]] = []
    for author in ordered_authors:
        shares = area_totals[author] / publication_counts[author]
        preference = preference_lookup.get(author, pd.NA)
        row: dict[str, object] = {
            "scientist_id": scientist_ids[author],
            "source_author_id": author,
            "favorite_convention": preference,
            "primary_area": int(np.argmax(shares)),
            "career_start_year": career_start[author],
        }
        row.update(
            {
                f"area_share_{area_id}": np.float32(shares[area_id])
                for area_id in range(len(RESEARCH_AREAS))
            }
        )
        scientist_rows.append(row)
    scientists = pd.DataFrame(scientist_rows)
    scientists["favorite_convention"] = scientists[
        "favorite_convention"
    ].astype("Int8")

    eligible_article_ids = {
        int(article_id)
        for article_id, count in article_author_counts.items()
        if 1 < count <= MAX_AUTHORS_PER_PAPER and article_id in article_year
    }
    coauthor_weights: dict[tuple[int, int], float] = {}
    first_years: dict[tuple[int, int], int] = {}
    coauthor_source = selected_authorships[
        selected_authorships["article_id"].isin(eligible_article_ids)
    ]
    for article_id, group in coauthor_source.groupby("article_id", sort=False):
        authors = sorted(set(group["source_author_id"]))
        denominator = int(article_author_counts.loc[article_id]) - 1
        increment = 1.0 / denominator
        year = int(article_year[int(article_id)])
        for first, second in combinations(authors, 2):
            edge = (scientist_ids[first], scientist_ids[second])
            coauthor_weights[edge] = coauthor_weights.get(edge, 0.0) + increment
            first_years[edge] = min(first_years.get(edge, year), year)
    coauthorship = pd.DataFrame(
        [
            {
                "source": source_id,
                "target": target_id,
                "weight": weight,
                "first_year": first_years[(source_id, target_id)],
            }
            for (source_id, target_id), weight in sorted(coauthor_weights.items())
        ],
        columns=COAUTHORSHIP_SCHEMA.names,
    )

    connected_old_ids = np.unique(
        coauthorship[["source", "target"]].to_numpy(dtype=np.int32)
    )
    connected_authors = {
        ordered_authors[int(scientist_id)]
        for scientist_id in connected_old_ids
    }
    old_to_new = {
        int(old_id): new_id
        for new_id, old_id in enumerate(connected_old_ids)
    }
    scientists = scientists[
        scientists["scientist_id"].isin(connected_old_ids)
    ].copy()
    scientists["scientist_id"] = scientists["scientist_id"].map(old_to_new)
    scientist_ids = {
        author: scientist_id
        for scientist_id, author in enumerate(sorted(connected_authors))
    }
    coauthorship["source"] = coauthorship["source"].map(old_to_new)
    coauthorship["target"] = coauthorship["target"].map(old_to_new)

    observed_authors = connected_authors.intersection(preference_lookup)
    paper_authors = (
        authorships[
            authorships["article_id"].isin(eligible_article_ids)
            & authorships["source_author_id"].isin(observed_authors)
        ]
        .groupby("article_id")["source_author_id"]
        .agg(lambda values: tuple(sorted(set(values))))
        .to_dict()
    )
    relevant_references = references[
        references["cites"].isin(eligible_article_ids)
        & references["cited"].isin(eligible_article_ids)
    ]
    citation_weights: dict[tuple[int, int], float] = {}
    for row in relevant_references.itertuples(index=False):
        citing_id = int(row.cites)
        cited_id = int(row.cited)
        citing_authors = paper_authors.get(citing_id, ())
        cited_authors = paper_authors.get(cited_id, ())
        if not citing_authors or not cited_authors:
            continue
        increment = 1.0 / (
            int(article_author_counts.loc[citing_id])
            * int(article_author_counts.loc[cited_id])
        )
        for citing_author in citing_authors:
            for cited_author in cited_authors:
                if citing_author == cited_author:
                    continue
                edge = (
                    scientist_ids[citing_author],
                    scientist_ids[cited_author],
                )
                citation_weights[edge] = citation_weights.get(edge, 0.0) + increment
    citations = pd.DataFrame(
        [
            {"source": source_id, "target": target_id, "weight": weight}
            for (source_id, target_id), weight in sorted(citation_weights.items())
        ],
        columns=CITATIONS_SCHEMA.names,
    )

    _write_table(
        scientists,
        SCIENTISTS_SCHEMA,
        output_directory / "scientists.parquet",
    )
    _write_table(
        coauthorship,
        COAUTHORSHIP_SCHEMA,
        output_directory / "coauthorship.parquet",
    )
    _write_table(
        citations,
        CITATIONS_SCHEMA,
        output_directory / "citations.parquet",
    )
    return ConversionStats(
        scientists=len(scientists),
        observed_preferences=int(scientists["favorite_convention"].notna().sum()),
        coauthorship_edges=len(coauthorship),
        citation_edges=len(citations),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        type=Path,
        help="DataLad checkout of dilemmas-conventions",
    )
    parser.add_argument(
        "output_directory",
        type=Path,
        help="directory for generated Parquet files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        stats = convert(args.source, args.output_directory)
    except (OSError, ValueError, pa.ArrowException) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Wrote {stats.scientists:,} scientists "
        f"({stats.observed_preferences:,} observed preferences), "
        f"{stats.coauthorship_edges:,} coauthorship edges, and "
        f"{stats.citation_edges:,} citation edges"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
