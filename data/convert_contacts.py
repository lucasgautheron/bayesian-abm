#!/usr/bin/env python3
"""Convert interval contact records to a Parquet file."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq


INTERVAL_SECONDS = 20
DAY_SECONDS = 24 * 60 * 60
BATCH_SIZE = 65_536
INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1

CONTACTS_SCHEMA = pa.schema(
    [
        pa.field("t", pa.int32(), nullable=False),
        pa.field("i", pa.int32(), nullable=False),
        pa.field("j", pa.int32(), nullable=False),
    ],
    metadata={
        b"description": b"Contacts active during the interval (t - 20, t] seconds",
        b"interval_seconds": str(INTERVAL_SECONDS).encode(),
    },
)


class ParseError(ValueError):
    """A malformed contact record."""


@dataclass
class ConversionStats:
    rows: int = 0
    people: int = 0
    min_t: Optional[int] = None
    max_t: Optional[int] = None


def parse_record(line: str, line_number: int) -> tuple[int, int, int]:
    fields = line.split()
    if len(fields) != 3:
        raise ParseError(
            f"line {line_number}: expected 3 whitespace-separated fields, "
            f"found {len(fields)}"
        )

    try:
        t, i, j = (int(field) for field in fields)
    except ValueError as exc:
        raise ParseError(f"line {line_number}: fields must be integers") from exc

    for name, value in (("t", t), ("i", i), ("j", j)):
        if not INT32_MIN <= value <= INT32_MAX:
            raise ParseError(f"line {line_number}: {name}={value} exceeds int32 range")

    if t < INTERVAL_SECONDS or t % INTERVAL_SECONDS:
        raise ParseError(
            f"line {line_number}: t={t} is not a positive "
            f"{INTERVAL_SECONDS}-second interval boundary"
        )
    if i == j:
        raise ParseError(f"line {line_number}: self-contact for person {i}")

    return t, i, j


def convert(
    source: Path,
    output: Path,
    *,
    first_day_only: bool = False,
) -> ConversionStats:
    source = source.resolve()
    output = output.resolve()
    if source == output:
        raise ValueError("source and output paths must differ")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.name}.tmp")
    columns: dict[str, list[int]] = {name: [] for name in CONTACTS_SCHEMA.names}
    people: set[int] = set()
    stats = ConversionStats()
    previous_t: Optional[int] = None
    first_day: Optional[int] = None
    writer: Optional[pq.ParquetWriter] = None

    def flush() -> None:
        if not columns["t"]:
            return
        assert writer is not None
        writer.write_table(pa.Table.from_pydict(columns, schema=CONTACTS_SCHEMA))
        for values in columns.values():
            values.clear()

    try:
        writer = pq.ParquetWriter(
            temporary_output,
            CONTACTS_SCHEMA,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        with source.open("r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                t, i, j = parse_record(line, line_number)
                if previous_t is not None and t < previous_t:
                    raise ParseError(
                        f"line {line_number}: timestamp {t} is earlier than "
                        f"previous timestamp {previous_t}"
                    )
                if first_day is None:
                    first_day = t // DAY_SECONDS
                elif first_day_only and t // DAY_SECONDS != first_day:
                    break
                previous_t = t

                columns["t"].append(t)
                columns["i"].append(i)
                columns["j"].append(j)
                people.update((i, j))
                stats.rows += 1
                stats.min_t = t if stats.min_t is None else min(stats.min_t, t)
                stats.max_t = t if stats.max_t is None else max(stats.max_t, t)

                if len(columns["t"]) >= BATCH_SIZE:
                    flush()

        if not stats.rows:
            raise ParseError("source contains no contact records")
        flush()
        writer.close()
        writer = None
        os.replace(temporary_output, output)
    finally:
        if writer is not None:
            writer.close()
        temporary_output.unlink(missing_ok=True)

    stats.people = len(people)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="whitespace-delimited t i j source file")
    parser.add_argument("output", type=Path, help="destination Parquet file")
    parser.add_argument(
        "--first-day-only",
        action="store_true",
        help="stop before the first contact on the following calendar day",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        stats = convert(
            args.source,
            args.output,
            first_day_only=args.first_day_only,
        )
    except (OSError, UnicodeError, ParseError, ValueError, pa.ArrowException) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {stats.rows:,} contacts involving {stats.people:,} people")
    print(f"Timestamp range: {stats.min_t:,} to {stats.max_t:,} seconds")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
