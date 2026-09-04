#!/usr/bin/env python3
"""Convert nested MemeTracker phrase clusters to analysis-ready Parquet files."""

from __future__ import annotations

import argparse
import gzip
import os
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Optional

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_START_DATE = date(2008, 8, 1)
DEFAULT_END_DATE = date(2009, 1, 31)
BLOG = "B"
MAINSTREAM = "M"

STORIES_SCHEMA = pa.schema(
    [
        pa.field("story_id", pa.int64(), nullable=False),
        pa.field("representative_phrase", pa.string(), nullable=False),
        pa.field("phrase_count", pa.int32(), nullable=False),
        pa.field("total_mentions", pa.int64(), nullable=False),
    ]
)

PHRASES_SCHEMA = pa.schema(
    [
        pa.field("phrase_id", pa.int64(), nullable=False),
        pa.field("story_id", pa.int64(), nullable=False),
        pa.field("phrase", pa.string(), nullable=False),
        pa.field("total_mentions", pa.int64(), nullable=False),
        pa.field("url_count", pa.int32(), nullable=False),
    ]
)

DAILY_SCHEMA = pa.schema(
    [
        pa.field("story_id", pa.int64(), nullable=False),
        pa.field("date", pa.date32(), nullable=False),
        pa.field("mentions", pa.int64(), nullable=False),
        pa.field("blog_mentions", pa.int64(), nullable=False),
        pa.field("mainstream_mentions", pa.int64(), nullable=False),
        pa.field("urls", pa.int32(), nullable=False),
        pa.field("blog_urls", pa.int32(), nullable=False),
        pa.field("mainstream_urls", pa.int32(), nullable=False),
        pa.field("active_phrases", pa.int32(), nullable=False),
    ]
)


class ParseError(ValueError):
    """A malformed or internally inconsistent input record."""


class BufferedParquetWriter:
    """Write dictionaries to Parquet in bounded-size record batches."""

    def __init__(
        self,
        path: Path,
        schema: pa.Schema,
        *,
        batch_size: int = 65_536,
    ) -> None:
        self.path = path
        self.schema = schema
        self.batch_size = batch_size
        self.columns: dict[str, list[Any]] = {name: [] for name in schema.names}
        self.writer = pq.ParquetWriter(
            path,
            schema,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )

    def append(self, row: dict[str, Any]) -> None:
        for name in self.schema.names:
            self.columns[name].append(row[name])
        if len(self) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not len(self):
            return
        table = pa.Table.from_pydict(self.columns, schema=self.schema)
        self.writer.write_table(table)
        self.columns = {name: [] for name in self.schema.names}

    def close(self) -> None:
        self.flush()
        self.writer.close()

    def __len__(self) -> int:
        return len(self.columns[self.schema.names[0]])


@dataclass
class DayStats:
    mentions: int = 0
    blog_mentions: int = 0
    mainstream_mentions: int = 0
    urls: set[str] = field(default_factory=set)
    blog_urls: set[str] = field(default_factory=set)
    mainstream_urls: set[str] = field(default_factory=set)
    active_phrases: set[int] = field(default_factory=set)


@dataclass
class Phrase:
    phrase_id: int
    story_id: int
    text: str
    declared_mentions: int
    declared_urls: int
    observed_mentions: int = 0
    observed_urls: int = 0


@dataclass
class Story:
    story_id: int
    representative_phrase: str
    declared_phrase_count: int
    declared_mentions: int
    observed_mentions: int = 0
    observed_phrase_count: int = 0
    days: dict[date, DayStats] = field(default_factory=dict)


@dataclass
class ConversionCounts:
    stories: int = 0
    phrases: int = 0
    observations: int = 0
    mentions: int = 0
    daily_rows: int = 0


def date_range(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def parse_text_record(raw: str, line_number: int, record_name: str) -> tuple[int, int, str, int]:
    parts = raw.split("\t")
    if len(parts) < 4:
        raise ParseError(
            f"line {line_number}: {record_name} record has {len(parts)} fields; expected 4"
        )
    try:
        first = int(parts[0])
        second = int(parts[1])
        text = "\t".join(parts[2:-1])
        record_id = int(parts[-1])
    except ValueError as exc:
        raise ParseError(f"line {line_number}: invalid {record_name} record: {raw!r}") from exc
    return first, second, text, record_id


def finish_phrase(
    phrase: Optional[Phrase],
    phrase_writer: BufferedParquetWriter,
    line_number: int,
) -> None:
    if phrase is None:
        return
    if phrase.observed_mentions != phrase.declared_mentions:
        raise ParseError(
            f"line {line_number}: phrase {phrase.phrase_id} declares "
            f"{phrase.declared_mentions} mentions but contains {phrase.observed_mentions}"
        )
    if phrase.observed_urls != phrase.declared_urls:
        raise ParseError(
            f"line {line_number}: phrase {phrase.phrase_id} declares "
            f"{phrase.declared_urls} URLs but contains {phrase.observed_urls}"
        )
    phrase_writer.append(
        {
            "phrase_id": phrase.phrase_id,
            "story_id": phrase.story_id,
            "phrase": phrase.text,
            "total_mentions": phrase.declared_mentions,
            "url_count": phrase.declared_urls,
        }
    )


def finish_story(
    story: Optional[Story],
    story_writer: BufferedParquetWriter,
    daily_writer: BufferedParquetWriter,
    all_dates: list[date],
    counts: ConversionCounts,
    line_number: int,
) -> None:
    if story is None:
        return
    if story.observed_phrase_count != story.declared_phrase_count:
        raise ParseError(
            f"line {line_number}: story {story.story_id} declares "
            f"{story.declared_phrase_count} phrases but contains "
            f"{story.observed_phrase_count}"
        )
    if story.observed_mentions != story.declared_mentions:
        raise ParseError(
            f"line {line_number}: story {story.story_id} declares "
            f"{story.declared_mentions} mentions but contains {story.observed_mentions}"
        )

    story_writer.append(
        {
            "story_id": story.story_id,
            "representative_phrase": story.representative_phrase,
            "phrase_count": story.declared_phrase_count,
            "total_mentions": story.declared_mentions,
        }
    )

    for day in all_dates:
        stats = story.days.get(day)
        daily_writer.append(
            {
                "story_id": story.story_id,
                "date": day,
                "mentions": stats.mentions if stats else 0,
                "blog_mentions": stats.blog_mentions if stats else 0,
                "mainstream_mentions": stats.mainstream_mentions if stats else 0,
                "urls": len(stats.urls) if stats else 0,
                "blog_urls": len(stats.blog_urls) if stats else 0,
                "mainstream_urls": len(stats.mainstream_urls) if stats else 0,
                "active_phrases": len(stats.active_phrases) if stats else 0,
            }
        )
        counts.daily_rows += 1
    counts.stories += 1


def convert(source: Path, output_dir: Path, start: date, end: date) -> ConversionCounts:
    if start > end:
        raise ValueError(f"start date {start} is after end date {end}")
    if not source.is_file():
        raise FileNotFoundError(f"source file does not exist: {source}")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_dates = list(date_range(start, end))
    final_paths = {
        "stories": output_dir / "stories.parquet",
        "phrases": output_dir / "phrases.parquet",
        "daily": output_dir / "story_daily.parquet",
    }
    temporary_paths = {name: path.with_suffix(".parquet.tmp") for name, path in final_paths.items()}
    for path in temporary_paths.values():
        path.unlink(missing_ok=True)

    writers: list[BufferedParquetWriter] = []
    success = False
    counts = ConversionCounts()
    try:
        story_writer = BufferedParquetWriter(temporary_paths["stories"], STORIES_SCHEMA)
        writers.append(story_writer)
        phrase_writer = BufferedParquetWriter(temporary_paths["phrases"], PHRASES_SCHEMA)
        writers.append(phrase_writer)
        daily_writer = BufferedParquetWriter(temporary_paths["daily"], DAILY_SCHEMA)
        writers.append(daily_writer)

        story: Optional[Story] = None
        phrase: Optional[Phrase] = None
        seen_story_ids: set[int] = set()
        seen_phrase_ids: set[int] = set()
        line_number = 0

        with gzip.open(source, "rt", encoding="utf-8", errors="strict", newline="") as handle:
            for line_number, line in enumerate(handle, 1):
                raw = line.rstrip("\r\n")
                if not raw:
                    continue
                if story is None and (
                    raw == "format:" or raw.lstrip("\t").startswith("<")
                ):
                    continue

                indentation = len(raw) - len(raw.lstrip("\t"))
                record = raw[indentation:]

                if indentation == 0:
                    finish_phrase(phrase, phrase_writer, line_number)
                    phrase = None
                    finish_story(
                        story,
                        story_writer,
                        daily_writer,
                        all_dates,
                        counts,
                        line_number,
                    )
                    phrase_count, total_mentions, root, story_id = parse_text_record(
                        record, line_number, "story"
                    )
                    if story_id in seen_story_ids:
                        raise ParseError(f"line {line_number}: duplicate story ID {story_id}")
                    seen_story_ids.add(story_id)
                    story = Story(story_id, root, phrase_count, total_mentions)

                elif indentation == 1:
                    if story is None:
                        raise ParseError(f"line {line_number}: phrase appears before a story")
                    finish_phrase(phrase, phrase_writer, line_number)
                    total_mentions, url_count, text, phrase_id = parse_text_record(
                        record, line_number, "phrase"
                    )
                    if phrase_id in seen_phrase_ids:
                        raise ParseError(f"line {line_number}: duplicate phrase ID {phrase_id}")
                    seen_phrase_ids.add(phrase_id)
                    phrase = Phrase(phrase_id, story.story_id, text, total_mentions, url_count)
                    story.observed_phrase_count += 1
                    counts.phrases += 1

                elif indentation == 2:
                    if story is None or phrase is None:
                        raise ParseError(
                            f"line {line_number}: URL observation appears before a phrase"
                        )
                    parts = record.split("\t", 3)
                    if len(parts) != 4:
                        raise ParseError(
                            f"line {line_number}: URL observation has {len(parts)} fields; "
                            "expected 4"
                        )
                    timestamp, frequency_text, url_type, url = parts
                    try:
                        observation_date = date.fromisoformat(timestamp[:10])
                        frequency = int(frequency_text)
                    except (ValueError, IndexError) as exc:
                        raise ParseError(
                            f"line {line_number}: invalid URL observation: {record!r}"
                        ) from exc
                    if observation_date < start or observation_date > end:
                        raise ParseError(
                            f"line {line_number}: date {observation_date} is outside "
                            f"{start} through {end}"
                        )
                    if url_type not in (BLOG, MAINSTREAM):
                        raise ParseError(
                            f"line {line_number}: unknown URL type {url_type!r}"
                        )
                    if frequency < 0:
                        raise ParseError(
                            f"line {line_number}: negative mention frequency {frequency}"
                        )

                    stats = story.days.setdefault(observation_date, DayStats())
                    stats.mentions += frequency
                    stats.urls.add(url)
                    stats.active_phrases.add(phrase.phrase_id)
                    if url_type == BLOG:
                        stats.blog_mentions += frequency
                        stats.blog_urls.add(url)
                    else:
                        stats.mainstream_mentions += frequency
                        stats.mainstream_urls.add(url)

                    phrase.observed_mentions += frequency
                    phrase.observed_urls += 1
                    story.observed_mentions += frequency
                    counts.observations += 1
                    counts.mentions += frequency

                else:
                    raise ParseError(
                        f"line {line_number}: unexpected indentation of {indentation} tabs"
                    )

        finish_phrase(phrase, phrase_writer, line_number + 1)
        finish_story(
            story,
            story_writer,
            daily_writer,
            all_dates,
            counts,
            line_number + 1,
        )
        for writer in writers:
            writer.close()
        writers.clear()

        for name, final_path in final_paths.items():
            os.replace(temporary_paths[name], final_path)
        success = True
        return counts
    finally:
        for writer in writers:
            writer.close()
        if not success:
            for path in temporary_paths.values():
                path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="nested MemeTracker .txt.gz source")
    parser.add_argument("output_dir", type=Path, help="directory for the three Parquet files")
    parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        default=DEFAULT_START_DATE,
        help=f"first dense-panel date (default: {DEFAULT_START_DATE})",
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        default=DEFAULT_END_DATE,
        help=f"last dense-panel date (default: {DEFAULT_END_DATE})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        counts = convert(args.source, args.output_dir, args.start_date, args.end_date)
    except (OSError, UnicodeError, ParseError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {counts.stories:,} stories")
    print(f"Wrote {counts.phrases:,} phrases")
    print(f"Aggregated {counts.observations:,} URL observations")
    print(f"Preserved {counts.mentions:,} mentions")
    print(f"Wrote {counts.daily_rows:,} dense daily rows")
    print(f"Output directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
