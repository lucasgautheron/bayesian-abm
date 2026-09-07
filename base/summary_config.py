"""Local, explicit selection of registered summary statistics."""

from __future__ import annotations

import configparser
from pathlib import Path
import re
from typing import Final

from base.summaries import validate_summary_names
from datasets.contacts.summaries import SUMMARY_BUILDERS
from datasets.scientist_conventions.schema import (
    SCIENTIST_CONVENTIONS_DATASET,
)
from datasets.scientist_conventions.summaries import SCIENTIST_SUMMARY_NAMES
from datasets.story_daily.schema import STORY_DATASET
from datasets.story_daily.summaries import STORY_SUMMARY_STATISTICS


ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_CONFIG_PATH: Final = ROOT / ".config" / "summary.ini"
AVAILABLE_SUMMARY_NAMES: Final[dict[str, tuple[str, ...]]] = {
    "contacts": tuple(SUMMARY_BUILDERS),
    STORY_DATASET: tuple(STORY_SUMMARY_STATISTICS),
    SCIENTIST_CONVENTIONS_DATASET: tuple(SCIENTIST_SUMMARY_NAMES),
}


class SummaryConfigurationError(ValueError):
    """An actionable error in the attendee's local summary selection."""


def available_summary_names(dataset: str) -> tuple[str, ...]:
    """Return registered statistic names for one dataset."""

    try:
        return AVAILABLE_SUMMARY_NAMES[dataset]
    except KeyError as exc:
        choices = ", ".join(sorted(AVAILABLE_SUMMARY_NAMES))
        raise ValueError(
            f"unknown dataset {dataset!r}; available datasets: {choices}"
        ) from exc


def _configuration_error(
    detail: str,
    *,
    dataset: str,
    path: Path,
    available: tuple[str, ...],
) -> SummaryConfigurationError:
    example = "\n        ".join(available)
    return SummaryConfigurationError(
        f"{detail}\n"
        f"Ask Cursor to choose summary statistics for dataset {dataset!r} "
        f"and update {path}.\n"
        f"Available statistics: {', '.join(available)}\n"
        f"Expected format:\n"
        f"[{dataset}]\n"
        f"enabled =\n"
        f"        {example}"
    )


def load_summary_names(
    dataset: str,
    path: Path = DEFAULT_SUMMARY_CONFIG_PATH,
) -> tuple[str, ...]:
    """Load and validate an ordered summary-statistic selection."""

    available = available_summary_names(dataset)
    path = Path(path)
    if not path.is_file():
        raise _configuration_error(
            "Summary statistics have not been configured.",
            dataset=dataset,
            path=path,
            available=available,
        )

    parser = configparser.ConfigParser(interpolation=None)
    try:
        with path.open(encoding="utf-8") as stream:
            parser.read_file(stream)
    except (OSError, configparser.Error) as exc:
        raise _configuration_error(
            f"Could not read the summary configuration: {exc}",
            dataset=dataset,
            path=path,
            available=available,
        ) from exc

    if not parser.has_section(dataset):
        raise _configuration_error(
            f"The summary configuration has no [{dataset}] section.",
            dataset=dataset,
            path=path,
            available=available,
        )
    if not parser.has_option(dataset, "enabled"):
        raise _configuration_error(
            f"The [{dataset}] section has no enabled option.",
            dataset=dataset,
            path=path,
            available=available,
        )

    raw_names = parser.get(dataset, "enabled", raw=True)
    names = tuple(name for name in re.split(r"[\s,]+", raw_names) if name)
    try:
        return validate_summary_names(
            names,
            available,
            label=f"{dataset} summary",
        )
    except ValueError as exc:
        raise _configuration_error(
            str(exc),
            dataset=dataset,
            path=path,
            available=available,
        ) from exc


__all__ = [
    "AVAILABLE_SUMMARY_NAMES",
    "DEFAULT_SUMMARY_CONFIG_PATH",
    "SummaryConfigurationError",
    "available_summary_names",
    "load_summary_names",
]
