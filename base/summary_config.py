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
RECOMMENDED_SUMMARY_NAMES: Final[dict[str, tuple[str, ...]]] = {
    "contacts": (
        "cumulative_network_connectivity",
        "cumulative_network_clustering",
        "cumulative_network_degree_variance",
    ),
    STORY_DATASET: (
        "total_mentions",
        "mention_concentration",
        "mean_story_autocorrelation",
    ),
    SCIENTIST_CONVENTIONS_DATASET: (
        "coauthorship_coupling",
        "citation_coupling",
        "field_theory_hep",
    ),
}


class SummaryConfigurationError(ValueError):
    """An actionable error in the attendee's local summary selection."""


def format_summary_configuration_error(
    error: SummaryConfigurationError,
    *,
    color: bool,
) -> str:
    """Format the first error line in red for an interactive terminal."""

    lines = str(error).splitlines()
    if color and lines:
        lines[0] = f"\033[31m{lines[0]}\033[0m"
    return "\n".join(lines)


def available_summary_names(dataset: str) -> tuple[str, ...]:
    """Return registered statistic names for one dataset."""

    try:
        return AVAILABLE_SUMMARY_NAMES[dataset]
    except KeyError as exc:
        choices = ", ".join(sorted(AVAILABLE_SUMMARY_NAMES))
        raise ValueError(
            f"unknown dataset {dataset!r}; available datasets: {choices}"
        ) from exc


def recommended_summary_names(dataset: str) -> tuple[str, ...]:
    """Return the validated recommended starting set for one dataset."""

    available = available_summary_names(dataset)
    try:
        recommended = RECOMMENDED_SUMMARY_NAMES[dataset]
    except KeyError as exc:
        raise ValueError(
            f"no recommended summary statistics for dataset {dataset!r}"
        ) from exc
    return validate_summary_names(
        recommended,
        available,
        label=f"recommended {dataset} summary",
    )


def _configuration_error(
    detail: str,
    *,
    dataset: str,
    path: Path,
    available: tuple[str, ...],
) -> SummaryConfigurationError:
    recommended = recommended_summary_names(dataset)
    return SummaryConfigurationError(
        f"{detail}\n"
        f"Run /configure-summary-stats in Cursor to choose statistics for "
        f"dataset {dataset!r} and update {path}.\n"
        f"Recommended starting set: {', '.join(recommended)}\n"
        f"Available statistics: {', '.join(available)}\n"
        f"Expected format:\n"
        f"[{dataset}]\n"
        f"enabled =\n"
        f"        <explicitly chosen registered statistic>"
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
    "RECOMMENDED_SUMMARY_NAMES",
    "SummaryConfigurationError",
    "available_summary_names",
    "format_summary_configuration_error",
    "load_summary_names",
    "recommended_summary_names",
]
