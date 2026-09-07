"""Dataset-independent summary-statistic contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

SummaryFunction = Callable[[Any], ArrayLike]
Summaries = Mapping[str, SummaryFunction]


def validate_summary_names(
    names: Sequence[str],
    available: Sequence[str] | Mapping[str, Any],
    *,
    label: str = "summary",
) -> tuple[str, ...]:
    """Return a validated, ordered, non-empty registry subset."""

    selected = tuple(names)
    if not selected:
        raise ValueError(f"at least one {label} statistic must be selected")
    if any(not isinstance(name, str) or not name for name in selected):
        raise ValueError(f"{label} statistic names must be non-empty strings")
    duplicates = sorted(
        name for name in set(selected) if selected.count(name) > 1
    )
    if duplicates:
        raise ValueError(
            f"duplicate {label} statistics: {', '.join(duplicates)}"
        )
    available_names = tuple(available)
    unknown = [name for name in selected if name not in available_names]
    if unknown:
        raise ValueError(
            f"unknown {label} statistics: {', '.join(unknown)}"
        )
    return selected


def select_summaries(
    available: Mapping[str, SummaryFunction],
    names: Sequence[str] | None = None,
    *,
    label: str = "summary",
) -> dict[str, SummaryFunction]:
    """Select named summary functions while preserving requested order."""

    if names is None:
        return dict(available)
    selected = validate_summary_names(names, available, label=label)
    return {name: available[name] for name in selected}


def compute_scalar_summaries(
    data: Any,
    summaries: Mapping[str, Callable[[Any], ArrayLike]],
    *,
    label: str = "summary",
) -> dict[str, NDArray[Any]]:
    """Apply summary functions and enforce finite scalar outputs."""

    result = {
        name: np.atleast_1d(np.asarray(function(data), dtype=np.float32))
        for name, function in summaries.items()
    }
    if any(value.size != 1 for value in result.values()):
        raise ValueError(f"{label} statistics must be scalar")
    if any(not np.all(np.isfinite(value)) for value in result.values()):
        raise ValueError(f"{label} statistics must be finite")
    return result


__all__ = [
    "Summaries",
    "SummaryFunction",
    "compute_scalar_summaries",
    "select_summaries",
    "validate_summary_names",
]
