"""Dataset-independent summary-statistic contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

SummaryFunction = Callable[[Any], ArrayLike]
Summaries = Mapping[str, SummaryFunction]


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


__all__ = ["Summaries", "SummaryFunction", "compute_scalar_summaries"]
