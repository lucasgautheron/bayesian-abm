from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse import csr_matrix


class FakePrior:
    current: "FakePrior | None" = None

    def __init__(self) -> None:
        self.variables: list[tuple[str, str, dict[str, object]]] = []

    def __enter__(self) -> "FakePrior":
        FakePrior.current = self
        return self

    def __exit__(self, *_: object) -> None:
        FakePrior.current = None


class FakePyMC:
    Model = FakePrior

    @staticmethod
    def _add(distribution: str, name: str, **kwargs: object) -> object:
        assert FakePrior.current is not None
        FakePrior.current.variables.append((distribution, name, kwargs))
        return object()

    @classmethod
    def Normal(cls, name: str, **kwargs: object) -> object:
        return cls._add("Normal", name, **kwargs)

    @classmethod
    def HalfNormal(cls, name: str, **kwargs: object) -> object:
        return cls._add("HalfNormal", name, **kwargs)

    @classmethod
    def Exponential(cls, name: str, **kwargs: object) -> object:
        return cls._add("Exponential", name, **kwargs)

    @classmethod
    def Beta(cls, name: str, **kwargs: object) -> object:
        return cls._add("Beta", name, **kwargs)


def scientist_context() -> dict[str, Any]:
    n_scientists = 4
    coauthorship = csr_matrix(
        (
            np.ones(6),
            (
                np.asarray([0, 1, 1, 2, 2, 3]),
                np.asarray([1, 0, 2, 1, 3, 2]),
            ),
        ),
        shape=(n_scientists, n_scientists),
    )
    citations = csr_matrix(
        (
            np.ones(4),
            (
                np.asarray([0, 1, 2, 3]),
                np.asarray([1, 2, 3, 0]),
            ),
        ),
        shape=(n_scientists, n_scientists),
    )
    return {
        "n_scientists": n_scientists,
        "primary_area": np.arange(4, dtype=np.int8),
        "area_shares": np.eye(4, dtype=np.float64),
        "career_start_year": np.arange(2000, 2004, dtype=np.int16),
        "observed_mask": np.ones(4, dtype=np.bool_),
        "coauthorship": coauthorship,
        "citations": citations,
        "first_coauthor": np.asarray([-1, 0, 1, 2], dtype=np.int32),
        "start_year": 2000,
        "end_year": 2003,
    }
