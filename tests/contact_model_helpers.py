from __future__ import annotations

import numpy as np


class FakePrior:
    current: FakePrior | None = None

    def __init__(self) -> None:
        self.variables: list[tuple[str, str, dict]] = []

    def __enter__(self) -> FakePrior:
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
    def LogNormal(cls, name: str, **kwargs: object) -> object:
        return cls._add("LogNormal", name, **kwargs)

    @classmethod
    def HalfNormal(cls, name: str, **kwargs: object) -> object:
        return cls._add("HalfNormal", name, **kwargs)

    @classmethod
    def Beta(cls, name: str, **kwargs: object) -> object:
        return cls._add("Beta", name, **kwargs)

    @classmethod
    def Bernoulli(cls, name: str, **kwargs: object) -> object:
        return cls._add("Bernoulli", name, **kwargs)

    @classmethod
    def Exponential(cls, name: str, **kwargs: object) -> object:
        return cls._add("Exponential", name, **kwargs)

    @classmethod
    def Pareto(cls, name: str, **kwargs: object) -> object:
        return cls._add("Pareto", name, **kwargs)


class ScriptedRng:
    def __init__(
        self,
        *,
        communities: list[int] | None = None,
        activities: list[float] | None = None,
    ) -> None:
        self.communities = communities
        self.activities = activities

    def integers(self, high, size=None, dtype=np.int32):
        del high
        values = self.communities if self.communities is not None else [0] * size
        return np.asarray(values, dtype=dtype)

    def lognormal(self, mean, sigma, size=None):
        del mean, sigma
        values = self.activities if self.activities is not None else [1.0] * size
        return np.asarray(values, dtype=np.float64)

    def normal(self, loc=0.0, scale=1.0, size=None):
        del loc, scale
        if size is None:
            return 0.0
        return np.zeros(size, dtype=np.float64)

    def random(self, size=None):
        if size is None:
            return 0.0
        return np.zeros(size, dtype=np.float64)

    def beta(self, a, b):
        return np.asarray(a, dtype=np.float64) / (
            np.asarray(a, dtype=np.float64) + np.asarray(b, dtype=np.float64)
        )

    def poisson(self, lam, size=None):
        value = 1 if float(np.asarray(lam).reshape(-1)[0]) > 0.0 else 0
        if size is None:
            return value
        return np.full(size, value)
