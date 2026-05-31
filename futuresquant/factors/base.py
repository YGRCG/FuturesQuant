"""
Factor base class.

All factors share the same interface:
    factor.compute(klines) -> pd.Series  (index = DatetimeIndex, name = factor.name)

Factors are stateless: compute() must be a pure function of the input DataFrame.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class Factor(ABC):
    """
    Abstract base for all alpha factors.

    Subclasses must implement compute() and set a class-level `name` attribute,
    or accept params in __init__ and build the name dynamically.
    """

    #: Override in subclass or set in __init__
    name: str = ""

    @abstractmethod
    def compute(self, klines: pd.DataFrame) -> pd.Series:
        """
        Compute factor values from a 1-min K-line DataFrame.

        Parameters
        ----------
        klines : DataFrame with columns open/high/low/close/volume/amount/open_interest
                 and a DatetimeIndex.

        Returns
        -------
        pd.Series with the same DatetimeIndex, name == self.name.
        Leading NaNs are allowed (warmup period).
        """
        ...

    # ------------------------------------------------------------------
    # Operator overloads — compose factors arithmetically
    # ------------------------------------------------------------------

    def __add__(self, other: Factor | float) -> _ComposedFactor:
        return _ComposedFactor(self, other, "+")

    def __sub__(self, other: Factor | float) -> _ComposedFactor:
        return _ComposedFactor(self, other, "-")

    def __mul__(self, other: Factor | float) -> _ComposedFactor:
        return _ComposedFactor(self, other, "*")

    def __truediv__(self, other: Factor | float) -> _ComposedFactor:
        return _ComposedFactor(self, other, "/")

    def __neg__(self) -> _ComposedFactor:
        return _ComposedFactor(self, -1.0, "*")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class _ComposedFactor(Factor):
    """Factor created by arithmetic composition of two factors."""

    def __init__(self, left: Factor | float, right: Factor | float, op: str):
        self._left = left
        self._right = right
        self._op = op
        l_name = left.name if isinstance(left, Factor) else str(left)
        r_name = right.name if isinstance(right, Factor) else str(right)
        self.name = f"({l_name}{op}{r_name})"

    def compute(self, klines: pd.DataFrame) -> pd.Series:
        lv = self._left.compute(klines) if isinstance(self._left, Factor) else self._left
        rv = self._right.compute(klines) if isinstance(self._right, Factor) else self._right
        ops = {"+": lv.__add__, "-": lv.__sub__, "*": lv.__mul__, "/": lv.__truediv__}
        result = ops[self._op](rv)
        if isinstance(result, pd.Series):
            result.name = self.name
        return result


# ---------------------------------------------------------------------------
# Normalization helpers (applied to a single factor's time series)
# ---------------------------------------------------------------------------

def zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score normalisation."""
    mu = series.rolling(window).mean()
    sigma = series.rolling(window).std(ddof=1)
    return ((series - mu) / sigma).rename(f"zscore({series.name},{window})")


def rank_normalize(series: pd.Series, window: int) -> pd.Series:
    """Rolling percentile rank in [0, 1]."""
    def _rank(x: pd.Series) -> float:
        return x.rank(pct=True).iloc[-1]
    return series.rolling(window).apply(_rank, raw=False).rename(
        f"rank({series.name},{window})"
    )
