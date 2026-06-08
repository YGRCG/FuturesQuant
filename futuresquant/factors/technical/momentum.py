"""Momentum / mean-reversion factors."""

from __future__ import annotations

import numpy as np
import pandas as pd

from futuresquant.factors.base import Factor


class ROC(Factor):
    """Rate of Change: (close_t / close_{t-n}) - 1."""

    def __init__(self, period: int = 20):
        self.period = period
        self.name = f"ROC_{period}"

    def compute(self, klines: pd.DataFrame) -> pd.Series:
        c = klines["close"]
        return (c / c.shift(self.period) - 1).rename(self.name)


class MOM(Factor):
    """Absolute momentum: close_t - close_{t-n}."""

    def __init__(self, period: int = 20):
        self.period = period
        self.name = f"MOM_{period}"

    def compute(self, klines: pd.DataFrame) -> pd.Series:
        return klines["close"].diff(self.period).rename(self.name)


class RSI(Factor):
    """Relative Strength Index (Wilder smoothing)."""

    def __init__(self, period: int = 14):
        self.period = period
        self.name = f"RSI_{period}"

    def compute(self, klines: pd.DataFrame) -> pd.Series:
        delta = klines["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(com=self.period - 1, min_periods=self.period).mean()
        avg_loss = loss.ewm(com=self.period - 1, min_periods=self.period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return (100 - 100 / (1 + rs)).rename(self.name)


class BollingerBand(Factor):
    """
    Bollinger %B: position of close within the band.
    0 = lower band, 0.5 = middle, 1 = upper band.
    """

    def __init__(self, period: int = 20, n_std: float = 2.0):
        self.period = period
        self.n_std = n_std
        self.name = f"BB_pct_{period}_{n_std}"

    def compute(self, klines: pd.DataFrame) -> pd.Series:
        c = klines["close"]
        mid = c.rolling(self.period).mean()
        std = c.rolling(self.period).std(ddof=1)
        upper = mid + self.n_std * std
        lower = mid - self.n_std * std
        return ((c - lower) / (upper - lower)).rename(self.name)


class TSMomentum(Factor):
    """
    Time-series momentum: sign of the average return over [slow, fast] window.
    Positive → uptrend, negative → downtrend.
    Computes as: mean(daily_return, slow) - mean(daily_return, fast).
    """

    def __init__(self, fast: int = 5, slow: int = 20):
        self.fast = fast
        self.slow = slow
        self.name = f"TSMom_{fast}_{slow}"

    def compute(self, klines: pd.DataFrame) -> pd.Series:
        ret = klines["close"].pct_change()
        return (ret.rolling(self.slow).mean() - ret.rolling(self.fast).mean()).rename(
            self.name
        )


class MA(Factor):
    """Price deviation from simple moving average: close / SMA(period) - 1."""

    def __init__(self, period: int = 20):
        self.period = period
        self.name = f"MA_{period}"

    def compute(self, klines: pd.DataFrame) -> pd.Series:
        c = klines["close"]
        return (c / c.rolling(self.period).mean() - 1).rename(self.name)
