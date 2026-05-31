"""Volatility factors."""

from __future__ import annotations

import numpy as np
import pandas as pd

from futuresquant.factors.base import Factor


class ATR(Factor):
    """
    Average True Range — raw volatility in price units.
    Useful as a position-sizing input.
    """

    def __init__(self, period: int = 14):
        self.period = period
        self.name = f"ATR_{period}"

    def compute(self, klines: pd.DataFrame) -> pd.Series:
        high, low, close = klines["high"], klines["low"], klines["close"]
        tr = pd.concat(
            [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
            axis=1,
        ).max(axis=1)
        return tr.ewm(com=self.period - 1, min_periods=self.period).mean().rename(self.name)


class NormATR(Factor):
    """ATR normalised by close price — comparable across contracts."""

    def __init__(self, period: int = 14):
        self.period = period
        self.name = f"NATR_{period}"
        self._atr = ATR(period)

    def compute(self, klines: pd.DataFrame) -> pd.Series:
        return (self._atr.compute(klines) / klines["close"]).rename(self.name)


class HistoricalVolatility(Factor):
    """
    Annualised historical volatility of log returns.
    window : rolling window in bars
    ann_factor : bars per year (default 250*240 for 1-min)
    """

    def __init__(self, window: int = 240, ann_factor: int = 250 * 240):
        self.window = window
        self.ann_factor = ann_factor
        self.name = f"HV_{window}"

    def compute(self, klines: pd.DataFrame) -> pd.Series:
        log_ret = np.log(klines["close"] / klines["close"].shift(1))
        return (
            log_ret.rolling(self.window).std(ddof=1) * np.sqrt(self.ann_factor)
        ).rename(self.name)


class VolatilityRatio(Factor):
    """
    Short-term / long-term volatility ratio.
    > 1 → volatility expanding (breakout environment)
    < 1 → volatility contracting (mean-reversion environment)
    """

    def __init__(self, fast: int = 20, slow: int = 120):
        self.fast = fast
        self.slow = slow
        self.name = f"VolRatio_{fast}_{slow}"

    def compute(self, klines: pd.DataFrame) -> pd.Series:
        log_ret = np.log(klines["close"] / klines["close"].shift(1))
        hv_fast = log_ret.rolling(self.fast).std(ddof=1)
        hv_slow = log_ret.rolling(self.slow).std(ddof=1)
        return (hv_fast / hv_slow).rename(self.name)
