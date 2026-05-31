"""Volume-based factors."""

from __future__ import annotations

import pandas as pd

from futuresquant.factors.base import Factor


class VolumeRatio(Factor):
    """
    Volume relative to its moving average.
    > 1 → above-average activity, < 1 → quiet market.
    """

    def __init__(self, period: int = 20):
        self.period = period
        self.name = f"VolRatio_{period}"

    def compute(self, klines: pd.DataFrame) -> pd.Series:
        vol = klines["volume"]
        return (vol / vol.rolling(self.period).mean()).rename(self.name)


class OBV(Factor):
    """
    On-Balance Volume — cumulative volume signed by price direction.
    Normalised as pct_change over `norm_period` bars to make it stationary.
    """

    def __init__(self, norm_period: int = 20):
        self.norm_period = norm_period
        self.name = f"OBV_{norm_period}"

    def compute(self, klines: pd.DataFrame) -> pd.Series:
        direction = klines["close"].diff().apply(
            lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
        )
        obv = (klines["volume"] * direction).cumsum()
        return obv.pct_change(self.norm_period).rename(self.name)


class VWAP(Factor):
    """
    Deviation of close from rolling VWAP, normalised by ATR.
    Positive → price above VWAP (strong), negative → below (weak).
    """

    def __init__(self, period: int = 60, atr_period: int = 14):
        self.period = period
        self.atr_period = atr_period
        self.name = f"VWAPdev_{period}"

    def compute(self, klines: pd.DataFrame) -> pd.Series:
        typical = (klines["high"] + klines["low"] + klines["close"]) / 3
        tp_vol = typical * klines["volume"]
        vwap = tp_vol.rolling(self.period).sum() / klines["volume"].rolling(self.period).sum()

        # Normalise deviation by ATR
        close = klines["close"]
        tr = pd.concat(
            [klines["high"] - klines["low"],
             (klines["high"] - close.shift(1)).abs(),
             (klines["low"] - close.shift(1)).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(com=self.atr_period - 1, min_periods=self.atr_period).mean()

        return ((close - vwap) / atr).rename(self.name)


class OpenInterestChange(Factor):
    """
    Rate of change in open interest — signals new money entering or leaving.
    Positive + price up → strong trend confirmation.
    """

    def __init__(self, period: int = 20):
        self.period = period
        self.name = f"OIChange_{period}"

    def compute(self, klines: pd.DataFrame) -> pd.Series:
        oi = klines["open_interest"]
        return oi.pct_change(self.period).rename(self.name)
