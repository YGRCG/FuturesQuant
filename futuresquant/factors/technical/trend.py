"""Trend-following factors."""

from __future__ import annotations

import pandas as pd

from futuresquant.factors.base import Factor


class MACross(Factor):
    """
    MA crossover signal: fast_MA / slow_MA - 1.
    Positive → fast above slow (bullish), negative → bearish.
    """

    def __init__(self, fast: int = 5, slow: int = 20, ma_type: str = "ema"):
        self.fast = fast
        self.slow = slow
        self.ma_type = ma_type
        self.name = f"MACross_{ma_type}_{fast}_{slow}"

    def compute(self, klines: pd.DataFrame) -> pd.Series:
        c = klines["close"]
        if self.ma_type == "ema":
            fast_ma = c.ewm(span=self.fast, adjust=False, min_periods=self.fast).mean()
            slow_ma = c.ewm(span=self.slow, adjust=False, min_periods=self.slow).mean()
        else:
            fast_ma = c.rolling(self.fast).mean()
            slow_ma = c.rolling(self.slow).mean()
        return (fast_ma / slow_ma - 1).rename(self.name)


class MACD(Factor):
    """
    MACD histogram: (fast_ema - slow_ema) - signal_ema.
    Normalised by price to make it comparable across contracts.
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.name = f"MACD_{fast}_{slow}_{signal}"

    def compute(self, klines: pd.DataFrame) -> pd.Series:
        c = klines["close"]
        fast_ema = c.ewm(span=self.fast, adjust=False, min_periods=self.fast).mean()
        slow_ema = c.ewm(span=self.slow, adjust=False, min_periods=self.slow).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=self.signal, adjust=False, min_periods=self.signal).mean()
        histogram = (macd_line - signal_line) / c
        return histogram.rename(self.name)


class ADX(Factor):
    """
    Average Directional Index — measures trend strength (0–100).
    High ADX (>25) = strong trend regardless of direction.
    """

    def __init__(self, period: int = 14):
        self.period = period
        self.name = f"ADX_{period}"

    def compute(self, klines: pd.DataFrame) -> pd.Series:
        high, low, close = klines["high"], klines["low"], klines["close"]
        prev_close = close.shift(1)

        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)

        dm_plus = (high - high.shift(1)).clip(lower=0)
        dm_minus = (low.shift(1) - low).clip(lower=0)
        # Zero out where the opposite move is larger
        mask = dm_plus >= dm_minus
        dm_plus = dm_plus.where(mask, 0)
        dm_minus = dm_minus.where(~mask, 0)

        atr = tr.ewm(com=self.period - 1, min_periods=self.period).mean()
        di_plus = 100 * dm_plus.ewm(com=self.period - 1, min_periods=self.period).mean() / atr
        di_minus = 100 * dm_minus.ewm(com=self.period - 1, min_periods=self.period).mean() / atr

        dx = (100 * (di_plus - di_minus).abs() / (di_plus + di_minus)).replace(
            [float("inf"), float("-inf")], float("nan")
        )
        return dx.ewm(com=self.period - 1, min_periods=self.period).mean().rename(self.name)


class PriceChannel(Factor):
    """
    Donchian channel position: (close - N-bar low) / (N-bar high - N-bar low).
    0 = at channel bottom, 1 = at channel top.
    """

    def __init__(self, period: int = 20):
        self.period = period
        self.name = f"Channel_{period}"

    def compute(self, klines: pd.DataFrame) -> pd.Series:
        c = klines["close"]
        ch_high = c.rolling(self.period).max()
        ch_low = c.rolling(self.period).min()
        return ((c - ch_low) / (ch_high - ch_low)).rename(self.name)
