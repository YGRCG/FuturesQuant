"""
布林带策略（均值回归 + 可选突破模式）。

均值回归模式（默认）
--------------------
- 价格触及下轨 → 买入（预期回归中轨）
- 价格触及上轨 → 卖出（预期回归中轨）
- 平仓：价格穿越中轨

突破模式（mode='breakout'）
---------------------------
- 价格从内部突破上轨 → 做多
- 价格从内部突破下轨 → 做空
- 平仓：价格回到中轨
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from futuresquant.strategy.base import Context, StrategyBase


class BollingerBandStrategy(StrategyBase):

    def __init__(
        self,
        period: int = 20,
        n_std: float = 2.0,
        mode: Literal["reversion", "breakout"] = "reversion",
        squeeze_filter: bool = True,
    ):
        """
        Parameters
        ----------
        squeeze_filter : if True, skip signals when band width is below
                         its 20-bar median (avoid flat/tight markets)
        """
        self.period = period
        self.n_std = n_std
        self.mode = mode
        self.squeeze_filter = squeeze_filter
        self.warmup_bars = period * 2

        self._mid: pd.Series | None = None
        self._upper: pd.Series | None = None
        self._lower: pd.Series | None = None
        self._width: pd.Series | None = None
        self._pct_b: pd.Series | None = None   # position within band [0,1]

    def on_start(self, klines: pd.DataFrame) -> None:
        c = klines["close"]
        self._mid = c.rolling(self.period).mean()
        std = c.rolling(self.period).std(ddof=1)
        self._upper = self._mid + self.n_std * std
        self._lower = self._mid - self.n_std * std
        self._width = (self._upper - self._lower) / self._mid
        band_range = self._upper - self._lower
        self._pct_b = (c - self._lower) / band_range.replace(0, float("nan"))

    def on_bar(self, ctx: Context) -> None:
        ts = ctx.bar.name
        mid = self._mid.loc[ts]
        upper = self._upper.loc[ts]
        lower = self._lower.loc[ts]
        pct_b = self._pct_b.loc[ts]
        width = self._width.loc[ts]
        close = float(ctx.bar["close"])

        if pd.isna(mid) or pd.isna(pct_b):
            return

        # 布林带收窄过滤（squeeze）
        if self.squeeze_filter:
            width_med = self._width.loc[:ts].iloc[-self.period:].median()
            if not pd.isna(width_med) and width < width_med:
                return

        pos = ctx.position

        if self.mode == "reversion":
            # 平仓：穿越中轨
            if pos > 0 and close >= mid:
                ctx.sell_close()
                return
            if pos < 0 and close <= mid:
                ctx.buy_close()
                return

            # 开仓：触及轨道
            if pct_b <= 0.0 and pos <= 0:
                if pos < 0:
                    ctx.buy_close()
                ctx.buy_open(1)
            elif pct_b >= 1.0 and pos >= 0:
                if pos > 0:
                    ctx.sell_close()
                ctx.sell_open(1)

        else:  # breakout
            # 平仓：回归中轨
            if pos > 0 and close <= mid:
                ctx.sell_close()
                return
            if pos < 0 and close >= mid:
                ctx.buy_close()
                return

            # 开仓：突破轨道
            if close > upper and pos <= 0:
                if pos < 0:
                    ctx.buy_close()
                ctx.buy_open(1)
            elif close < lower and pos >= 0:
                if pos > 0:
                    ctx.sell_close()
                ctx.sell_open(1)
