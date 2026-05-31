"""
RSI 均值回归策略。

逻辑
----
- RSI < oversold  → 买入开仓（预期价格反弹）
- RSI > overbought → 卖出开仓（预期价格回落）
- 平仓：RSI 回到中性区间 [exit_low, exit_high]
- 趋势过滤（可选）：只在 slow_ma 方向上开仓
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from futuresquant.strategy.base import Context, StrategyBase


class RSIMeanReversionStrategy(StrategyBase):

    def __init__(
        self,
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        exit_low: float = 45.0,
        exit_high: float = 55.0,
        trend_filter_period: int | None = 60,
    ):
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.exit_low = exit_low
        self.exit_high = exit_high
        self.trend_filter_period = trend_filter_period
        self.warmup_bars = max(rsi_period, trend_filter_period or 0) + 1

        self._rsi: pd.Series | None = None
        self._trend_ma: pd.Series | None = None

    def on_start(self, klines: pd.DataFrame) -> None:
        delta = klines["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(com=self.rsi_period - 1, min_periods=self.rsi_period).mean()
        avg_loss = loss.ewm(com=self.rsi_period - 1, min_periods=self.rsi_period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        self._rsi = 100 - 100 / (1 + rs)

        if self.trend_filter_period:
            self._trend_ma = klines["close"].rolling(self.trend_filter_period).mean()

    def on_bar(self, ctx: Context) -> None:
        ts = ctx.bar.name
        rsi = self._rsi.loc[ts]
        close = float(ctx.bar["close"])

        if pd.isna(rsi):
            return

        pos = ctx.position
        trend_up = True
        trend_dn = True

        if self._trend_ma is not None:
            ma = self._trend_ma.loc[ts]
            if not pd.isna(ma):
                trend_up = close > ma
                trend_dn = close < ma

        # --- 平仓 ---
        if pos > 0 and self.exit_low <= rsi <= self.exit_high:
            ctx.sell_close()
            return
        if pos < 0 and self.exit_low <= rsi <= self.exit_high:
            ctx.buy_close()
            return

        # --- 开仓 ---
        if rsi < self.oversold and trend_up and pos <= 0:
            if pos < 0:
                ctx.buy_close()
            ctx.buy_open(1)

        elif rsi > self.overbought and trend_dn and pos >= 0:
            if pos > 0:
                ctx.sell_close()
            ctx.sell_open(1)
