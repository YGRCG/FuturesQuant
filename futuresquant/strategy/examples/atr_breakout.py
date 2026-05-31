"""
ATR 通道突破策略。

逻辑
----
- 上轨 = N 日最高价  （Donchian 通道）
- 下轨 = N 日最低价
- 进场过滤：当前 ATR > ATR 均值（避免震荡市）
- 做多：收盘价突破上轨
- 做空：收盘价跌破下轨
- 止损：进场价 ± exit_atr_mult × ATR
"""

from __future__ import annotations

import pandas as pd

from futuresquant.strategy.base import Context, StrategyBase


class ATRBreakoutStrategy(StrategyBase):

    def __init__(
        self,
        channel_period: int = 20,
        atr_period: int = 14,
        atr_filter_period: int = 60,
        exit_atr_mult: float = 2.0,
    ):
        self.channel_period = channel_period
        self.atr_period = atr_period
        self.atr_filter_period = atr_filter_period
        self.exit_atr_mult = exit_atr_mult
        self.warmup_bars = max(channel_period, atr_filter_period)

        self._upper: pd.Series | None = None
        self._lower: pd.Series | None = None
        self._atr: pd.Series | None = None
        self._atr_ma: pd.Series | None = None
        self._entry_price: float | None = None

    def on_start(self, klines: pd.DataFrame) -> None:
        h, l, c = klines["high"], klines["low"], klines["close"]
        tr = pd.concat(
            [h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1
        ).max(axis=1)

        self._upper = h.rolling(self.channel_period).max()
        self._lower = l.rolling(self.channel_period).min()
        self._atr = tr.ewm(com=self.atr_period - 1, min_periods=self.atr_period).mean()
        self._atr_ma = self._atr.rolling(self.atr_filter_period).mean()

    def on_bar(self, ctx: Context) -> None:
        ts = ctx.bar.name
        close = float(ctx.bar["close"])
        upper = self._upper.loc[ts]
        lower = self._lower.loc[ts]
        atr = self._atr.loc[ts]
        atr_ma = self._atr_ma.loc[ts]

        if pd.isna(upper) or pd.isna(atr_ma):
            return

        pos = ctx.position
        in_trend = atr > atr_ma   # 波动率过滤

        # --- 止损检查（优先于开仓信号）---
        if pos > 0 and self._entry_price is not None:
            if close < self._entry_price - self.exit_atr_mult * atr:
                ctx.sell_close()
                self._entry_price = None
                return
        elif pos < 0 and self._entry_price is not None:
            if close > self._entry_price + self.exit_atr_mult * atr:
                ctx.buy_close()
                self._entry_price = None
                return

        # --- 开仓信号 ---
        if in_trend:
            if close > upper and pos <= 0:
                if pos < 0:
                    ctx.buy_close()
                ctx.buy_open(1)
                self._entry_price = close

            elif close < lower and pos >= 0:
                if pos > 0:
                    ctx.sell_close()
                ctx.sell_open(1)
                self._entry_price = close
