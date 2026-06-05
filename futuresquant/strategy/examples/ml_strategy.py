"""
ML 信号驱动策略。

接收预计算的 ML 连续预测信号，使用滚动分位数作为开平仓阈值（避免未来函数）：
  信号 >= 滚动 entry_quantile → 做多
  信号 <= 滚动 (1-entry_quantile) → 做空
  |信号排名| 回到 exit_quantile 以内 → 平仓
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from futuresquant.strategy.base import Context, StrategyBase


class MLStrategy(StrategyBase):

    def __init__(
        self,
        signal: pd.Series,
        entry_quantile: float = 0.8,
        exit_quantile: float = 0.5,
        rolling_window: int = 500,
    ):
        """
        Parameters
        ----------
        signal          : 预计算的 ML 连续信号（index=DatetimeIndex，值越大越看多）
        entry_quantile  : 滚动分位数阈值，信号 >= 此分位开多，<= (1-此值) 开空
        exit_quantile   : 信号回到此分位以内时平仓
        rolling_window  : 滚动窗口（bar 数），用于计算分位数
        """
        self.signal = signal
        self.entry_quantile = entry_quantile
        self.exit_quantile = exit_quantile
        self.rolling_window = rolling_window
        self.warmup_bars = rolling_window

        self._rolling_hi: pd.Series | None = None
        self._rolling_lo: pd.Series | None = None
        self._rolling_mid_hi: pd.Series | None = None
        self._rolling_mid_lo: pd.Series | None = None

    def on_start(self, klines: pd.DataFrame) -> None:
        sig = self.signal.dropna()
        roll = sig.rolling(self.rolling_window, min_periods=max(self.rolling_window // 2, 20))
        self._rolling_hi = roll.quantile(self.entry_quantile)
        self._rolling_lo = roll.quantile(1 - self.entry_quantile)
        self._rolling_mid_hi = roll.quantile(self.exit_quantile)
        self._rolling_mid_lo = roll.quantile(1 - self.exit_quantile)

    def on_bar(self, ctx: Context) -> None:
        ts = ctx.timestamp
        if ts not in self.signal.index:
            return

        sig = self.signal.loc[ts]
        if pd.isna(sig):
            return

        hi = self._rolling_hi.get(ts, np.nan)
        lo = self._rolling_lo.get(ts, np.nan)
        mid_hi = self._rolling_mid_hi.get(ts, np.nan)
        mid_lo = self._rolling_mid_lo.get(ts, np.nan)

        if pd.isna(hi) or pd.isna(lo):
            return

        pos = ctx.position

        if pos > 0 and sig < mid_hi:
            ctx.sell_close()
            return
        if pos < 0 and sig > mid_lo:
            ctx.buy_close()
            return

        if sig >= hi and pos <= 0:
            if pos < 0:
                ctx.buy_close()
            ctx.buy_open(1)
        elif sig <= lo and pos >= 0:
            if pos > 0:
                ctx.sell_close()
            ctx.sell_open(1)
