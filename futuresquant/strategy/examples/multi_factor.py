"""
多因子信号驱动策略。

将 MultiFactorSignal 接入回测引擎：
  signal >  entry_threshold  → 做多
  signal < -entry_threshold  → 做空
  |signal| <  exit_threshold → 平仓（信号衰减）
"""

from __future__ import annotations

import pandas as pd

from futuresquant.factors.signal import MultiFactorSignal
from futuresquant.strategy.base import Context, StrategyBase


class MultiFactorStrategy(StrategyBase):

    def __init__(
        self,
        signal_gen: MultiFactorSignal,
        entry_threshold: float = 1.0,
        exit_threshold: float = 0.3,
    ):
        """
        Parameters
        ----------
        signal_gen        : 已配置好的 MultiFactorSignal 实例
        entry_threshold   : 信号绝对值超过此值才开仓
        exit_threshold    : 信号绝对值低于此值时平仓
        """
        self.signal_gen = signal_gen
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.warmup_bars = signal_gen.ic_window + signal_gen.norm_window

        self._signal: pd.Series | None = None

    def on_start(self, klines: pd.DataFrame) -> None:
        self._signal = self.signal_gen.compute(klines)

    def on_bar(self, ctx: Context) -> None:
        ts = ctx.bar.name
        sig = self._signal.loc[ts]

        if pd.isna(sig):
            return

        pos = ctx.position

        # --- 平仓：信号衰减 ---
        if pos > 0 and sig < self.exit_threshold:
            ctx.sell_close()
            return
        if pos < 0 and sig > -self.exit_threshold:
            ctx.buy_close()
            return

        # --- 开仓：信号触发 ---
        if sig > self.entry_threshold and pos <= 0:
            if pos < 0:
                ctx.buy_close()
            ctx.buy_open(1)

        elif sig < -self.entry_threshold and pos >= 0:
            if pos > 0:
                ctx.sell_close()
            ctx.sell_open(1)
