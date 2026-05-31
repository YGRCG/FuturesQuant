"""
双均线趋势跟踪示例策略。

规则：
  - 快线上穿慢线 → 买入开仓（如有空仓先平空）
  - 快线下穿慢线 → 卖出开仓（如有多仓先平多）
  - 每次只持 1 手
"""

from __future__ import annotations

import pandas as pd

from futuresquant.strategy.base import Context, StrategyBase


class MACrossStrategy(StrategyBase):
    warmup_bars: int = 0  # set dynamically in on_start

    def __init__(self, fast: int = 5, slow: int = 20):
        self.fast = fast
        self.slow = slow
        self._fast_ma: pd.Series | None = None
        self._slow_ma: pd.Series | None = None

    def on_start(self, klines: pd.DataFrame) -> None:
        c = klines["close"]
        self._fast_ma = c.rolling(self.fast).mean()
        self._slow_ma = c.rolling(self.slow).mean()
        self.warmup_bars = self.slow  # skip first `slow` bars

    def on_bar(self, ctx: Context) -> None:
        i = ctx.bar.name  # current timestamp (index label)
        fast = self._fast_ma.loc[i]
        slow = self._slow_ma.loc[i]

        if pd.isna(fast) or pd.isna(slow):
            return

        pos = ctx.position

        if fast > slow:
            if pos <= 0:
                if pos < 0:
                    ctx.buy_close()   # 先平空
                ctx.buy_open(1)       # 再开多
        else:
            if pos >= 0:
                if pos > 0:
                    ctx.sell_close()  # 先平多
                ctx.sell_open(1)      # 再开空
