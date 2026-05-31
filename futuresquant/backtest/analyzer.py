"""Performance analytics for backtest results."""

from __future__ import annotations

import numpy as np
import pandas as pd


class PerformanceAnalyzer:
    """Compute standard quant performance metrics from an equity curve."""

    # 1-min bars: 240 bars/day × 250 trading days
    ANN_FACTOR = 240 * 250

    def __init__(self, equity: pd.Series, initial_capital: float):
        self.equity = equity.dropna()
        self.initial_capital = initial_capital

    def compute(self) -> dict:
        eq = self.equity
        if len(eq) < 2:
            return {}

        returns = eq.pct_change().dropna()

        total_return = (eq.iloc[-1] / self.initial_capital) - 1
        ann_return = (1 + total_return) ** (self.ANN_FACTOR / len(eq)) - 1

        ann_vol = returns.std(ddof=1) * np.sqrt(self.ANN_FACTOR)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

        downside = returns[returns < 0].std(ddof=1) * np.sqrt(self.ANN_FACTOR)
        sortino = ann_return / downside if downside > 0 else 0.0

        drawdown = self._max_drawdown(eq)
        calmar = ann_return / abs(drawdown) if drawdown != 0 else 0.0

        return {
            "total_return":    round(total_return, 6),
            "annual_return":   round(ann_return, 6),
            "annual_vol":      round(ann_vol, 6),
            "sharpe":          round(sharpe, 4),
            "sortino":         round(sortino, 4),
            "max_drawdown":    round(drawdown, 6),
            "calmar":          round(calmar, 4),
            "start":           eq.index[0],
            "end":             eq.index[-1],
        }

    @staticmethod
    def _max_drawdown(equity: pd.Series) -> float:
        peak = equity.cummax()
        dd = (equity - peak) / peak
        return float(dd.min())

    def drawdown_series(self) -> pd.Series:
        eq = self.equity
        peak = eq.cummax()
        return ((eq - peak) / peak).rename("drawdown")

    def rolling_sharpe(self, window_bars: int = 240 * 20) -> pd.Series:
        """20-day rolling Sharpe (1-min bars)."""
        r = self.equity.pct_change()
        mu = r.rolling(window_bars).mean() * self.ANN_FACTOR
        sigma = r.rolling(window_bars).std(ddof=1) * np.sqrt(self.ANN_FACTOR)
        return (mu / sigma).rename("rolling_sharpe")
