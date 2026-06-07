"""Performance analytics for backtest results."""

from __future__ import annotations

import numpy as np
import pandas as pd


class PerformanceAnalyzer:
    """Compute standard quant performance metrics from an equity curve.

    Parameters
    ----------
    bars_per_year : int, optional
        Annual bar count used for annualisation.  If None (default) it is
        inferred automatically from the equity-curve timestamps so the same
        class works correctly for 1-min, 5-min, daily, … data.
    """

    def __init__(
        self,
        equity: pd.Series,
        initial_capital: float,
        bars_per_year: int | None = None,
    ):
        self.equity = equity.dropna()
        self.initial_capital = initial_capital
        self._ann_factor = (
            bars_per_year
            if bars_per_year is not None
            else self._infer_ann_factor(self.equity)
        )

    # ------------------------------------------------------------------
    # Auto-detection
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_ann_factor(equity: pd.Series) -> int:
        """Estimate annual bar count from the equity-curve timestamps.

        Uses the actual calendar span of the data so it works for any bar
        frequency (1-min, 5-min, daily …) and correctly handles products
        with night sessions (e.g. FU which trades ~6.5 h/day).
        """
        if len(equity) < 10:
            return 240 * 250  # safe fallback
        period_days = (
            equity.index[-1] - equity.index[0]
        ).total_seconds() / 86_400
        if period_days <= 0:
            return 240 * 250
        period_years = period_days / 365.25
        return max(252, round(len(equity) / period_years))

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def compute(self) -> dict:
        eq = self.equity
        if len(eq) < 2:
            return {}

        returns = eq.pct_change().dropna()
        ann = self._ann_factor

        total_return = (eq.iloc[-1] / self.initial_capital) - 1
        ann_return = (1 + total_return) ** (ann / len(eq)) - 1

        ann_vol = returns.std(ddof=1) * np.sqrt(ann)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

        downside = returns[returns < 0].std(ddof=1) * np.sqrt(ann)
        sortino = ann_return / downside if downside > 0 else 0.0

        drawdown = self._max_drawdown(eq)
        calmar = ann_return / abs(drawdown) if drawdown != 0 else 0.0

        return {
            "total_return":  round(total_return, 6),
            "annual_return": round(ann_return, 6),
            "annual_vol":    round(ann_vol, 6),
            "sharpe":        round(sharpe, 4),
            "sortino":       round(sortino, 4),
            "max_drawdown":  round(drawdown, 6),
            "calmar":        round(calmar, 4),
            "start":         eq.index[0],
            "end":           eq.index[-1],
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

    def rolling_sharpe(self, window_bars: int) -> pd.Series:
        r = self.equity.pct_change()
        ann = self._ann_factor
        mu = r.rolling(window_bars).mean() * ann
        sigma = r.rolling(window_bars).std(ddof=1) * np.sqrt(ann)
        return (mu / sigma).rename("rolling_sharpe")
