"""
滚动窗口样本外验证（Walk-Forward Analysis, WFA）。

原理
----
将历史数据切成 N 个滚动窗口，每个窗口分为：
  - IS（In-Sample）  ：参数优化期
  - OOS（Out-of-Sample）：样本外验证期

          ┌────── IS ──────┬─ OOS ─┐
Window 1  │████████████████│░░░░░░░│
          └────── IS ──────┴─ OOS ─┘
                  ┌────── IS ──────┬─ OOS ─┐
Window 2          │████████████████│░░░░░░░│
                  └────── IS ──────┴─ OOS ─┘

最终指标
--------
- OOS Sharpe（真实绩效估计）
- IS/OOS 衰减比（越接近 1 越好，< 0 说明严重过拟合）
- 各窗口最优参数的稳定性

防过拟合规则
-----------
- OOS Sharpe / IS Sharpe > 0.5 → 可接受
- 各窗口最优参数一致性高 → 参数稳定，不是噪音拟合
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from futuresquant.backtest.analyzer import PerformanceAnalyzer
from futuresquant.backtest.engine import BacktestConfig, BacktestEngine
from futuresquant.optimize.grid_search import (
    Constraint, GridSearchOptimizer, OptimizeResult, StrategyFactory,
)

logger = logging.getLogger(__name__)


@dataclass
class WindowResult:
    window_id: int
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    best_params: dict[str, Any]
    is_score: float
    oos_score: float
    oos_equity: pd.Series


@dataclass
class WalkForwardResult:
    windows: list[WindowResult]
    metric: str
    oos_equity: pd.Series             # 拼接的 OOS 权益曲线
    oos_metrics: dict                 # 基于拼接 OOS 曲线的绩效
    degradation_ratio: float          # OOS / IS 平均得分比

    def summary(self) -> pd.DataFrame:
        rows = []
        for w in self.windows:
            rows.append({
                "窗口": w.window_id,
                "IS 起始": w.is_start.date(),
                "OOS 起始": w.oos_start.date(),
                "OOS 结束": w.oos_end.date(),
                f"IS {self.metric}": round(w.is_score, 4),
                f"OOS {self.metric}": round(w.oos_score, 4),
                "衰减比": round(w.oos_score / w.is_score, 3) if w.is_score != 0 else float("nan"),
                **{f"param_{k}": v for k, v in w.best_params.items()},
            })
        return pd.DataFrame(rows)

    def param_stability(self) -> pd.DataFrame:
        """各窗口最优参数，用于判断参数稳定性。"""
        rows = [{"窗口": w.window_id, **w.best_params} for w in self.windows]
        return pd.DataFrame(rows)


class WalkForwardOptimizer:
    """
    滚动窗口样本外验证框架。

    Parameters
    ----------
    strategy_factory  : 同 GridSearchOptimizer
    param_grid        : 参数搜索空间
    config            : BacktestConfig
    metric            : 优化目标指标
    constraints       : 参数约束
    n_splits          : 窗口数量
    is_ratio          : IS 占窗口的比例（默认 0.7）
    optimizer         : 'grid'（默认）或 'genetic'
    genetic_kwargs    : 传给 GeneticOptimizer 的额外参数
    """

    def __init__(
        self,
        strategy_factory: StrategyFactory,
        param_grid: dict[str, list],
        config: BacktestConfig,
        metric: str = "sharpe",
        constraints: list[Constraint] | None = None,
        n_splits: int = 5,
        is_ratio: float = 0.7,
        optimizer: str = "grid",
        genetic_kwargs: dict | None = None,
    ):
        self.strategy_factory = strategy_factory
        self.param_grid = param_grid
        self.config = config
        self.metric = metric
        self.constraints = constraints or []
        self.n_splits = n_splits
        self.is_ratio = is_ratio
        self.optimizer_type = optimizer
        self.genetic_kwargs = genetic_kwargs or {}

    def run(self, klines: pd.DataFrame, initial_capital: float | None = None) -> WalkForwardResult:
        capital = initial_capital or self.config.initial_capital
        windows = self._split(klines)
        window_results: list[WindowResult] = []
        oos_equity_pieces: list[pd.Series] = []

        for i, (is_klines, oos_klines) in enumerate(windows):
            logger.info(
                "WFA window %d/%d  IS: %s ~ %s  OOS: %s ~ %s",
                i + 1, self.n_splits,
                is_klines.index[0].date(), is_klines.index[-1].date(),
                oos_klines.index[0].date(), oos_klines.index[-1].date(),
            )

            # --- IS 参数优化 ---
            opt = self._make_optimizer()
            is_result: OptimizeResult = opt.run(is_klines)
            best_params = is_result.best_params
            is_score = is_result.best_score

            # --- OOS 验证（用 IS 最优参数跑 OOS 数据）---
            oos_score, oos_equity = self._evaluate_oos(best_params, oos_klines, capital)

            window_results.append(WindowResult(
                window_id=i + 1,
                is_start=is_klines.index[0],
                is_end=is_klines.index[-1],
                oos_start=oos_klines.index[0],
                oos_end=oos_klines.index[-1],
                best_params=best_params,
                is_score=is_score,
                oos_score=oos_score,
                oos_equity=oos_equity,
            ))
            oos_equity_pieces.append(oos_equity)
            capital = float(oos_equity.iloc[-1]) if len(oos_equity) else capital

        # 拼接 OOS 权益曲线
        oos_equity_full = pd.concat(oos_equity_pieces).sort_index()
        oos_metrics = PerformanceAnalyzer(
            oos_equity_full, self.config.initial_capital
        ).compute()

        is_scores = [w.is_score for w in window_results if w.is_score != 0]
        oos_scores = [w.oos_score for w in window_results]
        avg_is = np.mean(is_scores) if is_scores else 1.0
        avg_oos = np.mean(oos_scores)
        degradation = avg_oos / avg_is if avg_is != 0 else float("nan")

        return WalkForwardResult(
            windows=window_results,
            metric=self.metric,
            oos_equity=oos_equity_full,
            oos_metrics=oos_metrics,
            degradation_ratio=degradation,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _split(self, klines: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """将 klines 切成 n_splits 个 (IS, OOS) 窗口对。"""
        n = len(klines)
        window_size = n // self.n_splits
        is_size = int(window_size * self.is_ratio)

        # 整个序列从第 0 条到第 n 条，滑动窗口
        # 窗口 i 的 IS 起点 = i * (window_size - is_size)
        oos_size = window_size - is_size
        pairs = []
        for i in range(self.n_splits):
            is_start = i * oos_size
            is_end = is_start + is_size
            oos_end = is_end + oos_size
            if oos_end > n:
                oos_end = n
            is_klines = klines.iloc[is_start:is_end]
            oos_klines = klines.iloc[is_end:oos_end]
            if len(is_klines) > 10 and len(oos_klines) > 10:
                pairs.append((is_klines, oos_klines))
        return pairs

    def _make_optimizer(self) -> GridSearchOptimizer:
        if self.optimizer_type == "genetic":
            from futuresquant.optimize.genetic import GeneticOptimizer
            return GeneticOptimizer(
                strategy_factory=self.strategy_factory,
                param_grid=self.param_grid,
                config=self.config,
                metric=self.metric,
                constraints=self.constraints,
                **self.genetic_kwargs,
            )
        return GridSearchOptimizer(
            strategy_factory=self.strategy_factory,
            param_grid=self.param_grid,
            config=self.config,
            metric=self.metric,
            constraints=self.constraints,
        )

    def _evaluate_oos(
        self,
        params: dict,
        oos_klines: pd.DataFrame,
        capital: float,
    ) -> tuple[float, pd.Series]:
        cfg = BacktestConfig(
            symbol=self.config.symbol,
            initial_capital=capital,
            slippage_ticks=self.config.slippage_ticks,
            specs=self.config.specs,
        )
        try:
            strategy = self.strategy_factory(**params)
            result = BacktestEngine(strategy, cfg).run(oos_klines)
            score = float(result.metrics.get(self.metric, float("-inf")))
            equity = result.account.equity_curve()
        except Exception as e:
            logger.warning("OOS eval failed %s: %s", params, e)
            score = float("-inf")
            equity = pd.Series(dtype=float)
        return score, equity
