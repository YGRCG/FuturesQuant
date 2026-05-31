"""
网格搜索参数优化器。

用法
----
    from futuresquant.optimize.grid_search import GridSearchOptimizer

    optimizer = GridSearchOptimizer(
        strategy_factory=lambda fast, slow: MACrossStrategy(fast, slow),
        param_grid={'fast': [3, 5, 10], 'slow': [20, 30, 40, 60]},
        config=BacktestConfig(symbol='FU2210'),
        metric='sharpe',
        constraints=[lambda p: p['fast'] < p['slow']],
    )
    result = optimizer.run(klines)
    print(result.best_params)
    result.summary()
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd
from rich.progress import track

from futuresquant.backtest.engine import BacktestConfig, BacktestEngine
from futuresquant.strategy.base import StrategyBase

logger = logging.getLogger(__name__)

StrategyFactory = Callable[..., StrategyBase]
Constraint = Callable[[dict[str, Any]], bool]


@dataclass
class OptimizeResult:
    """All parameter combinations ranked by the chosen metric."""
    records: list[dict]          # [{params, sharpe, max_drawdown, …}, …]
    metric: str
    best_params: dict[str, Any]
    best_score: float

    def summary(self, top_n: int = 10) -> pd.DataFrame:
        df = pd.DataFrame(self.records).sort_values(self.metric, ascending=False)
        return df.head(top_n).reset_index(drop=True)

    def heatmap_data(self, x_param: str, y_param: str) -> pd.DataFrame:
        """Pivot table of metric values for two parameters (for imshow)."""
        df = pd.DataFrame(self.records)
        return df.pivot_table(index=y_param, columns=x_param, values=self.metric)


class GridSearchOptimizer:
    """
    全量枚举参数组合，对每组运行回测并排名。

    Parameters
    ----------
    strategy_factory : 接收 **params 关键字参数，返回 StrategyBase 实例
    param_grid       : {参数名: [候选值列表]}
    config           : BacktestConfig（symbol / initial_capital 等）
    metric           : 排名依据，支持 'sharpe'/'sortino'/'calmar'/'total_return'
    constraints      : 约束函数列表，返回 False 的参数组合直接跳过
    """

    def __init__(
        self,
        strategy_factory: StrategyFactory,
        param_grid: dict[str, list],
        config: BacktestConfig,
        metric: str = "sharpe",
        constraints: list[Constraint] | None = None,
    ):
        self.strategy_factory = strategy_factory
        self.param_grid = param_grid
        self.config = config
        self.metric = metric
        self.constraints = constraints or []

    def run(self, klines: pd.DataFrame) -> OptimizeResult:
        """枚举所有合法参数组合并回测。"""
        combinations = self._valid_combinations()
        logger.info("Grid search: %d valid combinations", len(combinations))

        records = []
        for params in track(combinations, description="Grid search"):
            record = self._evaluate(params, klines)
            if record:
                records.append(record)

        if not records:
            raise RuntimeError("No valid results produced.")

        records.sort(key=lambda r: r.get(self.metric, float("-inf")), reverse=True)
        best = records[0]
        return OptimizeResult(
            records=records,
            metric=self.metric,
            best_params={k: v for k, v in best.items() if k in self.param_grid},
            best_score=best[self.metric],
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _valid_combinations(self) -> list[dict]:
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        result = []
        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            if all(c(params) for c in self.constraints):
                result.append(params)
        return result

    def _evaluate(self, params: dict, klines: pd.DataFrame) -> dict | None:
        try:
            strategy = self.strategy_factory(**params)
            result = BacktestEngine(strategy, self.config).run(klines)
            m = result.metrics
            return {**params, **m}
        except Exception as e:
            logger.debug("Params %s failed: %s", params, e)
            return None
