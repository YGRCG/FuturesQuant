"""
遗传算法参数优化器。

适合参数空间大、全量枚举代价高的情况。
比网格搜索快，但结果是启发式的（不保证全局最优）。

算法细节
--------
- 编码  : 每个参数取离散候选值集的索引
- 选择  : 锦标赛选择（tournament size=3）
- 交叉  : 均匀交叉（每个基因独立以 0.5 概率互换）
- 变异  : 每个基因以 mutation_rate 概率替换为随机合法值
- 精英  : 每代保留 top-k 个体不变（elitism）
- 适应度: 指定 metric 的回测值（无效参数得分 = -inf）
- 缓存  : 相同参数组合不重复回测
"""

from __future__ import annotations

import logging
import random
from typing import Any, Callable

import pandas as pd
from rich.progress import track

from futuresquant.backtest.engine import BacktestConfig, BacktestEngine
from futuresquant.optimize.grid_search import Constraint, OptimizeResult, StrategyFactory

logger = logging.getLogger(__name__)


class GeneticOptimizer:
    """
    遗传算法参数优化。

    Parameters
    ----------
    strategy_factory  : 同 GridSearchOptimizer
    param_grid        : {参数名: [候选值列表]}
    config            : BacktestConfig
    metric            : 适应度指标
    constraints       : 约束函数（返回 False = 非法个体，直接给 -inf 适应度）
    population_size   : 种群大小
    n_generations     : 最大进化代数
    mutation_rate     : 单基因变异概率
    crossover_rate    : 交叉发生概率（否则直接复制父代）
    elitism_k         : 每代保留最优个体数
    tournament_size   : 锦标赛选择的参与者数
    seed              : 随机种子（可复现）
    """

    def __init__(
        self,
        strategy_factory: StrategyFactory,
        param_grid: dict[str, list],
        config: BacktestConfig,
        metric: str = "sharpe",
        constraints: list[Constraint] | None = None,
        population_size: int = 30,
        n_generations: int = 20,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.8,
        elitism_k: int = 3,
        tournament_size: int = 3,
        seed: int | None = 42,
    ):
        self.strategy_factory = strategy_factory
        self.param_grid = param_grid
        self.config = config
        self.metric = metric
        self.constraints = constraints or []
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_k = elitism_k
        self.tournament_size = tournament_size
        self._rng = random.Random(seed)
        self._cache: dict[tuple, float] = {}
        self._all_records: list[dict] = []

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def run(self, klines: pd.DataFrame) -> OptimizeResult:
        keys = list(self.param_grid.keys())
        population = self._init_population(keys)

        for gen in track(range(self.n_generations), description="Genetic optimize"):
            scored = [(ind, self._fitness(ind, keys, klines)) for ind in population]
            scored.sort(key=lambda x: x[1], reverse=True)
            best_score = scored[0][1]
            logger.info("Gen %d/%d  best %s=%.4f  cache_hits=%d",
                        gen + 1, self.n_generations, self.metric,
                        best_score, len(self._cache))

            # 精英保留
            elites = [ind for ind, _ in scored[: self.elitism_k]]
            # 生成下一代
            next_pop = list(elites)
            while len(next_pop) < self.population_size:
                p1 = self._tournament(scored)
                p2 = self._tournament(scored)
                child = self._crossover(p1, p2)
                child = self._mutate(child, keys)
                next_pop.append(child)
            population = next_pop

        # 最终评估并收集所有已评估个体
        all_records = [
            r for r in self._all_records
            if r.get(self.metric) is not None and r[self.metric] > float("-inf")
        ]
        all_records.sort(key=lambda r: r[self.metric], reverse=True)
        best = all_records[0] if all_records else {}
        best_params = {k: best[k] for k in keys if k in best}

        return OptimizeResult(
            records=all_records,
            metric=self.metric,
            best_params=best_params,
            best_score=best.get(self.metric, float("-inf")),
        )

    # ------------------------------------------------------------------
    # GA 操作
    # ------------------------------------------------------------------

    def _init_population(self, keys: list[str]) -> list[list[int]]:
        """随机初始化种群（索引编码）。"""
        pop = []
        attempts = 0
        while len(pop) < self.population_size and attempts < self.population_size * 10:
            ind = [self._rng.randrange(len(self.param_grid[k])) for k in keys]
            if self._is_valid(ind, keys):
                pop.append(ind)
            attempts += 1
        return pop

    def _fitness(self, ind: list[int], keys: list[str], klines: pd.DataFrame) -> float:
        params = {k: self.param_grid[k][i] for k, i in zip(keys, ind)}
        cache_key = tuple(params[k] for k in keys)

        if cache_key in self._cache:
            return self._cache[cache_key]

        if not self._is_valid(ind, keys):
            self._cache[cache_key] = float("-inf")
            return float("-inf")

        try:
            strategy = self.strategy_factory(**params)
            result = BacktestEngine(strategy, self.config).run(klines)
            m = result.metrics
            score = float(m.get(self.metric, float("-inf")))
            self._cache[cache_key] = score
            self._all_records.append({**params, **m})
        except Exception as e:
            logger.debug("Fitness eval failed %s: %s", params, e)
            score = float("-inf")
            self._cache[cache_key] = score

        return score

    def _is_valid(self, ind: list[int], keys: list[str]) -> bool:
        params = {k: self.param_grid[k][i] for k, i in zip(keys, ind)}
        return all(c(params) for c in self.constraints)

    def _tournament(self, scored: list[tuple[list[int], float]]) -> list[int]:
        contestants = self._rng.sample(scored, min(self.tournament_size, len(scored)))
        return max(contestants, key=lambda x: x[1])[0]

    def _crossover(self, p1: list[int], p2: list[int]) -> list[int]:
        if self._rng.random() > self.crossover_rate:
            return list(p1)
        return [
            p1[i] if self._rng.random() < 0.5 else p2[i]
            for i in range(len(p1))
        ]

    def _mutate(self, ind: list[int], keys: list[str]) -> list[int]:
        result = list(ind)
        for i, k in enumerate(keys):
            if self._rng.random() < self.mutation_rate:
                result[i] = self._rng.randrange(len(self.param_grid[k]))
        return result
