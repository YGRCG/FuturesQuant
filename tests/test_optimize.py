"""
优化器单元测试（快速版，使用小参数空间和短数据）。
"""
import pytest
import pandas as pd

from futuresquant.data.loader import FuturesDataLoader
from futuresquant.backtest.engine import BacktestConfig
from futuresquant.optimize.grid_search import GridSearchOptimizer, OptimizeResult
from futuresquant.optimize.genetic import GeneticOptimizer
from futuresquant.optimize.walk_forward import WalkForwardOptimizer
from futuresquant.strategy.examples import MACrossStrategy

DATA_DIR = r"I:\stock\FuturesQuant\raw_data\1min_FU"
SYMBOL = "FU2210"
CONFIG = BacktestConfig(symbol=SYMBOL, initial_capital=500_000)

FACTORY = lambda fast, slow: MACrossStrategy(fast=fast, slow=slow)
PARAM_GRID = {"fast": [3, 5], "slow": [20, 30]}
CONSTRAINTS = [lambda p: p["fast"] < p["slow"]]


@pytest.fixture(scope="module")
def klines():
    # 只取 2022 Q3，减少测试时间
    return FuturesDataLoader(DATA_DIR).load(SYMBOL, start="2022-07-01", end="2022-09-30")


# ------------------------------------------------------------------
# GridSearchOptimizer
# ------------------------------------------------------------------

def test_grid_search_returns_result(klines):
    opt = GridSearchOptimizer(FACTORY, PARAM_GRID, CONFIG,
                               metric="sharpe", constraints=CONSTRAINTS)
    result = opt.run(klines)
    assert isinstance(result, OptimizeResult)
    assert result.best_params
    assert "fast" in result.best_params
    assert "slow" in result.best_params


def test_grid_search_constraints_respected(klines):
    opt = GridSearchOptimizer(FACTORY, PARAM_GRID, CONFIG, constraints=CONSTRAINTS)
    result = opt.run(klines)
    for r in result.records:
        assert r["fast"] < r["slow"], "Constraint violated"


def test_grid_search_summary(klines):
    opt = GridSearchOptimizer(FACTORY, PARAM_GRID, CONFIG, constraints=CONSTRAINTS)
    result = opt.run(klines)
    df = result.summary()
    assert isinstance(df, pd.DataFrame)
    assert len(df) <= len(result.records)
    assert "sharpe" in df.columns


def test_grid_search_best_score_is_max(klines):
    opt = GridSearchOptimizer(FACTORY, PARAM_GRID, CONFIG,
                               metric="sharpe", constraints=CONSTRAINTS)
    result = opt.run(klines)
    all_scores = [r["sharpe"] for r in result.records]
    assert result.best_score == max(all_scores)


def test_grid_all_metric_support(klines):
    for metric in ("sharpe", "total_return", "calmar"):
        opt = GridSearchOptimizer(FACTORY, PARAM_GRID, CONFIG,
                                   metric=metric, constraints=CONSTRAINTS)
        result = opt.run(klines)
        assert result.metric == metric


# ------------------------------------------------------------------
# GeneticOptimizer
# ------------------------------------------------------------------

def test_genetic_returns_result(klines):
    opt = GeneticOptimizer(
        FACTORY, PARAM_GRID, CONFIG,
        metric="sharpe", constraints=CONSTRAINTS,
        population_size=6, n_generations=3, seed=0,
    )
    result = opt.run(klines)
    assert isinstance(result, OptimizeResult)
    assert result.best_params
    assert result.best_score > float("-inf")


def test_genetic_constraints_respected(klines):
    opt = GeneticOptimizer(
        FACTORY, PARAM_GRID, CONFIG, constraints=CONSTRAINTS,
        population_size=6, n_generations=3, seed=1,
    )
    result = opt.run(klines)
    for r in result.records:
        assert r["fast"] < r["slow"]


def test_genetic_reproducible(klines):
    kwargs = dict(population_size=6, n_generations=3, seed=42)
    r1 = GeneticOptimizer(FACTORY, PARAM_GRID, CONFIG, **kwargs).run(klines)
    r2 = GeneticOptimizer(FACTORY, PARAM_GRID, CONFIG, **kwargs).run(klines)
    assert r1.best_params == r2.best_params
    assert r1.best_score == pytest.approx(r2.best_score)


# ------------------------------------------------------------------
# WalkForwardOptimizer
# ------------------------------------------------------------------

def test_walk_forward_runs(klines):
    wf = WalkForwardOptimizer(
        FACTORY, PARAM_GRID, CONFIG,
        metric="sharpe", constraints=CONSTRAINTS,
        n_splits=3, is_ratio=0.6,
    )
    result = wf.run(klines)
    assert len(result.windows) == 3
    assert isinstance(result.oos_equity, pd.Series)
    assert len(result.oos_equity) > 0


def test_walk_forward_summary(klines):
    wf = WalkForwardOptimizer(
        FACTORY, PARAM_GRID, CONFIG,
        constraints=CONSTRAINTS, n_splits=3,
    )
    result = wf.run(klines)
    df = result.summary()
    assert len(df) == 3
    assert "衰减比" in df.columns


def test_walk_forward_degradation_ratio_finite(klines):
    wf = WalkForwardOptimizer(
        FACTORY, PARAM_GRID, CONFIG,
        constraints=CONSTRAINTS, n_splits=3,
    )
    result = wf.run(klines)
    assert not (result.degradation_ratio != result.degradation_ratio)  # not NaN check


def test_walk_forward_param_stability(klines):
    wf = WalkForwardOptimizer(
        FACTORY, PARAM_GRID, CONFIG,
        constraints=CONSTRAINTS, n_splits=3,
    )
    result = wf.run(klines)
    stab = result.param_stability()
    assert "fast" in stab.columns
    assert "slow" in stab.columns
    assert len(stab) == 3


def test_walk_forward_oos_metrics_keys(klines):
    wf = WalkForwardOptimizer(
        FACTORY, PARAM_GRID, CONFIG,
        constraints=CONSTRAINTS, n_splits=3,
    )
    result = wf.run(klines)
    for key in ("sharpe", "max_drawdown", "total_return"):
        assert key in result.oos_metrics
