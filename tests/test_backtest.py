"""End-to-end backtest engine tests — no tqsdk credentials needed."""

import pytest
import pandas as pd
import numpy as np

from futuresquant.data.loader import FuturesDataLoader
from futuresquant.backtest.engine import BacktestEngine, BacktestConfig
from futuresquant.backtest.account import SimAccount
from futuresquant.backtest.order import Direction, Offset
from futuresquant.backtest.analyzer import PerformanceAnalyzer
from futuresquant.strategy.examples.ma_cross import MACrossStrategy

DATA_DIR = r"I:\stock\FuturesQuant\raw_data\1min_FU"
SYMBOL = "FU2210"


@pytest.fixture(scope="module")
def klines():
    loader = FuturesDataLoader(DATA_DIR)
    return loader.load(SYMBOL, start="2022-01-01", end="2022-10-31")


@pytest.fixture(scope="module")
def backtest_result(klines):
    strategy = MACrossStrategy(fast=5, slow=20)
    config = BacktestConfig(symbol=SYMBOL, initial_capital=1_000_000)
    engine = BacktestEngine(strategy, config)
    return engine.run(klines)


# ------------------------------------------------------------------
# Account / SimAccount unit tests
# ------------------------------------------------------------------

def test_account_initial_cash():
    acc = SimAccount(500_000)
    assert acc.cash == 500_000


def test_account_buy_open_reduces_cash(klines):
    acc = SimAccount(1_000_000)
    ts = klines.index[0]
    acc.submit_order(SYMBOL, Direction.LONG, Offset.OPEN, 1, ts)
    acc.fill_pending_orders({SYMBOL: 2800.0}, ts, slippage_ticks=0)
    # FU: multiplier=10, margin_ratio=0.12, commission=0.4
    expected_margin = 2800 * 10 * 0.12
    expected_cash = 1_000_000 - expected_margin - 0.4
    assert abs(acc.cash - expected_cash) < 1.0


def test_account_position_tracked(klines):
    acc = SimAccount(1_000_000)
    ts = klines.index[0]
    acc.submit_order(SYMBOL, Direction.LONG, Offset.OPEN, 2, ts)
    acc.fill_pending_orders({SYMBOL: 2800.0}, ts, slippage_ticks=0)
    assert acc.get_position(SYMBOL).net == 2


def test_account_close_books_pnl(klines):
    acc = SimAccount(1_000_000)
    ts0, ts1 = klines.index[0], klines.index[1]
    acc.submit_order(SYMBOL, Direction.LONG, Offset.OPEN, 1, ts0)
    acc.fill_pending_orders({SYMBOL: 2800.0}, ts0, slippage_ticks=0)
    cash_after_open = acc.cash

    acc.submit_order(SYMBOL, Direction.SHORT, Offset.CLOSE, 1, ts1)
    acc.fill_pending_orders({SYMBOL: 2850.0}, ts1, slippage_ticks=0)

    expected_pnl = (2850 - 2800) * 10 * 1   # 50 × 10 = 500
    assert acc.get_position(SYMBOL).net == 0
    assert acc.cash > cash_after_open  # realised gain


# ------------------------------------------------------------------
# Engine end-to-end
# ------------------------------------------------------------------

def test_engine_produces_equity_curve(backtest_result):
    eq = backtest_result.account.equity_curve()
    assert len(eq) > 0
    assert isinstance(eq.index, pd.DatetimeIndex)


def test_engine_equity_starts_near_capital(backtest_result):
    eq = backtest_result.account.equity_curve()
    assert abs(eq.iloc[0] / 1_000_000 - 1) < 0.05


def test_engine_fills_recorded(backtest_result):
    fills = backtest_result.account.fills
    assert len(fills) > 0, "Expected at least one fill"


def test_engine_no_remaining_position(backtest_result):
    pos = backtest_result.account.get_position(SYMBOL)
    assert pos.net == 0, "Engine should close all positions at end"


def test_engine_metrics_keys(backtest_result):
    m = backtest_result.metrics
    for key in ("total_return", "sharpe", "max_drawdown", "annual_return"):
        assert key in m, f"Missing metric: {key}"


def test_engine_max_drawdown_negative(backtest_result):
    assert backtest_result.metrics["max_drawdown"] <= 0


# ------------------------------------------------------------------
# PerformanceAnalyzer
# ------------------------------------------------------------------

def test_analyzer_flat_equity():
    times = pd.date_range("2022-01-01", periods=100, freq="1min")
    eq = pd.Series(1_000_000.0, index=times)
    m = PerformanceAnalyzer(eq, 1_000_000).compute()
    assert m["total_return"] == pytest.approx(0.0, abs=1e-6)
    assert m["max_drawdown"] == pytest.approx(0.0, abs=1e-6)


def test_analyzer_monotone_increase():
    times = pd.date_range("2022-01-01", periods=500, freq="1min")
    eq = pd.Series(np.linspace(1_000_000, 1_200_000, 500), index=times)
    m = PerformanceAnalyzer(eq, 1_000_000).compute()
    assert m["total_return"] == pytest.approx(0.2, abs=1e-4)
    assert m["max_drawdown"] == pytest.approx(0.0, abs=1e-6)
    assert m["sharpe"] > 0
