"""
BacktestEngine — bar-by-bar simulation loop.

Execution model (avoids lookahead bias):
  Bar T  → strategy sees bar T, calls ctx.buy_open() / ctx.sell_open() etc.
  Bar T+1 → pending orders filled at bar T+1's open price (+/- slippage)

Usage
-----
    from futuresquant.backtest.engine import BacktestEngine, BacktestConfig
    from futuresquant.data.loader import FuturesDataLoader

    loader = FuturesDataLoader(r"I:/stock/FuturesQuant/1min_FU")
    klines = loader.load("FU2210", start="2022-01-01", end="2022-10-31")

    result = BacktestEngine(MyStrategy(), BacktestConfig("FU2210")).run(klines)
    result.report()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import pandas as pd
from rich.progress import track

from futuresquant.backtest.account import SimAccount
from futuresquant.backtest.contract import ContractSpec
from futuresquant.backtest.analyzer import PerformanceAnalyzer
from futuresquant.strategy.base import Context, StrategyBase


@dataclass
class BacktestConfig:
    symbol: str
    initial_capital: float = 1_000_000.0
    slippage_ticks: int = 1
    # Optional per-product spec overrides (if defaults in contract.py are wrong)
    specs: Dict[str, ContractSpec] = field(default_factory=dict)


class BacktestResult:
    def __init__(self, account: SimAccount, klines: pd.DataFrame, config: BacktestConfig):
        self.account = account
        self.klines = klines
        self.config = config
        self.metrics = PerformanceAnalyzer(account.equity_curve(), config.initial_capital).compute()

    def report(self, open_browser: bool = True) -> None:
        from futuresquant.backtest.report import build_report
        build_report(self, open_browser=open_browser)

    def print_summary(self) -> None:
        m = self.metrics
        print(f"\n{'='*50}")
        print(f"  Backtest: {self.config.symbol}")
        print(f"{'='*50}")
        for k, v in m.items():
            if isinstance(v, float):
                print(f"  {k:<28} {v:>10.4f}")
            else:
                print(f"  {k:<28} {str(v):>10}")
        print(f"{'='*50}\n")


class BacktestEngine:
    """Drives bar-by-bar strategy simulation."""

    def __init__(self, strategy: StrategyBase, config: BacktestConfig):
        self.strategy = strategy
        self.config = config

    def run(self, klines: pd.DataFrame) -> BacktestResult:
        """
        Run backtest on a K-line DataFrame.

        Parameters
        ----------
        klines : 1-min (or any frequency) DataFrame with DatetimeIndex and
                 columns: open/high/low/close/volume/amount/open_interest

        Returns
        -------
        BacktestResult containing filled account, equity curve, and metrics.
        """
        cfg = self.config
        account = SimAccount(cfg.initial_capital, specs=cfg.specs)
        strategy = self.strategy

        strategy.on_start(klines)
        warmup = max(strategy.warmup_bars, 0)

        pending_order_symbols: set[str] = set()   # symbols with orders queued for next-bar open

        for i, (ts, bar) in enumerate(track(
            klines.iterrows(), total=len(klines), description="Backtesting"
        )):
            # --- Step 1: fill orders queued from the previous bar at THIS bar's open ---
            if pending_order_symbols:
                open_prices = {sym: bar["open"] for sym in pending_order_symbols}
                fills = account.fill_pending_orders(open_prices, ts, cfg.slippage_ticks)
                for fill in fills:
                    strategy.on_fill(fill)
                pending_order_symbols = set()

            # --- Step 2: mark-to-market at current close ---
            account.mark_to_market({cfg.symbol: bar["close"]}, ts)

            # --- Step 3: let strategy act (skip warmup bars) ---
            if i >= warmup:
                ctx = Context(
                    symbol=cfg.symbol,
                    bar=bar,
                    account=account,
                    timestamp=ts,
                )
                strategy.on_bar(ctx)

                # Queue any new orders to fill at NEXT bar's open
                if account._pending_orders:
                    pending_order_symbols.add(cfg.symbol)

        # Close any remaining open positions at last bar's close
        account.cancel_all()
        last_bar = klines.iloc[-1]
        pos = account.get_position(cfg.symbol)
        if pos.net != 0:
            last_ts = klines.index[-1]
            from futuresquant.backtest.order import Direction, Offset
            account.submit_order(
                cfg.symbol,
                Direction.SHORT if pos.net > 0 else Direction.LONG,
                Offset.CLOSE,
                abs(pos.net),
                last_ts,
            )
            account.fill_pending_orders({cfg.symbol: last_bar["close"]}, last_ts, 0)
            # 用微小偏移避免与循环中最后一个权益点时间戳重复
            close_ts = last_ts + pd.Timedelta(milliseconds=1)
            account.mark_to_market({cfg.symbol: last_bar["close"]}, close_ts)

        strategy.on_end(account)
        return BacktestResult(account, klines, cfg)
