"""
Strategy base class.

Strategies implement on_bar() and optionally on_start() / on_end().
The Context object provides the order-placement API, keeping strategy code
independent of whether it runs in backtest or live mode.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from futuresquant.backtest.order import Direction, Fill, Offset, Order

if TYPE_CHECKING:
    from futuresquant.backtest.account import SimAccount


@dataclass
class Context:
    """
    Passed to every on_bar() call.  Wraps the account so strategy code
    never touches SimAccount directly.
    """
    symbol: str
    bar: pd.Series             # current bar (open/high/low/close/volume/…)
    account: "SimAccount"
    timestamp: pd.Timestamp

    # ------------------------------------------------------------------
    # Order helpers — mirror tqsdk semantics
    # ------------------------------------------------------------------

    def buy_open(self, volume: int = 1, limit_price: float | None = None) -> Order:
        """买入开仓 (做多)"""
        return self.account.submit_order(
            self.symbol, Direction.LONG, Offset.OPEN, volume, self.timestamp, limit_price
        )

    def sell_close(self, volume: int | None = None, limit_price: float | None = None) -> Order:
        """卖出平仓 (平多)"""
        pos = self.account.get_position(self.symbol)
        vol = volume if volume is not None else max(pos.net, 0)
        return self.account.submit_order(
            self.symbol, Direction.SHORT, Offset.CLOSE, vol, self.timestamp, limit_price
        )

    def sell_open(self, volume: int = 1, limit_price: float | None = None) -> Order:
        """卖出开仓 (做空)"""
        return self.account.submit_order(
            self.symbol, Direction.SHORT, Offset.OPEN, volume, self.timestamp, limit_price
        )

    def buy_close(self, volume: int | None = None, limit_price: float | None = None) -> Order:
        """买入平仓 (平空)"""
        pos = self.account.get_position(self.symbol)
        vol = volume if volume is not None else abs(min(pos.net, 0))
        return self.account.submit_order(
            self.symbol, Direction.LONG, Offset.CLOSE, vol, self.timestamp, limit_price
        )

    def close_position(self) -> Order | None:
        """平掉全部持仓（无论方向）."""
        pos = self.account.get_position(self.symbol)
        if pos.net > 0:
            return self.sell_close(pos.net)
        if pos.net < 0:
            return self.buy_close(abs(pos.net))
        return None

    @property
    def position(self) -> int:
        """Current net position in lots (positive=long, negative=short)."""
        return self.account.get_position(self.symbol).net


class StrategyBase(ABC):
    """
    Base class for all strategies.

    Lifecycle:
        on_start(klines)  → called once before the first bar
        on_bar(ctx)       → called on every bar (after warmup)
        on_fill(fill)     → called after each fill
        on_end(account)   → called once after the last bar
    """

    # Override in subclass to set a minimum warmup length (bars to skip at start)
    warmup_bars: int = 0

    def on_start(self, klines: pd.DataFrame) -> None:
        """Called once with the full K-line DataFrame before simulation starts."""

    @abstractmethod
    def on_bar(self, ctx: Context) -> None:
        """Core strategy logic. Place orders via ctx.buy_open() etc."""

    def on_fill(self, fill: Fill) -> None:
        """Called after each fill. Override for custom fill handling."""

    def on_end(self, account: "SimAccount") -> None:
        """Called after the last bar. Override for cleanup / logging."""
