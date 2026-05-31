"""
RiskManager — pre-trade and intraday risk controls.

All checks are stateless w.r.t. tqsdk: they receive plain numbers so the
same class works in both backtest and live contexts.

Circuit breakers (checked in order):
  1. Trading halted flag — once tripped, no new opens allowed
  2. Daily loss limit     — total PnL for today < -limit → halt
  3. Max drawdown        — equity < peak * (1 - threshold) → halt new opens
  4. Position limit      — reject opens that would exceed max lots
  5. Notional limit      — reject orders with value > max_order_notional
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RiskState:
    """Mutable intraday risk state, reset each session."""
    peak_equity: float = 0.0
    daily_pnl: float = 0.0
    halted: bool = False          # True = stop all new opens
    halt_reason: str = ""


class RiskManager:
    """
    Pre-trade risk gate called before every order submission.

    Parameters
    ----------
    max_position_lots   : maximum net |position| per symbol (in lots)
    max_drawdown_pct    : halt new opens when drawdown exceeds this fraction
    daily_loss_limit    : halt all trading when today's loss exceeds this (yuan)
    max_order_notional  : reject any single order whose notional value exceeds this
    """

    def __init__(
        self,
        max_position_lots: int = 10,
        max_drawdown_pct: float = 0.05,
        daily_loss_limit: float = 5_000.0,
        max_order_notional: float = 500_000.0,
    ):
        self.max_position_lots = max_position_lots
        self.max_drawdown_pct = max_drawdown_pct
        self.daily_loss_limit = daily_loss_limit
        self.max_order_notional = max_order_notional
        self._state = RiskState()

    # ------------------------------------------------------------------
    # Called each bar — update intraday state
    # ------------------------------------------------------------------

    def update(self, equity: float, initial_capital: float) -> None:
        """Update peak equity and daily PnL; check circuit breakers."""
        state = self._state

        if equity > state.peak_equity:
            state.peak_equity = equity

        state.daily_pnl = equity - initial_capital   # simplified: total PnL as daily proxy

        # Drawdown circuit breaker
        if state.peak_equity > 0:
            dd = (state.peak_equity - equity) / state.peak_equity
            if dd >= self.max_drawdown_pct and not state.halted:
                state.halted = True
                state.halt_reason = (
                    f"Max drawdown breached: {dd:.2%} >= {self.max_drawdown_pct:.2%}"
                )
                logger.warning("RISK HALT — %s", state.halt_reason)

        # Daily loss circuit breaker
        if state.daily_pnl <= -self.daily_loss_limit and not state.halted:
            state.halted = True
            state.halt_reason = (
                f"Daily loss limit hit: {state.daily_pnl:.0f} <= -{self.daily_loss_limit:.0f}"
            )
            logger.warning("RISK HALT — %s", state.halt_reason)

    # ------------------------------------------------------------------
    # Pre-trade check — returns (approved, reason)
    # ------------------------------------------------------------------

    def check_open(
        self,
        symbol: str,
        volume: int,
        price: float,
        multiplier: float,
        current_position_lots: int,
    ) -> tuple[bool, str]:
        """
        Check whether an opening order is allowed.

        Returns
        -------
        (True, "")            → order approved
        (False, reason_str)   → order rejected
        """
        state = self._state

        if state.halted:
            return False, f"Trading halted: {state.halt_reason}"

        new_net = abs(current_position_lots) + volume
        if new_net > self.max_position_lots:
            return False, (
                f"Position limit: {new_net} lots would exceed max {self.max_position_lots}"
            )

        notional = volume * price * multiplier
        if notional > self.max_order_notional:
            return False, (
                f"Notional {notional:,.0f} exceeds limit {self.max_order_notional:,.0f}"
            )

        return True, ""

    def check_close(self, volume: int, current_position_lots: int) -> tuple[bool, str]:
        """Closing orders are almost always allowed; only reject if no position."""
        if abs(current_position_lots) < volume:
            return False, (
                f"Insufficient position to close {volume} lots "
                f"(have {abs(current_position_lots)})"
            )
        return True, ""

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def reset_daily(self, initial_capital: float) -> None:
        """Call at start of each trading day to reset daily PnL counter."""
        self._state = RiskState(peak_equity=initial_capital)
        logger.info("RiskManager daily state reset.")

    @property
    def is_halted(self) -> bool:
        return self._state.halted

    @property
    def halt_reason(self) -> str:
        return self._state.halt_reason

    @property
    def daily_pnl(self) -> float:
        return self._state.daily_pnl
