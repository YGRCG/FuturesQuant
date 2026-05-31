"""
LiveBroker — wraps tqsdk TqApi to mirror the SimAccount interface.

Strategy code written against SimAccount's submit_order() / get_position()
works here without modification.  The key difference is that orders go to
the real exchange (or TqSim paper trading).

tqsdk order model
-----------------
  api.insert_order(symbol, direction, offset, volume, limit_price)
    direction : "BUY" | "SELL"
    offset    : "OPEN" | "CLOSE" | "CLOSETODAY"
  Returns a tqsdk Order object (live DataFrame row).

Position in tqsdk
-----------------
  pos = api.get_position(symbol)
  pos.pos_long  → long lots held
  pos.pos_short → short lots held (positive number)
  net = pos.pos_long - pos.pos_short
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from futuresquant.backtest.contract import ContractSpec, get_spec
from futuresquant.backtest.order import Direction, Fill, Offset, Order, OrderStatus, Position

logger = logging.getLogger(__name__)

# tqsdk direction / offset string mapping
_DIR_MAP = {Direction.LONG: "BUY", Direction.SHORT: "SELL"}
_OFF_MAP = {Offset.OPEN: "OPEN", Offset.CLOSE: "CLOSE"}


class LiveBroker:
    """
    Thin wrapper around tqsdk TqApi.

    Provides the same submit_order() / get_position() / mark_to_market()
    interface as SimAccount so strategies are plug-and-play between
    backtest and live modes.
    """

    def __init__(self, api: Any, initial_capital: float,
                 specs: Dict[str, ContractSpec] | None = None):
        """
        Parameters
        ----------
        api             : tqsdk TqApi instance (already connected)
        initial_capital : starting equity for drawdown tracking
        specs           : optional ContractSpec overrides
        """
        self._api = api
        self.initial_capital = initial_capital
        self._specs = specs or {}
        self._order_counter = 0
        self._fills: List[Fill] = []
        self._live_orders: Dict[int, Any] = {}   # order_id → tqsdk order object
        self._equity_curve: List[tuple] = []

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def submit_order(
        self,
        symbol: str,
        direction: Direction,
        offset: Offset,
        volume: int,
        time: pd.Timestamp,
        limit_price: float | None = None,
    ) -> Order:
        """Submit order to exchange via tqsdk."""
        self._order_counter += 1
        tq_dir = _DIR_MAP[direction]
        tq_off = _OFF_MAP[offset]

        tq_order = self._api.insert_order(
            symbol=symbol,
            direction=tq_dir,
            offset=tq_off,
            volume=volume,
            limit_price=limit_price,  # None → market order (tqsdk uses best-price)
        )

        order = Order(
            order_id=self._order_counter,
            symbol=symbol,
            direction=direction,
            offset=offset,
            volume=volume,
            submit_time=time,
            limit_price=limit_price,
        )
        self._live_orders[self._order_counter] = tq_order
        logger.info(
            "ORDER %d  %s %s %s  %d lots  @%s",
            self._order_counter, symbol, tq_dir, tq_off, volume,
            f"{limit_price:.2f}" if limit_price else "MKT",
        )
        return order

    # ------------------------------------------------------------------
    # Position / account queries (delegate to tqsdk live data)
    # ------------------------------------------------------------------

    def get_position(self, symbol: str) -> Position:
        tq_pos = self._api.get_position(symbol)
        net = int(tq_pos.pos_long - tq_pos.pos_short)
        avg = float(tq_pos.open_price_long if net > 0 else tq_pos.open_price_short)
        return Position(symbol=symbol, net=net, avg_price=avg)

    def mark_to_market(self, prices: Dict[str, float], time: pd.Timestamp) -> float:
        account = self._api.get_account()
        equity = float(account.balance)
        self._equity_curve.append((time, equity))
        return equity

    def equity_curve(self) -> pd.Series:
        if not self._equity_curve:
            return pd.Series(dtype=float)
        times, values = zip(*self._equity_curve)
        return pd.Series(values, index=pd.DatetimeIndex(times), name="equity")

    def cancel_all(self) -> None:
        """Cancel all outstanding tqsdk orders."""
        self._api.cancel_order("")   # empty string = cancel all

    # ------------------------------------------------------------------
    # Fill monitoring (call inside wait_update loop)
    # ------------------------------------------------------------------

    def collect_fills(self, time: pd.Timestamp) -> List[Fill]:
        """
        Check pending tqsdk orders for new fills; convert to Fill objects.
        Call this inside the while api.wait_update() loop.
        """
        new_fills: List[Fill] = []
        for order_id, tq_order in list(self._live_orders.items()):
            if not self._api.is_changing(tq_order, "status"):
                continue
            status = tq_order.status
            if status == "FINISHED":
                filled_vol = int(tq_order.volume_orign - tq_order.volume_left)
                if filled_vol > 0:
                    direction = Direction.LONG if tq_order.direction == "BUY" else Direction.SHORT
                    offset = Offset.OPEN if "OPEN" in tq_order.offset else Offset.CLOSE
                    spec = self._get_spec(tq_order.exchange_order_id.split(".")[0] if "." in tq_order.exchange_order_id else tq_order.instrument_id)
                    commission = SimAccount._calc_commission(spec, float(tq_order.trade_price), filled_vol)
                    fill = Fill(
                        order_id=order_id,
                        symbol=tq_order.instrument_id,
                        direction=direction,
                        offset=offset,
                        volume=filled_vol,
                        price=float(tq_order.trade_price),
                        time=time,
                        commission=commission,
                    )
                    new_fills.append(fill)
                    self._fills.append(fill)
                    logger.info(
                        "FILL  %s %s %s  %d lots @%.2f  commission=%.2f",
                        fill.symbol, fill.direction.value, fill.offset.value,
                        fill.volume, fill.price, fill.commission,
                    )
                del self._live_orders[order_id]
        return new_fills

    @property
    def fills(self) -> List[Fill]:
        return list(self._fills)

    def _get_spec(self, symbol_or_product: str) -> ContractSpec:
        product = "".join(c for c in symbol_or_product if c.isalpha()).upper()
        if product in self._specs:
            return self._specs[product]
        return get_spec(product)


# Avoid circular import — import only what's needed for commission calc
from futuresquant.backtest.account import SimAccount  # noqa: E402
