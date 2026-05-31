"""SimAccount — simulated futures trading account."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import pandas as pd

from futuresquant.backtest.contract import ContractSpec, get_spec
from futuresquant.backtest.order import Direction, Fill, Offset, Order, OrderStatus, Position


class SimAccount:
    """
    Tracks cash, positions, margin, and equity for a simulated account.

    Orders are submitted via submit_order(); they are held as PENDING until
    the engine calls fill_pending_orders() at the next bar's open.
    """

    def __init__(self, initial_capital: float, specs: Dict[str, ContractSpec] | None = None):
        self.initial_capital = initial_capital
        self._specs = specs or {}           # product → ContractSpec overrides
        self._cash = initial_capital        # available cash
        self._positions: Dict[str, Position] = defaultdict(lambda: Position(symbol=""))
        self._pending_orders: List[Order] = []
        self._fills: List[Fill] = []
        self._equity_curve: List[tuple] = []  # (timestamp, equity)
        self._order_counter = 0

    # ------------------------------------------------------------------
    # Order submission (called from strategy)
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
        self._order_counter += 1
        order = Order(
            order_id=self._order_counter,
            symbol=symbol,
            direction=direction,
            offset=offset,
            volume=volume,
            submit_time=time,
            limit_price=limit_price,
        )
        self._pending_orders.append(order)
        return order

    # ------------------------------------------------------------------
    # Engine interface
    # ------------------------------------------------------------------

    def fill_pending_orders(
        self,
        prices: Dict[str, float],
        time: pd.Timestamp,
        slippage_ticks: int = 1,
    ) -> List[Fill]:
        """
        Fill all pending market orders at the given prices (next bar open).
        Limit orders are filled only if price is within limit.
        """
        new_fills: List[Fill] = []
        remaining: List[Order] = []

        for order in self._pending_orders:
            if order.symbol not in prices:
                remaining.append(order)
                continue

            raw_price = prices[order.symbol]
            spec = self._get_spec(order.symbol)
            slip = slippage_ticks * spec.tick_size
            fill_price = raw_price + slip if order.direction == Direction.LONG else raw_price - slip

            # Limit order check
            if order.limit_price is not None:
                if order.direction == Direction.LONG and fill_price > order.limit_price:
                    remaining.append(order)
                    continue
                if order.direction == Direction.SHORT and fill_price < order.limit_price:
                    remaining.append(order)
                    continue

            commission = self._calc_commission(spec, fill_price, order.volume)
            fill = Fill(
                order_id=order.order_id,
                symbol=order.symbol,
                direction=order.direction,
                offset=order.offset,
                volume=order.volume,
                price=fill_price,
                time=time,
                commission=commission,
            )
            self._apply_fill(fill, spec)
            order.status = OrderStatus.FILLED
            order.fill_price = fill_price
            order.fill_time = time
            new_fills.append(fill)
            self._fills.append(fill)

        self._pending_orders = remaining
        return new_fills

    def mark_to_market(self, prices: Dict[str, float], time: pd.Timestamp) -> float:
        """Recompute equity = cash + sum of unrealised PnL. Records equity curve point."""
        equity = self._cash
        for symbol, pos in self._positions.items():
            if pos.net != 0 and symbol in prices:
                spec = self._get_spec(symbol)
                equity += pos.unrealised_pnl(prices[symbol], spec.multiplier)
        self._equity_curve.append((time, equity))
        return equity

    def cancel_all(self) -> None:
        for o in self._pending_orders:
            o.status = OrderStatus.CANCELLED
        self._pending_orders = []

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def fills(self) -> List[Fill]:
        return list(self._fills)

    @property
    def positions(self) -> Dict[str, Position]:
        return dict(self._positions)

    def get_position(self, symbol: str) -> Position:
        if symbol not in self._positions:
            self._positions[symbol] = Position(symbol=symbol)
        return self._positions[symbol]

    def equity_curve(self) -> pd.Series:
        if not self._equity_curve:
            return pd.Series(dtype=float)
        times, values = zip(*self._equity_curve)
        return pd.Series(values, index=pd.DatetimeIndex(times), name="equity")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _apply_fill(self, fill: Fill, spec: ContractSpec) -> None:
        pos = self.get_position(fill.symbol)
        pos.symbol = fill.symbol

        # Cash impact: opening adds margin requirement, closing releases it and books PnL
        if fill.offset == Offset.OPEN:
            margin = fill.volume * fill.price * spec.multiplier * spec.margin_ratio
            self._cash -= margin
        else:
            # Release margin at avg entry price and book realised PnL
            margin_released = fill.volume * pos.avg_price * spec.multiplier * spec.margin_ratio
            if fill.direction == Direction.LONG:
                realised_pnl = fill.volume * (fill.price - pos.avg_price) * spec.multiplier
            else:
                realised_pnl = fill.volume * (pos.avg_price - fill.price) * spec.multiplier
            self._cash += margin_released + realised_pnl

        self._cash -= fill.commission
        pos.update(fill.direction, fill.offset, fill.volume, fill.price)

    def _get_spec(self, symbol: str) -> ContractSpec:
        product = "".join(c for c in symbol.split(".")[-1] if c.isalpha()).upper()
        if product in self._specs:
            return self._specs[product]
        return get_spec(product)

    @staticmethod
    def _calc_commission(spec: ContractSpec, price: float, volume: int) -> float:
        if spec.commission_per_lot > 0:
            return spec.commission_per_lot * volume
        return price * spec.multiplier * volume * spec.commission_rate
