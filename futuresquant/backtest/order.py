"""Order, Fill, Position dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import pandas as pd


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class Offset(str, Enum):
    OPEN = "OPEN"
    CLOSE = "CLOSE"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"


@dataclass
class Order:
    order_id: int
    symbol: str
    direction: Direction
    offset: Offset
    volume: int                        # lots
    submit_time: pd.Timestamp
    status: OrderStatus = OrderStatus.PENDING
    fill_price: Optional[float] = None
    fill_time: Optional[pd.Timestamp] = None
    # Optional: limit price (None = market order)
    limit_price: Optional[float] = None


@dataclass
class Fill:
    order_id: int
    symbol: str
    direction: Direction
    offset: Offset
    volume: int
    price: float
    time: pd.Timestamp
    commission: float


@dataclass
class Position:
    symbol: str
    net: int = 0           # positive = long lots, negative = short lots
    avg_price: float = 0.0 # volume-weighted average entry price

    @property
    def direction(self) -> Optional[Direction]:
        if self.net > 0:
            return Direction.LONG
        if self.net < 0:
            return Direction.SHORT
        return None

    def unrealised_pnl(self, current_price: float, multiplier: float) -> float:
        return self.net * (current_price - self.avg_price) * multiplier

    def update(self, direction: Direction, offset: Offset, volume: int, price: float) -> None:
        """Update position after a fill."""
        signed_vol = volume if direction == Direction.LONG else -volume

        if offset == Offset.OPEN:
            if self.net == 0:
                self.avg_price = price
            elif (self.net > 0) == (signed_vol > 0):
                # Adding to existing side
                total = abs(self.net) + volume
                self.avg_price = (abs(self.net) * self.avg_price + volume * price) / total
            self.net += signed_vol

        else:  # CLOSE
            self.net += signed_vol
            if self.net == 0:
                self.avg_price = 0.0
