"""
Futures-specific structural factors.

Group 1 — Time features (no extra data needed):
    DayOfWeek       : 0=Mon … 4=Fri
    MinuteOfDay     : normalized minute-of-day [0, 1]
    SessionCode     : trading session (0=夜盘 1=早盘 2=午前 3=午后 NaN=非交易)
    DaysToExpiry    : calendar days to contract delivery month
                      (requires 'contract' column produced by ContinuousContract)

Group 2 — Lagged wrapper (no extra data needed):
    Lagged(factor, lag) : shift any Factor's output by N bars
"""

from __future__ import annotations

import re
import datetime

import numpy as np
import pandas as pd

from futuresquant.factors.base import Factor

# ---------------------------------------------------------------------------
# Contract name parser (shared with universe.py logic)
# ---------------------------------------------------------------------------
_CONTRACT_RE = re.compile(r"[A-Za-z]+(\d{2})(\d{2})$")


def _contract_to_expiry(contract_id: str) -> pd.Timestamp:
    m = _CONTRACT_RE.match(str(contract_id))
    if not m:
        return pd.NaT
    yy, mm = int(m.group(1)), int(m.group(2))
    year = 2000 + yy if yy <= 30 else 1900 + yy
    return pd.Timestamp(year=year, month=mm, day=1)


# ---------------------------------------------------------------------------
# Time features
# ---------------------------------------------------------------------------

class DayOfWeek(Factor):
    """Day of week: 0=Monday … 4=Friday."""

    name = "DayOfWeek"

    def compute(self, klines: pd.DataFrame) -> pd.Series:
        return pd.Series(
            klines.index.dayofweek.astype(float),
            index=klines.index,
            name=self.name,
        )


class MinuteOfDay(Factor):
    """
    Minutes since midnight, normalized to [0, 1].
    Encodes intraday position as a continuous scalar; ML models can learn
    non-linear time-of-day effects directly from this single feature.
    """

    name = "MinuteOfDay"

    def compute(self, klines: pd.DataFrame) -> pd.Series:
        mins = klines.index.hour * 60 + klines.index.minute
        return pd.Series(mins / 1439.0, index=klines.index, name=self.name)


class SessionCode(Factor):
    """
    Trading session code (product-configurable, default = FU/SHFE):
        0 = 夜盘  21:00–23:00
        1 = 早盘  09:00–10:15
        2 = 午前  10:30–11:30
        3 = 午后  13:30–15:00
        NaN = 非交易时段

    Pass custom ``sessions`` dict to override for other products, e.g.::

        # 大连商品交易所品种（无夜盘）
        SessionCode(sessions={
            1: (datetime.time(9,0),  datetime.time(11,30)),
            3: (datetime.time(13,30), datetime.time(15,0)),
        })
    """

    name = "SessionCode"

    _DEFAULT_SESSIONS = {
        0: (datetime.time(21, 0),  datetime.time(23, 59)),  # 夜盘
        1: (datetime.time(9,  0),  datetime.time(10, 15)),  # 早盘
        2: (datetime.time(10, 30), datetime.time(11, 30)),  # 午前
        3: (datetime.time(13, 30), datetime.time(15,  0)),  # 午后
    }

    def __init__(self, sessions: dict | None = None):
        self.sessions = sessions or self._DEFAULT_SESSIONS

    def compute(self, klines: pd.DataFrame) -> pd.Series:
        t = pd.Series(klines.index.time, index=klines.index)
        result = pd.Series(np.nan, index=klines.index, name=self.name)
        for code, (start, end) in self.sessions.items():
            if start <= end:
                mask = (t >= start) & (t <= end)
            else:
                # 跨午夜（夜盘延伸场景）
                mask = (t >= start) | (t <= end)
            result[mask] = float(code)
        return result


class DaysToExpiry(Factor):
    """
    Calendar days from each bar to the first day of the contract's delivery
    month.  Requires a ``contract`` column in klines (produced by
    ContinuousContract.build()).  Returns NaN for single-contract klines.

    This is a key futures-specific feature: roll pressure and liquidity
    change systematically as expiry approaches.
    """

    name = "DaysToExpiry"

    def compute(self, klines: pd.DataFrame) -> pd.Series:
        if "contract" not in klines.columns:
            return pd.Series(np.nan, index=klines.index, name=self.name)

        expiry = klines["contract"].map(_contract_to_expiry)
        days = (expiry - klines.index.normalize()).dt.days.astype(float)
        # Clip negative values (bars past nominal expiry): treat as 0
        days = days.clip(lower=0)
        return days.rename(self.name)


# ---------------------------------------------------------------------------
# Lagged wrapper
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Futures-specific structural factors (derived from multi-contract data)
# ---------------------------------------------------------------------------

class TermStructure(Factor):
    """
    Term structure: (next_contract_close - front_contract_close) / front_contract_close.

    Positive → contango (远期升水, holding cost high, bearish signal for longs).
    Negative → backwardation (近期升水, supply tight, bullish signal).

    This is a daily signal.  Pre-compute the column with
    ``ContinuousContract.build_term_structure()`` and attach it to klines::

        ts = cc.build_term_structure(start, end)
        klines["term_structure"] = ts.reindex(klines.index, method="ffill")

    Returns NaN for every bar when the ``term_structure`` column is absent.
    """

    name = "TermStructure"

    def compute(self, klines: pd.DataFrame) -> pd.Series:
        if "term_structure" not in klines.columns:
            return pd.Series(np.nan, index=klines.index, name=self.name)
        return klines["term_structure"].rename(self.name)


class OvernightGap(Factor):
    """
    Overnight information gap: (today_open - prev_day_close) / prev_day_close.

    Captures the price impact of news that arrived while the exchange was closed
    (e.g. overnight crude-oil moves affecting FU).  Computed once per trading
    day and forward-filled to every intraday bar.
    """

    name = "OvernightGap"

    def compute(self, klines: pd.DataFrame) -> pd.Series:
        daily_open = klines["open"].resample("1D").first()
        daily_prev_close = klines["close"].resample("1D").last().shift(1)
        daily_gap = (daily_open - daily_prev_close) / daily_prev_close
        # Forward-fill the daily value to every minute bar of that session
        return daily_gap.reindex(klines.index, method="ffill").rename(self.name)


class OIAcceleration(Factor):
    """
    Acceleration of open-interest change.

    ``OIChange_N``     = OI.pct_change(N)          — first derivative (velocity)
    ``OIAccel_N``      = OIChange_N.diff(N)         — second derivative (acceleration)

    Positive acceleration: new money flowing in at an increasing rate → trend strengthening.
    Negative acceleration: OI growth decelerating / reversing → trend fading.
    """

    def __init__(self, period: int = 20):
        self.period = period
        self.name = f"OIAccel_{period}"

    def compute(self, klines: pd.DataFrame) -> pd.Series:
        oi_change = klines["open_interest"].pct_change(self.period)
        return oi_change.diff(self.period).rename(self.name)


# ---------------------------------------------------------------------------
# Lagged wrapper
# ---------------------------------------------------------------------------

class Lagged(Factor):
    """
    Shift any factor's output by ``lag`` bars.

    Usage::

        from futuresquant.factors.technical import ROC
        from futuresquant.factors.technical.futures_structure import Lagged

        Lagged(ROC(20), lag=1)   # ROC_20 value from 1 bar ago
        Lagged(ROC(20), lag=5)   # ROC_20 value from 5 bars ago
    """

    def __init__(self, factor: Factor, lag: int):
        if lag < 1:
            raise ValueError(f"lag must be >= 1, got {lag}")
        self.factor = factor
        self.lag = lag
        self.name = f"{factor.name}_lag{lag}"

    def compute(self, klines: pd.DataFrame) -> pd.Series:
        return self.factor.compute(klines).shift(self.lag).rename(self.name)
