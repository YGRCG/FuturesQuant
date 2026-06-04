"""
Continuous contract construction and main-contract (主力合约) identification.

Strategy: at each minute bar, the "main contract" is the one with the
highest open_interest among currently active contracts.  Roll happens
when a different contract takes the lead.

Adjustment methods
------------------
- "none"   : raw prices, price gap at roll
- "back"   : back-adjusted (前复权 from the perspective of today)
             Older contracts are shifted down/up so the series is
             continuous at the roll point.
- "ratio"  : ratio-adjusted (multiplicative)
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from .loader import ContractInfo, FuturesDataLoader

AdjustMethod = Literal["none", "back", "ratio"]


class ContinuousContract:
    """Build a continuous contract series from individual CSV files."""

    def __init__(
        self,
        loader: FuturesDataLoader,
        product: str,
        adjust: AdjustMethod = "back",
        roll_n_days_before_expiry: int = 5,
        roll_hysteresis: float = 0.20,
        roll_min_hold_days: int = 3,
    ):
        self.loader = loader
        self.product = product.upper()
        self.adjust = adjust
        self.roll_days = roll_n_days_before_expiry
        self.roll_hysteresis = roll_hysteresis
        self.roll_min_hold_days = roll_min_hold_days

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """
        Return a continuous 1-min K-line DataFrame.

        Columns: open, high, low, close, volume, amount, open_interest,
                 contract  (which underlying contract is active at each bar)
        """
        contracts = self.loader.list_contracts(self.product)
        if not contracts:
            raise ValueError(f"No contracts found for product {self.product}")

        all_data = {
            c.contract_id: self.loader.load(c, start=start, end=end)
            for c in contracts
        }
        # Drop empty frames
        all_data = {k: v for k, v in all_data.items() if not v.empty}
        if not all_data:
            raise ValueError("All contract data is empty for the requested date range.")

        roll_schedule = self._build_roll_schedule(contracts, all_data)
        continuous = self._stitch(all_data, roll_schedule)

        if self.adjust != "none":
            continuous = self._adjust(continuous)

        return continuous

    def build_term_structure(
        self,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.Series:
        """
        Daily term structure: (next_contract_close - front_contract_close) / front_contract_close.

        Front = contract with the highest open interest (excluding near-expiry).
        Next  = contract with the nearest expiry after the front contract.

        Returns a daily Series named "term_structure".  NaN on days when no
        next contract data is available (e.g. the last available delivery month).

        Usage in notebooks::

            ts = cc.build_term_structure(start=START_DATE, end=END_DATE)
            klines["term_structure"] = ts.reindex(klines.index, method="ffill")
        """
        contracts = self.loader.list_contracts(self.product)
        if not contracts:
            return pd.Series(dtype=float, name="term_structure")

        all_data: dict[str, pd.DataFrame] = {}
        for c in contracts:
            df = self.loader.load(c, start=start, end=end)
            if not df.empty:
                all_data[c.contract_id] = df

        if not all_data:
            return pd.Series(dtype=float, name="term_structure")

        expiry_map: dict[str, pd.Timestamp] = {
            c.contract_id: c.expiry_month for c in contracts
        }
        sorted_cids = sorted(
            all_data.keys(), key=lambda c: expiry_map.get(c, pd.Timestamp.max)
        )

        # Daily close and OI matrices
        close_matrix = pd.DataFrame(
            {cid: all_data[cid]["close"].resample("1D").last() for cid in sorted_cids}
        )
        oi_matrix = pd.DataFrame(
            {cid: all_data[cid]["open_interest"].resample("1D").last() for cid in sorted_cids}
        )

        # Mask near-expiry contracts
        for cid in sorted_cids:
            cutoff = expiry_map[cid] + pd.Timedelta(days=self.roll_days)
            oi_matrix.loc[oi_matrix.index >= cutoff, cid] = np.nan

        # Front contract = highest OI per day
        front_contract = oi_matrix.idxmax(axis=1)
        front_contract[oi_matrix.isna().all(axis=1)] = np.nan

        result = pd.Series(np.nan, index=close_matrix.index, name="term_structure")

        # Vectorised: iterate over unique front contracts (typically ~30 over 15 years)
        for front_cid in front_contract.dropna().unique():
            front_expiry = expiry_map.get(front_cid)
            if front_expiry is None:
                continue

            next_candidates = [
                c for c in sorted_cids
                if expiry_map.get(c, pd.Timestamp.max) > front_expiry
            ]
            if not next_candidates:
                continue
            next_cid = next_candidates[0]

            if front_cid not in close_matrix.columns or next_cid not in close_matrix.columns:
                continue

            dates = front_contract[front_contract == front_cid].index
            front_close = close_matrix.loc[dates, front_cid]
            next_close = close_matrix.loc[dates, next_cid]

            valid = front_close.notna() & next_close.notna() & (front_close != 0)
            result.loc[front_close[valid].index] = (
                (next_close[valid] - front_close[valid]) / front_close[valid]
            )

        return result

    def main_contract_series(
        self,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.Series:
        """Return a Series mapping datetime → active contract_id."""
        contracts = self.loader.list_contracts(self.product)
        all_data = {
            c.contract_id: self.loader.load(c, start=start, end=end)
            for c in contracts
        }
        all_data = {k: v for k, v in all_data.items() if not v.empty}
        roll_schedule = self._build_roll_schedule(contracts, all_data)
        return roll_schedule

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_roll_schedule(
        self,
        contracts: list[ContractInfo],
        all_data: dict[str, pd.DataFrame],
    ) -> pd.Series:
        """
        Determine which contract is "main" at each 1-min bar.

        Vectorized: build an OI matrix (timestamps × contracts), mask out
        contracts near expiry, then idxmax per row to pick the active contract.
        Ties broken by nearest expiry (columns sorted ascending by expiry).
        """
        expiry_map: dict[str, pd.Timestamp] = {
            c.contract_id: c.expiry_month for c in contracts
        }

        # OI matrix: index=all timestamps, columns=contract_ids, sorted by expiry
        sorted_cids = sorted(all_data.keys(), key=lambda c: expiry_map.get(c, pd.Timestamp.max))
        oi_frames = {cid: all_data[cid]["open_interest"] for cid in sorted_cids}
        oi_matrix = pd.DataFrame(oi_frames)  # NaN where contract has no bar

        # Mask contracts that are too close to (or past) expiry:
        # original condition: exp > ts - roll_days  ↔  ts < exp + roll_days
        for cid in sorted_cids:
            cutoff = expiry_map[cid] + pd.Timedelta(days=self.roll_days)
            oi_matrix.loc[oi_matrix.index >= cutoff, cid] = np.nan

        # idxmax picks leftmost (nearest-expiry) column on ties; NaN rows → NaN
        raw_active = oi_matrix.idxmax(axis=1)
        raw_active[oi_matrix.isna().all(axis=1)] = np.nan

        # Apply two-layer anti-flip filter:
        #   1. Minimum hold period: after switching, stay for at least
        #      roll_min_hold_days trading days before considering another switch.
        #   2. Hysteresis: new contract OI must exceed current by
        #      roll_hysteresis (default 20%) to trigger a switch.
        active = raw_active.copy()
        current = None
        last_switch_idx = -999999
        min_hold_bars = self.roll_min_hold_days * 240  # approx bars per trading day

        for i in range(len(active)):
            if pd.isna(raw_active.iloc[i]):
                current = None
                continue
            if current is None:
                current = raw_active.iloc[i]
                last_switch_idx = i
                continue
            if raw_active.iloc[i] == current:
                continue

            # Layer 1: minimum hold period
            bars_since_switch = i - last_switch_idx
            if bars_since_switch < min_hold_bars:
                active.iloc[i] = current
                continue

            # Layer 2: hysteresis check
            ts = active.index[i]
            oi_current = oi_matrix.loc[ts, current] if current in oi_matrix.columns else np.nan
            oi_new = oi_matrix.loc[ts, raw_active.iloc[i]] if raw_active.iloc[i] in oi_matrix.columns else np.nan
            if pd.notna(oi_current) and pd.notna(oi_new) and oi_current > 0:
                if (oi_new - oi_current) / oi_current >= self.roll_hysteresis:
                    current = raw_active.iloc[i]
                    last_switch_idx = i
                else:
                    active.iloc[i] = current
            else:
                current = raw_active.iloc[i]
                last_switch_idx = i

        return active.rename("contract")

    def _stitch(
        self,
        all_data: dict[str, pd.DataFrame],
        roll_schedule: pd.Series,
    ) -> pd.DataFrame:
        """Concatenate bars from the active contract at each timestamp."""
        pieces = []
        for cid, df in all_data.items():
            active_times = roll_schedule.index[roll_schedule == cid]
            valid = active_times[active_times.isin(df.index)]
            if valid.empty:
                continue
            chunk = df.loc[valid].copy()
            chunk["contract"] = cid
            pieces.append(chunk)

        if not pieces:
            return pd.DataFrame()

        result = pd.concat(pieces).sort_index()
        result.index.name = "datetime"
        return result

    def _adjust(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply back-adjustment or ratio-adjustment at roll points."""
        df = df.copy()
        price_cols = ["open", "high", "low", "close"]

        # Detect roll points (where contract changes)
        roll_mask = df["contract"] != df["contract"].shift(1)
        roll_times = df.index[roll_mask & (df.index > df.index[0])]

        # Work backwards from the most recent bar (no adjustment needed for latest)
        cumulative_offset = 0.0
        cumulative_ratio = 1.0

        roll_times_sorted = sorted(roll_times, reverse=True)

        for roll_ts in roll_times_sorted:
            # Price of new contract at roll open
            new_price = df.loc[roll_ts, "close"]
            # Price of old contract at the bar just before roll
            prev_loc = df.index.get_loc(roll_ts) - 1
            if prev_loc < 0:
                continue
            old_price = df.iloc[prev_loc]["close"]

            if self.adjust == "back":
                gap = new_price - old_price
                cumulative_offset += gap
                # Shift all bars before this roll
                mask = df.index < roll_ts
                df.loc[mask, price_cols] += gap

            elif self.adjust == "ratio":
                if old_price == 0:
                    continue
                ratio = new_price / old_price
                cumulative_ratio *= ratio
                mask = df.index < roll_ts
                df.loc[mask, price_cols] *= ratio

        return df
