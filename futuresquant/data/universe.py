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
    ):
        self.loader = loader
        self.product = product.upper()
        self.adjust = adjust
        self.roll_days = roll_n_days_before_expiry

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

        Simple rule: among contracts whose expiry_month is strictly
        after the current bar's date (minus roll_days), pick the one
        with the highest open_interest.  Ties broken by nearest expiry.
        """
        # Collect all timestamps across all contracts
        all_times = pd.DatetimeIndex(
            sorted(set().union(*[df.index for df in all_data.values()]))
        )

        expiry_map: dict[str, pd.Timestamp] = {
            c.contract_id: c.expiry_month for c in contracts
        }

        active: list[str] = []
        for ts in all_times:
            candidates = [
                cid for cid, exp in expiry_map.items()
                if cid in all_data
                and exp > ts - pd.Timedelta(days=self.roll_days)
                and ts in all_data[cid].index
            ]
            if not candidates:
                active.append(np.nan)
                continue
            # Pick highest open_interest; break ties by nearest expiry
            def sort_key(cid: str) -> tuple:
                oi = all_data[cid].loc[ts, "open_interest"] if ts in all_data[cid].index else -1
                exp = expiry_map[cid]
                return (-oi, exp)

            best = min(candidates, key=sort_key)
            active.append(best)

        return pd.Series(active, index=all_times, name="contract")

    def _stitch(
        self,
        all_data: dict[str, pd.DataFrame],
        roll_schedule: pd.Series,
    ) -> pd.DataFrame:
        """Concatenate bars from the active contract at each timestamp."""
        rows = []
        for ts, cid in roll_schedule.items():
            if pd.isna(cid) or cid not in all_data:
                continue
            df = all_data[cid]
            if ts not in df.index:
                continue
            row = df.loc[ts].copy()
            row["contract"] = cid
            rows.append(row)

        result = pd.DataFrame(rows)
        result.index = roll_schedule.dropna().index[:len(result)]
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
