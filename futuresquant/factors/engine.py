"""
FactorEngine — batch-compute multiple factors over one or more contracts.

Usage
-----
    from futuresquant.factors.engine import FactorEngine
    from futuresquant.factors.technical.momentum import ROC, RSI
    from futuresquant.factors.technical.volatility import NormATR

    engine = FactorEngine([ROC(20), RSI(14), NormATR(14)])
    factor_df = engine.compute(klines)          # single contract → wide DataFrame
    panel = engine.compute_panel(klines_dict)   # {contract_id: klines} → MultiIndex
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from futuresquant.factors.base import Factor, zscore, rank_normalize

NormMethod = Literal["none", "zscore", "rank"]


class FactorEngine:
    """Compute and optionally normalise a list of factors."""

    def __init__(
        self,
        factors: list[Factor],
        norm: NormMethod = "none",
        norm_window: int = 240,
    ):
        """
        Parameters
        ----------
        factors     : list of Factor instances to evaluate
        norm        : normalisation applied to each factor time-series after compute
                      "none"   → raw values
                      "zscore" → rolling z-score (window = norm_window)
                      "rank"   → rolling percentile rank in [0, 1]
        norm_window : lookback window for normalisation (in bars)
        """
        self.factors = factors
        self.norm = norm
        self.norm_window = norm_window

    # ------------------------------------------------------------------
    # Single-contract
    # ------------------------------------------------------------------

    def compute(self, klines: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all factors for one contract.

        Returns
        -------
        DataFrame : index = DatetimeIndex, columns = factor names
        """
        series = {}
        for f in self.factors:
            s = f.compute(klines)
            s = self._normalise(s)
            series[f.name] = s
        return pd.DataFrame(series, index=klines.index)

    # ------------------------------------------------------------------
    # Multi-contract panel
    # ------------------------------------------------------------------

    def compute_panel(
        self,
        klines_dict: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Compute factors for multiple contracts.

        Returns
        -------
        DataFrame with MultiIndex (contract, datetime) and factor columns.
        """
        frames = []
        for contract_id, klines in klines_dict.items():
            df = self.compute(klines)
            df.index = pd.MultiIndex.from_arrays(
                [[contract_id] * len(df), df.index],
                names=["contract", "datetime"],
            )
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames).sort_index()

    # ------------------------------------------------------------------
    # Cross-sectional normalisation (across contracts at each timestamp)
    # ------------------------------------------------------------------

    @staticmethod
    def cross_section_zscore(panel: pd.DataFrame) -> pd.DataFrame:
        """
        Z-score each factor cross-sectionally at each timestamp.
        Input: MultiIndex (contract, datetime) DataFrame from compute_panel().
        """
        return panel.groupby(level="datetime").transform(
            lambda x: (x - x.mean()) / x.std(ddof=1)
        )

    @staticmethod
    def cross_section_rank(panel: pd.DataFrame) -> pd.DataFrame:
        """Percentile rank across contracts at each timestamp."""
        return panel.groupby(level="datetime").transform(
            lambda x: x.rank(pct=True)
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _normalise(self, s: pd.Series) -> pd.Series:
        if self.norm == "zscore":
            return zscore(s, self.norm_window)
        if self.norm == "rank":
            return rank_normalize(s, self.norm_window)
        return s
