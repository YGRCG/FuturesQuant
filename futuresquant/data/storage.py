"""
Parquet cache layer for 1-min K-line CSV files.

Strategy
--------
Each CSV file → one Parquet file under cache_dir/{PRODUCT}/{stem}.parquet
Cache is considered fresh when parquet mtime >= csv mtime.
Stale or missing cache entries are rebuilt transparently on first read.

Directory layout
----------------
cache/
└── parquet/
    ├── FU/
    │   ├── FU0503.parquet
    │   └── FU2505.parquet
    └── RB/
        └── RB2510.parquet
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from futuresquant.data.loader import ContractInfo, _COL_MAP

logger = logging.getLogger(__name__)


class ParquetCache:
    """
    Read-through Parquet cache for individual contract CSV files.

    Parameters
    ----------
    cache_dir : root directory for cached Parquet files
                (created automatically if it does not exist)
    """

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir) / "parquet"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        contract: ContractInfo,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """
        Return K-line DataFrame for one contract, using Parquet cache.
        Rebuilds the cache entry if the source CSV is newer.
        """
        parquet_path = self._parquet_path(contract)

        if not self._is_fresh(contract, parquet_path):
            self._build(contract, parquet_path)

        df = pd.read_parquet(parquet_path)

        if start is not None:
            df = df[df.index >= pd.Timestamp(start)]
        if end is not None:
            end_ts = pd.Timestamp(end)
            if end_ts.hour == 0 and end_ts.minute == 0 and end_ts.second == 0:
                end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            df = df[df.index <= end_ts]

        return df

    def warm_up(self, contracts: list[ContractInfo], force: bool = False) -> int:
        """
        Pre-build Parquet cache for a list of contracts.

        Parameters
        ----------
        force : rebuild even if cache is already fresh

        Returns
        -------
        Number of contracts actually rebuilt.
        """
        rebuilt = 0
        for c in contracts:
            p = self._parquet_path(c)
            if force or not self._is_fresh(c, p):
                self._build(c, p)
                rebuilt += 1
        logger.info("ParquetCache warm-up: %d / %d rebuilt", rebuilt, len(contracts))
        return rebuilt

    def invalidate(self, contract: ContractInfo) -> None:
        """Delete the cached Parquet for one contract (forces rebuild on next read)."""
        p = self._parquet_path(contract)
        if p.exists():
            p.unlink()
            logger.debug("Cache invalidated: %s", p)

    def cache_size_mb(self) -> float:
        """Total size of all cached Parquet files in MB."""
        total = sum(f.stat().st_size for f in self.cache_dir.rglob("*.parquet"))
        return total / 1024 / 1024

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _parquet_path(self, contract: ContractInfo) -> Path:
        return self.cache_dir / contract.product / f"{contract.contract_id}.parquet"

    def _is_fresh(self, contract: ContractInfo, parquet_path: Path) -> bool:
        if not parquet_path.exists():
            return False
        csv_mtime = contract.csv_path.stat().st_mtime
        parquet_mtime = parquet_path.stat().st_mtime
        return parquet_mtime >= csv_mtime

    def _build(self, contract: ContractInfo, parquet_path: Path) -> None:
        """Convert one CSV file to Parquet."""
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug("Building cache: %s → %s", contract.csv_path.name, parquet_path.name)

        df = pd.read_csv(
            contract.csv_path,
            encoding="utf-8-sig",
            parse_dates=["时间"],
        )
        df = df.rename(columns=_COL_MAP)
        df = df.set_index("datetime").sort_index()
        df.index.name = "datetime"

        # Cast to compact dtypes to reduce file size
        for col in ("open", "high", "low", "close", "amount"):
            if col in df.columns:
                df[col] = df[col].astype("float32")
        for col in ("volume", "open_interest"):
            if col in df.columns:
                df[col] = df[col].astype("int64")

        df.to_parquet(parquet_path, engine="pyarrow", compression="snappy")
        logger.debug("Cached %d rows → %s", len(df), parquet_path)
