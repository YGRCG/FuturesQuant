"""
CSV data loader for local futures K-line files.

File naming convention : {PRODUCT}{YYMM}.csv   e.g. FU0503.csv
Directory convention   : raw_data/1min_{PRODUCT}/  e.g. raw_data/1min_FU/

Two entry points
----------------
FuturesDataLoader(data_dir, cache_dir=None)
    Load a single product directory (old API, still valid).
    If cache_dir is given, reads/writes Parquet cache transparently.

DataRegistry(raw_data_dir, cache_dir=None)
    Discover all 1min_* subdirectories automatically.
    registry.list_products()        → ["FU", "RB", "CU", …]
    registry.get_loader("FU")       → FuturesDataLoader
    registry.warm_up_cache()        → pre-build all Parquet files
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Column mapping: Chinese CSV headers → internal English names
# ---------------------------------------------------------------------------
_COL_MAP = {
    "时间": "datetime",
    "开盘价": "open",
    "最高价": "high",
    "最低价": "low",
    "收盘价": "close",
    "成交量": "volume",
    "成交额": "amount",
    "持仓量": "open_interest",
}

_CONTRACT_RE = re.compile(r"^([A-Za-z]+)(\d{2})(\d{2})$")
_PRODUCT_DIR_RE = re.compile(r"^1min_([A-Za-z]+)$", re.IGNORECASE)


@dataclass(frozen=True)
class ContractInfo:
    product: str        # uppercase, e.g. "FU"
    year: int           # full 4-digit year, e.g. 2005
    month: int          # 1–12
    csv_path: Path

    @property
    def contract_id(self) -> str:
        return f"{self.product}{self.year % 100:02d}{self.month:02d}"

    @property
    def tqsdk_symbol(self) -> str:
        exchange = _PRODUCT_EXCHANGE.get(self.product.upper(), "SHFE")
        return f"{exchange}.{self.product.lower()}{self.year % 100:02d}{self.month:02d}"

    @property
    def expiry_month(self) -> pd.Timestamp:
        return pd.Timestamp(year=self.year, month=self.month, day=1)


_PRODUCT_EXCHANGE: dict[str, str] = {
    "FU": "SHFE", "RB": "SHFE", "HC": "SHFE", "CU": "SHFE",
    "AL": "SHFE", "ZN": "SHFE", "NI": "SHFE", "SN": "SHFE",
    "AU": "SHFE", "AG": "SHFE", "BU": "SHFE", "RU": "SHFE", "SP": "SHFE",
    "M":  "DCE",  "Y":  "DCE",  "P":  "DCE",  "A":  "DCE",
    "B":  "DCE",  "C":  "DCE",  "CS": "DCE",  "L":  "DCE",
    "V":  "DCE",  "J":  "DCE",  "JM": "DCE",  "I":  "DCE",  "EG": "DCE",
    "IF": "CFFEX","IC": "CFFEX","IH": "CFFEX","IM": "CFFEX",
    "T":  "CFFEX","TF": "CFFEX","TS": "CFFEX",
    "SR": "CZCE", "CF": "CZCE", "TA": "CZCE", "MA": "CZCE",
    "OI": "CZCE", "RM": "CZCE", "ZC": "CZCE", "AP": "CZCE", "CJ": "CZCE",
}


def _parse_contract_name(stem: str) -> Optional[ContractInfo]:
    m = _CONTRACT_RE.match(stem)
    if not m:
        return None
    product, yy, mm = m.group(1).upper(), int(m.group(2)), int(m.group(3))
    year = 2000 + yy if yy <= 30 else 1900 + yy
    if not (1 <= mm <= 12):
        return None
    return ContractInfo(product=product, year=year, month=mm, csv_path=Path())


# ---------------------------------------------------------------------------
# FuturesDataLoader — single-product directory
# ---------------------------------------------------------------------------

class FuturesDataLoader:
    """Load 1-minute K-line files from one product directory."""

    def __init__(
        self,
        data_dir: str | Path,
        cache_dir: str | Path | None = None,
    ):
        """
        Parameters
        ----------
        data_dir  : directory containing CSV files, e.g. raw_data/1min_FU
        cache_dir : root for Parquet cache (e.g. cache/).
                    If None, reads CSV every time.
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        self._cache = None
        if cache_dir is not None:
            from futuresquant.data.storage import ParquetCache
            self._cache = ParquetCache(cache_dir)

        self._contracts: list[ContractInfo] | None = None

    # ------------------------------------------------------------------
    # Contract discovery
    # ------------------------------------------------------------------

    def list_contracts(self, product: str | None = None) -> list[ContractInfo]:
        if self._contracts is None:
            self._contracts = self._scan_directory()
        if product is None:
            return self._contracts
        return [c for c in self._contracts if c.product == product.upper()]

    def _scan_directory(self) -> list[ContractInfo]:
        contracts = []
        for csv_file in sorted(self.data_dir.glob("*.csv")):
            info = _parse_contract_name(csv_file.stem)
            if info is None:
                continue
            info = ContractInfo(
                product=info.product, year=info.year,
                month=info.month, csv_path=csv_file,
            )
            contracts.append(info)
        return sorted(contracts, key=lambda c: (c.product, c.year, c.month))

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load(
        self,
        contract: ContractInfo | str,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """
        Load 1-min K-lines for one contract.
        Uses Parquet cache when cache_dir was provided at construction.
        """
        if isinstance(contract, str):
            contract = self._find_contract(contract)

        if self._cache is not None:
            return self._cache.load(contract, start=start, end=end)

        return self._load_csv(contract, start, end)

    def load_product(
        self,
        product: str,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Load all contracts for a product; returns {contract_id: DataFrame}."""
        return {
            c.contract_id: self.load(c, start=start, end=end)
            for c in self.list_contracts(product)
        }

    def warm_up_cache(self, force: bool = False) -> int:
        """Pre-build Parquet cache for all contracts in this directory."""
        if self._cache is None:
            raise RuntimeError("cache_dir was not set — pass cache_dir to FuturesDataLoader()")
        return self._cache.warm_up(self.list_contracts(), force=force)

    def _load_csv(
        self,
        contract: ContractInfo,
        start: str | pd.Timestamp | None,
        end: str | pd.Timestamp | None,
    ) -> pd.DataFrame:
        df = pd.read_csv(
            contract.csv_path,
            encoding="utf-8-sig",
            parse_dates=["时间"],
        )
        df = df.rename(columns=_COL_MAP)
        df = df.set_index("datetime").sort_index()
        df.index.name = "datetime"

        if start is not None:
            df = df[df.index >= pd.Timestamp(start)]
        if end is not None:
            end_ts = pd.Timestamp(end)
            if end_ts.hour == 0 and end_ts.minute == 0 and end_ts.second == 0:
                end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            df = df[df.index <= end_ts]

        return df

    def _find_contract(self, stem: str) -> ContractInfo:
        for c in self.list_contracts():
            if c.contract_id.upper() == stem.upper():
                return c
        raise KeyError(f"Contract '{stem}' not found in {self.data_dir}")


# ---------------------------------------------------------------------------
# DataRegistry — multi-product root directory
# ---------------------------------------------------------------------------

class DataRegistry:
    """
    Discover and manage all 1min_* product directories under a root.

    Expected layout::

        raw_data/
        ├── 1min_FU/   ← product FU
        ├── 1min_RB/   ← product RB
        └── 1min_CU/   ← product CU

    Usage::

        registry = DataRegistry("raw_data", cache_dir="cache")
        registry.list_products()         # ["CU", "FU", "RB"]
        loader = registry.get_loader("FU")
        df = loader.load("FU2210")
        registry.warm_up_cache()         # pre-build all Parquet files
    """

    def __init__(
        self,
        raw_data_dir: str | Path,
        cache_dir: str | Path | None = None,
    ):
        self.raw_data_dir = Path(raw_data_dir)
        if not self.raw_data_dir.exists():
            raise FileNotFoundError(f"raw_data directory not found: {self.raw_data_dir}")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._loaders: dict[str, FuturesDataLoader] = {}
        self._product_dirs: dict[str, Path] = self._discover()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover(self) -> dict[str, Path]:
        """Scan raw_data_dir for 1min_* subdirectories."""
        result: dict[str, Path] = {}
        for subdir in sorted(self.raw_data_dir.iterdir()):
            if not subdir.is_dir():
                continue
            m = _PRODUCT_DIR_RE.match(subdir.name)
            if m:
                product = m.group(1).upper()
                result[product] = subdir
        return result

    def list_products(self) -> list[str]:
        """Return sorted list of all discovered product codes."""
        return sorted(self._product_dirs.keys())

    # ------------------------------------------------------------------
    # Loader access
    # ------------------------------------------------------------------

    def get_loader(self, product: str) -> FuturesDataLoader:
        """Return a (cached) FuturesDataLoader for one product."""
        product = product.upper()
        if product not in self._product_dirs:
            available = ", ".join(self.list_products())
            raise KeyError(f"Product '{product}' not found. Available: {available}")
        if product not in self._loaders:
            self._loaders[product] = FuturesDataLoader(
                self._product_dirs[product],
                cache_dir=self.cache_dir,
            )
        return self._loaders[product]

    def list_contracts(self, product: str) -> list[ContractInfo]:
        return self.get_loader(product).list_contracts()

    def load(
        self,
        contract_id: str,
        product: str | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """
        Load one contract by ID, optionally specifying product for speed.
        If product is None, searches all discovered products.
        """
        if product:
            return self.get_loader(product).load(contract_id, start=start, end=end)
        # Infer product from contract_id prefix
        m = _CONTRACT_RE.match(contract_id)
        if m:
            inferred = m.group(1).upper()
            if inferred in self._product_dirs:
                return self.get_loader(inferred).load(contract_id, start=start, end=end)
        raise KeyError(f"Cannot resolve product for contract '{contract_id}'")

    def warm_up_cache(self, products: list[str] | None = None, force: bool = False) -> dict[str, int]:
        """
        Pre-build Parquet cache for all (or specified) products.

        Returns
        -------
        {product: n_rebuilt}
        """
        if self.cache_dir is None:
            raise RuntimeError("cache_dir was not set — pass cache_dir to DataRegistry()")
        targets = [p.upper() for p in products] if products else self.list_products()
        result = {}
        for product in targets:
            loader = self.get_loader(product)
            result[product] = loader.warm_up_cache(force=force)
        return result
