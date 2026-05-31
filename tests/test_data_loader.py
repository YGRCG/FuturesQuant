"""Basic smoke tests for the CSV data loader — no tqsdk credentials needed."""

import pytest
import pandas as pd
from futuresquant.data.loader import FuturesDataLoader, DataRegistry, ContractInfo
from futuresquant.data.universe import ContinuousContract

RAW_DATA_DIR = r"I:\stock\FuturesQuant\raw_data"
DATA_DIR = rf"{RAW_DATA_DIR}\1min_FU"
CACHE_DIR = r"I:\stock\FuturesQuant\cache"


@pytest.fixture(scope="module")
def loader():
    return FuturesDataLoader(DATA_DIR)


def test_scan_finds_fu_contracts(loader):
    contracts = loader.list_contracts("FU")
    assert len(contracts) > 200
    assert all(isinstance(c, ContractInfo) for c in contracts)
    assert all(c.product == "FU" for c in contracts)


def test_contract_id_and_symbol(loader):
    contracts = loader.list_contracts("FU")
    first = contracts[0]
    assert first.contract_id == "FU0503"
    assert first.tqsdk_symbol == "SHFE.fu0503"


def test_load_single_contract(loader):
    df = loader.load("FU0503")
    assert set(df.columns) >= {"open", "high", "low", "close", "volume", "amount", "open_interest"}
    assert isinstance(df.index, pd.DatetimeIndex)
    assert not df.empty
    assert df["close"].isna().sum() == 0


def test_load_with_date_filter(loader):
    df = loader.load("FU0503", start="2005-01-04", end="2005-01-04")
    assert df.index.min().date() == pd.Timestamp("2005-01-04").date()
    assert df.index.max().date() == pd.Timestamp("2005-01-04").date()


def test_continuous_contract_back_adjust(loader):
    cc = ContinuousContract(loader, "FU", adjust="back")
    df = cc.build(start="2020-01-01", end="2021-12-31")
    assert not df.empty
    assert "contract" in df.columns
    # Back-adjusted: no sudden price jumps > 20% at roll points
    roll_mask = df["contract"] != df["contract"].shift(1)
    roll_times = df.index[roll_mask & (df.index > df.index[0])]
    for rt in roll_times:
        loc = df.index.get_loc(rt)
        prev_close = df.iloc[loc - 1]["close"]
        curr_close = df.iloc[loc]["close"]
        if prev_close > 0:
            assert abs(curr_close / prev_close - 1) < 0.20, (
                f"Large gap at roll {rt}: {prev_close} → {curr_close}"
            )


# ------------------------------------------------------------------
# DataRegistry
# ------------------------------------------------------------------

def test_registry_discovers_fu():
    registry = DataRegistry(RAW_DATA_DIR)
    products = registry.list_products()
    assert "FU" in products


def test_registry_get_loader():
    registry = DataRegistry(RAW_DATA_DIR)
    loader = registry.get_loader("FU")
    assert isinstance(loader, FuturesDataLoader)
    assert len(loader.list_contracts()) > 200


def test_registry_load_by_contract_id():
    registry = DataRegistry(RAW_DATA_DIR)
    df = registry.load("FU2210", start="2022-06-01", end="2022-06-30")
    assert not df.empty
    assert df.index.min().date() >= pd.Timestamp("2022-06-01").date()


def test_registry_unknown_product_raises():
    registry = DataRegistry(RAW_DATA_DIR)
    with pytest.raises(KeyError, match="UNKNOWN"):
        registry.get_loader("UNKNOWN")


# ------------------------------------------------------------------
# Parquet cache
# ------------------------------------------------------------------

def test_parquet_cache_loads_same_data(tmp_path):
    """Cached load must return identical data to direct CSV load."""
    loader_csv = FuturesDataLoader(DATA_DIR)
    loader_cached = FuturesDataLoader(DATA_DIR, cache_dir=tmp_path)

    df_csv = loader_csv.load("FU2210")
    df_cached = loader_cached.load("FU2210")

    # Values should match; dtype may differ (float32 vs float64)
    pd.testing.assert_index_equal(df_csv.index, df_cached.index)
    for col in ("open", "high", "low", "close"):
        assert (df_csv[col].values - df_cached[col].values.astype("float64") < 0.01).all()


def test_parquet_cache_is_faster(tmp_path):
    """Second (cached) load should be faster than the first CSV load."""
    import time
    loader = FuturesDataLoader(DATA_DIR, cache_dir=tmp_path)

    t0 = time.perf_counter()
    loader.load("FU2210")   # first call: CSV → Parquet
    t_first = time.perf_counter() - t0

    t0 = time.perf_counter()
    loader.load("FU2210")   # second call: Parquet read
    t_cached = time.perf_counter() - t0

    assert t_cached < t_first, (
        f"Cached read ({t_cached:.3f}s) should be faster than CSV ({t_first:.3f}s)"
    )


def test_warm_up_cache(tmp_path):
    loader = FuturesDataLoader(DATA_DIR, cache_dir=tmp_path)
    # Warm up only the last 3 contracts
    contracts = loader.list_contracts()[-3:]
    from futuresquant.data.storage import ParquetCache
    cache = ParquetCache(tmp_path)
    rebuilt = cache.warm_up(contracts)
    assert rebuilt == 3
    # Second warm-up should rebuild nothing (all fresh)
    rebuilt2 = cache.warm_up(contracts)
    assert rebuilt2 == 0
