"""Factor correctness tests — no tqsdk credentials needed."""

import numpy as np
import pandas as pd
import pytest

from futuresquant.data.loader import FuturesDataLoader
from futuresquant.factors.engine import FactorEngine
from futuresquant.factors.technical import (
    ROC, MOM, RSI, BollingerBand, TSMomentum,
    MACross, MACD, ADX, PriceChannel,
    ATR, NormATR, HistoricalVolatility, VolatilityRatio,
    VolumeRatio, OBV, VWAP, OpenInterestChange,
)

DATA_DIR = r"I:\stock\FuturesQuant\raw_data\1min_FU"
ALL_FACTORS = [
    ROC(20), MOM(20), RSI(14), BollingerBand(20), TSMomentum(5, 20),
    MACross(5, 20), MACD(12, 26, 9), ADX(14), PriceChannel(20),
    ATR(14), NormATR(14), HistoricalVolatility(240), VolatilityRatio(20, 120),
    VolumeRatio(20), OBV(20), VWAP(60, 14), OpenInterestChange(20),
]


@pytest.fixture(scope="module")
def klines():
    loader = FuturesDataLoader(DATA_DIR)
    return loader.load("FU2210", start="2022-01-01", end="2022-10-31")


# ------------------------------------------------------------------
# Basic contract: all factors produce a Series aligned to input index
# ------------------------------------------------------------------

@pytest.mark.parametrize("factor", ALL_FACTORS, ids=lambda f: f.name)
def test_output_shape_and_index(factor, klines):
    result = factor.compute(klines)
    assert isinstance(result, pd.Series)
    assert len(result) == len(klines)
    assert result.index.equals(klines.index)
    assert result.name == factor.name


@pytest.mark.parametrize("factor", ALL_FACTORS, ids=lambda f: f.name)
def test_no_all_nan(factor, klines):
    result = factor.compute(klines)
    assert result.notna().any(), f"{factor.name} returned all NaN"


@pytest.mark.parametrize("factor", ALL_FACTORS, ids=lambda f: f.name)
def test_warmup_nans_at_start(factor, klines):
    """First bar should be NaN for any factor that needs history."""
    result = factor.compute(klines)
    assert pd.isna(result.iloc[0]), f"{factor.name}: expected NaN at bar 0"


# ------------------------------------------------------------------
# Specific value checks
# ------------------------------------------------------------------

def test_roc_zero_when_price_unchanged(klines):
    flat = klines.copy()
    flat["close"] = 100.0
    result = ROC(5).compute(flat)
    assert (result.dropna() == 0).all()


def test_rsi_bounds(klines):
    result = RSI(14).compute(klines).dropna()
    assert (result >= 0).all() and (result <= 100).all()


def test_bollinger_mostly_in_range(klines):
    result = BollingerBand(20).compute(klines).dropna()
    # Most values should be within a wide band; extreme outliers are ok
    assert (result.between(-1, 2)).mean() > 0.90


def test_atr_positive(klines):
    result = ATR(14).compute(klines).dropna()
    assert (result > 0).all()


def test_norm_atr_small(klines):
    result = NormATR(14).compute(klines).dropna()
    assert (result > 0).all()
    assert result.median() < 0.05  # typically < 5% for futures


def test_price_channel_bounds(klines):
    result = PriceChannel(20).compute(klines).dropna()
    assert result.between(0, 1).mean() > 0.99


def test_adx_positive(klines):
    result = ADX(14).compute(klines).dropna()
    assert (result >= 0).all()


# ------------------------------------------------------------------
# Factor composition
# ------------------------------------------------------------------

def test_factor_arithmetic(klines):
    combined = ROC(20) - MOM(20)
    result = combined.compute(klines)
    assert isinstance(result, pd.Series)
    assert "ROC_20" in combined.name
    assert "MOM_20" in combined.name


def test_factor_negation(klines):
    neg = -ROC(20)
    pos = ROC(20).compute(klines)
    result = neg.compute(klines)
    pd.testing.assert_series_equal(result, -pos, check_names=False)


# ------------------------------------------------------------------
# FactorEngine
# ------------------------------------------------------------------

def test_engine_single_contract(klines):
    engine = FactorEngine([ROC(20), RSI(14), ATR(14)])
    df = engine.compute(klines)
    assert set(df.columns) == {"ROC_20", "RSI_14", "ATR_14"}
    assert len(df) == len(klines)


def test_engine_zscore_norm(klines):
    engine = FactorEngine([ROC(20)], norm="zscore", norm_window=120)
    df = engine.compute(klines)
    vals = df["ROC_20"].dropna()
    assert abs(vals.mean()) < 0.5   # roughly centred
    assert abs(vals.std() - 1.0) < 0.3


def test_engine_panel(klines):
    loader = FuturesDataLoader(DATA_DIR)
    klines_dict = {
        "FU2209": loader.load("FU2209", start="2022-01-01", end="2022-09-30"),
        "FU2210": klines,
    }
    engine = FactorEngine([ROC(20), ATR(14)])
    panel = engine.compute_panel(klines_dict)
    assert panel.index.names == ["contract", "datetime"]
    assert set(panel.columns) == {"ROC_20", "ATR_14"}
    assert "FU2209" in panel.index.get_level_values("contract")
    assert "FU2210" in panel.index.get_level_values("contract")
