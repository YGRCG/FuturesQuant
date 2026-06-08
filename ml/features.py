"""
特征构建流水线

调用方式:
    from ml.features import build_features
    X = build_features(klines, klines_clean, cfg['features'], freq='1D')
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.resample import session_resample_last
from futuresquant.factors.engine import FactorEngine
from futuresquant.factors.technical import (
    ROC, MOM, RSI, BollingerBand, TSMomentum, MA,
    MACross, MACD, ADX, PriceChannel,
    ATR, NormATR, HistoricalVolatility, VolatilityRatio,
    VolumeRatio, OBV, VWAP, OpenInterestChange,
    DayOfWeek, MinuteOfDay, SessionCode, DaysToExpiry, Lagged,
)


def _default_factors() -> list:
    base = [
        MA(5), MA(20), MA(60),
        ROC(20), MOM(20), RSI(14), BollingerBand(20), TSMomentum(5, 20),
        MACross(5, 20), MACD(12, 26, 9), ADX(14), PriceChannel(20),
        ATR(8), NormATR(8), ATR(14), NormATR(14), HistoricalVolatility(240), VolatilityRatio(20, 120),
        VolumeRatio(20), OBV(20), VWAP(60, 14), OpenInterestChange(20),
        DayOfWeek(), MinuteOfDay(), SessionCode(), DaysToExpiry(),
    ]
    return base


def build_features(
    klines: pd.DataFrame,
    klines_clean: pd.DataFrame,
    cfg: dict,
    factors: list | None = None,
    freq: str = '1D',
) -> pd.DataFrame:
    """
    构建 ML 特征矩阵。

    Parameters
    ----------
    klines       : 原始（后复权）连续 K 线，用于时间索引
    klines_clean : 换月节点 volume/open_interest 已置 NaN 的 K 线
    cfg          : config.yaml 中的 features 段
    factors      : 因子列表，None 则使用默认全集
    freq         : 聚合频率（'1min'=不聚合，'5min'/'15min'/'30min'/'1H'/'4H'/'1D'）

    Returns
    -------
    pd.DataFrame  index=DatetimeIndex（按 freq 聚合），columns=特征名
    """
    if factors is None:
        factors = _default_factors()

    # 1. 计算因子（分钟级）
    engine = FactorEngine(factors)
    factor_df = engine.compute(klines_clean)

    # 2. 按 freq 聚合（session-aware，不跨交易时段）
    agg_f = session_resample_last(factor_df, freq)

    # 类别/时间特征不参与滞后和滚动统计
    _cat_cols = ['DayOfWeek', 'SessionCode', 'MinuteOfDay', 'DaysToExpiry']
    _num_cols = [c for c in agg_f.columns if c not in _cat_cols]
    agg_f_num = agg_f[_num_cols]

    frames = [agg_f]

    # 3. 滞后特征（仅数值特征）
    for lag in cfg.get('lags', [1, 5, 20]):
        frames.append(agg_f_num.shift(lag).add_suffix(f'_lag{lag}'))

    # 4. 滚动统计特征（仅数值特征）
    for w in cfg.get('rolling_windows', [20]):
        frames.append(agg_f_num.rolling(w).mean().add_suffix(f'_rmean{w}'))
        frames.append(agg_f_num.rolling(w).std().add_suffix(f'_rstd{w}'))

    X = pd.concat(frames, axis=1)
    X = X.replace([np.inf, -np.inf], np.nan)
    return X


def get_feature_names(cfg: dict, factors: list | None = None, freq: str = '1D') -> list[str]:
    """返回 build_features 会产生的所有列名（不实际计算）。"""
    if factors is None:
        factors = _default_factors()
    base_names = [f.name for f in factors]
    _cat_cols = {'DayOfWeek', 'SessionCode', 'MinuteOfDay', 'DaysToExpiry'}
    num_names = [n for n in base_names if n not in _cat_cols]
    names = list(base_names)
    for lag in cfg.get('lags', [1, 5, 20]):
        names += [f'{n}_lag{lag}' for n in num_names]
    for w in cfg.get('rolling_windows', [20]):
        names += [f'{n}_rmean{w}' for n in num_names]
        names += [f'{n}_rstd{w}' for n in num_names]
    return names
