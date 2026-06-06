"""
标签构建

调用方式:
    from ml.labels import build_labels
    y = build_labels(klines, cfg['labels'], freq='1D')

支持的 label_type:
    regression      — 未来 forward_bars 根 bar 的收益率
    classification  — 三分类 (-1/0/1)，基于固定阈值
    triple_barrier  — 三重障碍标签 (-1/0/1)，基于 ATR 自适应止盈止损
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from ml.resample import session_resample_last, session_resample_ohlc


# ---------------------------------------------------------------------------
# Triple Barrier 核心（numba 加速）
# ---------------------------------------------------------------------------

@njit
def _triple_barrier_scan(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    max_bars: int,
) -> np.ndarray:
    """
    逐 bar 向前扫描，判断先触碰哪个障碍。

    Parameters
    ----------
    close : 收盘价序列
    high  : 最高价序列
    low   : 最低价序列
    upper : 每根 bar 的止盈价位（绝对价格）
    lower : 每根 bar 的止损价位（绝对价格）
    max_bars : 时间障碍（最长持仓 bar 数）

    Returns
    -------
    labels : +1（触止盈）/ -1（触止损）/ 0（超时）/ NaN（数据不足）
    """
    n = len(close)
    labels = np.empty(n, dtype=np.float64)
    labels[:] = np.nan

    for i in range(n):
        if np.isnan(upper[i]) or np.isnan(lower[i]):
            continue

        entry = close[i]
        tp = upper[i]
        sl = lower[i]
        end = min(i + max_bars, n)

        if i + 1 >= n:
            continue

        hit = 0
        for j in range(i + 1, end):
            if high[j] >= tp:
                hit = 1
                break
            if low[j] <= sl:
                hit = -1
                break

        if hit != 0:
            labels[i] = hit
        elif end <= n and end > i + 1:
            ret = close[min(end, n) - 1] / entry - 1.0
            if ret > 0:
                labels[i] = 1
            elif ret < 0:
                labels[i] = -1
            else:
                labels[i] = 0

    return labels


def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                 period: int = 14) -> pd.Series:
    """计算 ATR，不依赖因子系统。"""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, min_periods=period).mean()


def _build_triple_barrier(
    klines: pd.DataFrame,
    cfg: dict,
    freq: str,
) -> pd.Series:
    """
    构建 Triple Barrier 标签。

    cfg 可选参数:
        forward_bars   : 时间障碍（最长持仓 bar 数），默认 20
        atr_period     : ATR 计算周期，默认 14
        atr_multiplier : 止盈止损 = close ± multiplier * ATR，默认 1.5
    """
    forward_bars = cfg.get('forward_bars', cfg.get('forward_days', 20))
    atr_period = cfg.get('atr_period', 14)
    atr_mult = cfg.get('atr_multiplier', 1.5)

    ohlc = session_resample_ohlc(klines, freq)

    atr = _compute_atr(ohlc['high'], ohlc['low'], ohlc['close'], atr_period)
    upper_price = ohlc['close'] + atr_mult * atr
    lower_price = ohlc['close'] - atr_mult * atr

    labels = _triple_barrier_scan(
        ohlc['close'].values,
        ohlc['high'].values,
        ohlc['low'].values,
        upper_price.values,
        lower_price.values,
        forward_bars,
    )

    return pd.Series(labels, index=ohlc.index, name='label')


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------

def build_labels(klines: pd.DataFrame, cfg: dict, freq: str = '1D') -> pd.Series:
    """
    构建预测标签。

    Parameters
    ----------
    klines : K 线 DataFrame（需含 close 列；triple_barrier 还需 high/low）
    cfg    : config.yaml 中的 labels 段
    freq   : 聚合频率（与 build_features 保持一致）

    Returns
    -------
    pd.Series  index=DatetimeIndex（按 freq 聚合），name='label'
               regression      : 未来 forward_bars 根 bar 的收益率
               classification  : -1 / 0 / 1（固定阈值三分类）
               triple_barrier  : -1 / 0 / 1（触止损 / 超时 / 触止盈）
    """
    label_type = cfg.get('label_type', 'regression')

    if label_type == 'triple_barrier':
        return _build_triple_barrier(klines, cfg, freq)

    forward_bars = cfg.get('forward_bars', cfg.get('forward_days', 1))

    close = session_resample_last(klines[['close']], freq)['close']

    forward_ret = close.pct_change(forward_bars).shift(-forward_bars)

    if label_type == 'regression':
        return forward_ret.rename('label')

    if label_type == 'classification':
        thr = cfg.get('threshold', 0.003)
        y = pd.cut(
            forward_ret,
            bins=[-np.inf, -thr, thr, np.inf],
            labels=[-1, 0, 1],
        ).astype(float).rename('label')
        return y

    raise ValueError(
        f"label_type 必须为 'regression'/'classification'/'triple_barrier'，"
        f"收到: {label_type!r}"
    )
