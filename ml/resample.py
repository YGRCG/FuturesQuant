"""
Session-aware resampling for futures data.

期货交易时段不连续（有午休、夜盘间隔），pandas resample 按日历时间切窗口
会产生跨时段的 bar。本模块确保聚合 bar 不会跨越 session 边界。

FU 默认时段:
    0 = 夜盘  21:00–23:00
    1 = 早盘  09:00–10:15
    2 = 午前  10:30–11:30
    3 = 午后  13:30–15:00
"""

from __future__ import annotations

import datetime
import numpy as np
import pandas as pd


def _normalize_freq(freq: str) -> str:
    """将 freq 中的大写时间单位转为 pandas 2.x 要求的小写（'1H'->'1h' 等）。"""
    for old, new in [('min', 'min'), ('H', 'h'), ('D', 'D'), ('T', 'min')]:
        if freq.endswith(old):
            return freq[:-len(old)] + new
    return freq


# FU/SHFE 默认时段（与 SessionCode 因子一致）
DEFAULT_SESSIONS = [
    (datetime.time(21, 0),  datetime.time(23, 59)),  # 夜盘
    (datetime.time(9,  0),  datetime.time(10, 15)),  # 早盘
    (datetime.time(10, 30), datetime.time(11, 30)),  # 午前
    (datetime.time(13, 30), datetime.time(15,  0)),  # 午后
]


def _assign_session_id(index: pd.DatetimeIndex) -> pd.Series:
    """给每根 bar 分配 session ID（日期+时段编号），跨时段的 bar 不会共享 ID。"""
    times = pd.Series(index.time, index=index)
    dates = pd.Series(index.date, index=index)

    session_code = pd.Series(-1, index=index, dtype=int)
    for i, (start, end) in enumerate(DEFAULT_SESSIONS):
        if start <= end:
            mask = (times >= start) & (times <= end)
        else:
            mask = (times >= start) | (times <= end)
        session_code[mask] = i

    # 夜盘属于下一个交易日：21:00 之后的 date 映射到 date+1
    night_mask = session_code == 0
    trade_dates = dates.copy()
    trade_dates[night_mask] = dates[night_mask] + pd.Timedelta(days=1)

    return trade_dates.astype(str) + '_S' + session_code.astype(str)


def session_resample_last(
    df: pd.DataFrame | pd.Series,
    freq: str,
) -> pd.DataFrame | pd.Series:
    """
    Session-aware resample，取每个窗口最后一个值。
    不会让任何一根聚合 bar 跨越 session 边界。

    对于 '1D' 频率，直接用标准 resample（日级别没有跨时段问题）。
    对于 '1min' 频率，不做聚合直接返回。
    """
    freq = _normalize_freq(freq)
    if freq == '1min':
        return df
    if freq in ('1D', '1d'):
        return df.resample(freq).last()

    sid = _assign_session_id(df.index)

    results = []
    for session_id, group in df.groupby(sid):
        if len(group) == 0:
            continue
        resampled = group.resample(freq).last().dropna(how='all')
        results.append(resampled)

    if not results:
        return df.iloc[:0]
    return pd.concat(results).sort_index()


def session_resample_ohlc(
    klines: pd.DataFrame,
    freq: str,
) -> pd.DataFrame:
    """
    Session-aware OHLC resample。
    返回 DataFrame 含 open/high/low/close 列。
    """
    freq = _normalize_freq(freq)
    if freq == '1min':
        return klines[['open', 'high', 'low', 'close']].copy()
    if freq in ('1D', '1d'):
        return klines[['open', 'high', 'low', 'close']].resample(freq).agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
        }).dropna(subset=['close'])

    sid = _assign_session_id(klines.index)

    results = []
    for session_id, group in klines.groupby(sid):
        if len(group) == 0:
            continue
        resampled = group[['open', 'high', 'low', 'close']].resample(freq).agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
        }).dropna(subset=['close'])
        results.append(resampled)

    if not results:
        return klines[['open', 'high', 'low', 'close']].iloc[:0]
    return pd.concat(results).sort_index()
