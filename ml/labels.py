"""
标签构建

调用方式:
    from ml.labels import build_labels
    y = build_labels(klines, cfg['labels'], freq='1D')
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_labels(klines: pd.DataFrame, cfg: dict, freq: str = '1D') -> pd.Series:
    """
    构建预测标签。

    Parameters
    ----------
    klines : K 线 DataFrame（需含 close 列）
    cfg    : config.yaml 中的 labels 段
    freq   : 聚合频率（与 build_features 保持一致）

    Returns
    -------
    pd.Series  index=DatetimeIndex（按 freq 聚合），name='label'
               regression   : 未来 forward_bars 根 bar 的收益率（浮点）
               classification: -1 / 0 / 1（空仓 / 观望 / 多头）
    """
    forward_bars = cfg.get('forward_bars', cfg.get('forward_days', 1))

    if freq == '1min':
        close = klines['close']
    else:
        close = klines['close'].resample(freq).last()

    forward_ret = close.pct_change(forward_bars).shift(-forward_bars)

    label_type = cfg.get('label_type', 'regression')

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

    raise ValueError(f"label_type 必须为 'regression' 或 'classification'，收到: {label_type!r}")
