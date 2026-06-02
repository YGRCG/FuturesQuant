"""
标签构建

调用方式:
    from ml.labels import build_labels
    y = build_labels(klines, cfg['labels'])
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_labels(klines: pd.DataFrame, cfg: dict) -> pd.Series:
    """
    构建日频预测标签。

    Parameters
    ----------
    klines : K 线 DataFrame（需含 close 列）
    cfg    : config.yaml 中的 labels 段

    Returns
    -------
    pd.Series  index=交易日，name='label'
               regression   : 未来 forward_days 日收益率（浮点）
               classification: -1 / 0 / 1（空仓 / 观望 / 多头）
    """
    forward_days = cfg['forward_days']
    daily_close  = klines['close'].resample('1D').last()
    forward_ret  = daily_close.pct_change(forward_days).shift(-forward_days)

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
