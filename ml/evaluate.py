"""
评估工具

包含：
    rank_ic          — 预测值与实际收益的 Spearman 相关系数
    oof_metrics      — 汇总所有 OOF 折的评估指标
    sharpe_from_pred — 根据预测信号计算多空 Sharpe
    plot_importance  — 特征重要性柱状图
    plot_oof_nav     — OOF 多空净值曲线
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px


# ---------------------------------------------------------------------------
# 频率工具
# ---------------------------------------------------------------------------

# FU 每日交易约 240 根 1min bar（夜盘 21:00-23:00 + 日盘 09:00-15:00，扣休息）
_FU_MINUTES_PER_DAY = 240
_TRADING_DAYS_PER_YEAR = 252

_FREQ_TO_BARS_PER_YEAR: dict[str, int] = {
    '1min':  _TRADING_DAYS_PER_YEAR * _FU_MINUTES_PER_DAY,        # ~60480
    '5min':  _TRADING_DAYS_PER_YEAR * (_FU_MINUTES_PER_DAY // 5), # ~12096
    '15min': _TRADING_DAYS_PER_YEAR * (_FU_MINUTES_PER_DAY // 15),# ~4032
    '30min': _TRADING_DAYS_PER_YEAR * (_FU_MINUTES_PER_DAY // 30),# ~2016
    '1H':    _TRADING_DAYS_PER_YEAR * (_FU_MINUTES_PER_DAY // 60),# ~1008
    '4H':    _TRADING_DAYS_PER_YEAR * 1,                          # FU 每天交易~4h，恰好1根4H bar
    '1D':    _TRADING_DAYS_PER_YEAR,                               # 252
}


def freq_to_annual_factor(freq: str) -> int:
    """将频率字符串转换为年化 bar 数（用于 Sharpe 等年化计算）。"""
    if freq in _FREQ_TO_BARS_PER_YEAR:
        return _FREQ_TO_BARS_PER_YEAR[freq]
    raise ValueError(
        f"不支持的频率 '{freq}'，可选: {list(_FREQ_TO_BARS_PER_YEAR.keys())}")


# ---------------------------------------------------------------------------
# 核心指标
# ---------------------------------------------------------------------------

def rank_ic(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Spearman Rank IC（期货 ML 最核心的评估指标）。"""
    aligned = pd.concat([y_true, y_pred], axis=1).dropna()
    if len(aligned) < 5:
        return np.nan
    rho, _ = stats.spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return rho


def oof_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    fold_indices: list[tuple[np.ndarray, np.ndarray]] | None = None,
    bars_per_year: int = 252,
) -> dict:
    """
    汇总 OOF 评估指标。

    Parameters
    ----------
    y_true        : 真实标签
    y_pred        : OOF 预测值（与 y_true 同索引）
    fold_indices  : [(train_idx, val_idx), ...] 可选，用于逐折 IC 统计
    bars_per_year : 年化系数（日频=252，小时频=1008 等）

    Returns
    -------
    dict with keys: ic, icir, ic_pos_pct, sharpe, max_drawdown
    """
    ic_val = rank_ic(y_true, y_pred)

    fold_ics = []
    if fold_indices:
        for _, val_idx in fold_indices:
            yt = y_true.iloc[val_idx]
            yp = pd.Series(y_pred.values[val_idx], index=yt.index)
            fold_ics.append(rank_ic(yt, yp))

    icir = (np.nanmean(fold_ics) / np.nanstd(fold_ics)
            if len(fold_ics) > 1 else np.nan)
    ic_pos = np.mean([v > 0 for v in fold_ics]) if fold_ics else np.nan

    sharpe, mdd = sharpe_from_pred(y_true, y_pred,
                                   bars_per_year=bars_per_year)

    return {
        'IC':         round(ic_val, 4),
        'ICIR':       round(icir, 4) if not np.isnan(icir) else np.nan,
        'IC_pos_pct': round(ic_pos, 3) if not np.isnan(ic_pos) else np.nan,
        'Sharpe':     round(sharpe, 3),
        'MaxDrawdown': round(mdd, 4),
        'fold_ICs':   [round(v, 4) for v in fold_ics],
    }


def sharpe_from_pred(
    y_true: pd.Series,
    y_pred: pd.Series,
    n_quantiles: int = 5,
    bars_per_year: int = 252,
) -> tuple[float, float]:
    """
    根据预测信号构造多空组合，计算年化 Sharpe 和最大回撤。

    做多：预测值 >= 80th 百分位；做空：<= 20th 百分位。
    bars_per_year 控制年化系数（日频=252，小时频=1008 等）。
    """
    df = pd.concat([y_true, y_pred], axis=1).dropna()
    df.columns = ['ret', 'pred']

    thr = 1.0 / n_quantiles
    df['signal'] = 0
    df.loc[df['pred'] >= df['pred'].quantile(1 - thr), 'signal'] = 1
    df.loc[df['pred'] <= df['pred'].quantile(thr),     'signal'] = -1

    ls_ret = df['signal'].shift(1) * df['ret']
    ls_ret = ls_ret.dropna()

    if ls_ret.std() == 0 or len(ls_ret) < 5:
        return np.nan, np.nan

    sharpe = ls_ret.mean() / ls_ret.std() * np.sqrt(bars_per_year)
    nav = (1 + ls_ret).cumprod()
    mdd = ((nav.cummax() - nav) / nav.cummax()).max()
    return sharpe, mdd


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def plot_importance(importance: pd.Series, top_n: int = 30) -> go.Figure:
    """特征重要性条形图（Gain）。"""
    top = importance.sort_values(ascending=False).head(top_n)
    fig = px.bar(
        top.reset_index(),
        x='index', y=top.name or 0,
        title=f'特征重要性 Top {top_n}（Gain）',
        labels={'index': '特征', top.name or 0: 'Importance'},
    )
    fig.update_layout(height=450, xaxis_tickangle=-45)
    return fig


def plot_oof_nav(y_true: pd.Series, y_pred: pd.Series) -> go.Figure:
    """OOF 多空净值曲线 vs 买入持有。"""
    df = pd.concat([y_true, y_pred], axis=1).dropna()
    df.columns = ['ret', 'pred']

    thr = 0.2
    df['signal'] = 0
    df.loc[df['pred'] >= df['pred'].quantile(1 - thr), 'signal'] = 1
    df.loc[df['pred'] <= df['pred'].quantile(thr),     'signal'] = -1

    df['ls_ret'] = df['signal'].shift(1) * df['ret']
    df['ls_nav'] = (1 + df['ls_ret'].fillna(0)).cumprod()
    df['bh_nav'] = (1 + df['ret'].fillna(0)).cumprod()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['ls_nav'],
                             name='ML 多空策略', line=dict(width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['bh_nav'],
                             name='买入持有', line=dict(dash='dot', color='gray')))
    fig.update_layout(title='OOF 多空净值曲线', height=400,
                      yaxis_title='净值', xaxis_title='日期')
    return fig


def plot_fold_ic(fold_ics: list[float]) -> go.Figure:
    """逐折 IC 柱状图。"""
    colors = ['#2ca02c' if v > 0 else '#d62728' for v in fold_ics]
    fig = go.Figure(go.Bar(
        x=[f'Fold {i+1}' for i in range(len(fold_ics))],
        y=fold_ics,
        marker_color=colors,
    ))
    fig.add_hline(y=0, line_color='black', line_width=0.8)
    mean_ic = np.nanmean(fold_ics)
    fig.add_hline(y=mean_ic, line_dash='dash', line_color='orange',
                  annotation_text=f'均值={mean_ic:.4f}')
    fig.update_layout(title='各折 OOF Rank IC', height=350,
                      yaxis_title='IC', xaxis_title='')
    return fig
