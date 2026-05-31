"""
多因子 IC 加权信号合成。

原理
----
1. 对每个因子计算滚动 Rank IC（因子值与未来收益的 Spearman 相关系数）
2. 用滚动 IC 作为权重（正 IC 做多，负 IC 做空，绝对值越大权重越高）
3. 对因子值做截面/时序 z-score 标准化
4. 合成信号 = Σ( IC_i × z_factor_i ) / Σ|IC_i|
5. 输出信号再做时序标准化，使其分布稳定

用法
----
    signal_gen = MultiFactorSignal(
        factors=[ROC(20), MACross(5, 20), NormATR(14)],
        forward_bars=60,   # 计算 IC 使用的前瞻窗口
        ic_window=20,      # 滚动 IC 的回看窗口（bar 数）
        norm_window=240,   # 信号时序标准化窗口
    )
    signal = signal_gen.compute(klines)   # pd.Series, 范围约 [-3, +3]
    # > threshold → 做多；< -threshold → 做空
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from futuresquant.factors.base import Factor, zscore


class MultiFactorSignal:
    """
    IC-加权多因子合成信号。

    Parameters
    ----------
    factors       : 参与合成的 Factor 列表
    forward_bars  : 计算 IC 的前瞻期（bars）
    ic_window     : 滚动 IC 的回看窗口（bars）；越大越稳定但响应越慢
    norm_window   : 合成信号的时序 z-score 窗口（bars）
    min_ic_abs    : 低于此绝对 IC 的因子权重置 0（过滤噪声因子）
    """

    def __init__(
        self,
        factors: Sequence[Factor],
        forward_bars: int = 60,
        ic_window: int = 240 * 5,   # 5 天 × 240 根 1min bar
        norm_window: int = 240 * 20,
        min_ic_abs: float = 0.01,
    ):
        self.factors = list(factors)
        self.forward_bars = forward_bars
        self.ic_window = ic_window
        self.norm_window = norm_window
        self.min_ic_abs = min_ic_abs

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------

    def compute(self, klines: pd.DataFrame) -> pd.Series:
        """
        计算合成信号序列。

        Returns
        -------
        pd.Series，index=DatetimeIndex，name="MultiFactorSignal"
        正值偏多，负值偏空；幅度反映信号强度。
        """
        factor_df = self._compute_factors(klines)
        ic_df = self._rolling_ic(factor_df, klines)
        signal = self._combine(factor_df, ic_df)
        return zscore(signal, self.norm_window).rename("MultiFactorSignal")

    def factor_weights(self, klines: pd.DataFrame) -> pd.DataFrame:
        """返回每个因子的滚动权重，便于可视化。"""
        factor_df = self._compute_factors(klines)
        ic_df = self._rolling_ic(factor_df, klines)
        ic_abs_sum = ic_df.abs().sum(axis=1).replace(0, np.nan)
        weights = ic_df.div(ic_abs_sum, axis=0)
        weights[ic_df.abs() < self.min_ic_abs] = 0.0
        return weights

    # ------------------------------------------------------------------
    # 内部步骤
    # ------------------------------------------------------------------

    def _compute_factors(self, klines: pd.DataFrame) -> pd.DataFrame:
        """计算并 z-score 标准化所有因子。"""
        series = {}
        for f in self.factors:
            raw = f.compute(klines)
            series[f.name] = zscore(raw, self.norm_window)
        return pd.DataFrame(series, index=klines.index)

    def _rolling_ic(
        self, factor_df: pd.DataFrame, klines: pd.DataFrame
    ) -> pd.DataFrame:
        """
        逐 bar 滚动 Rank IC：
        corr( factor[t-window:t], fwd_ret[t-window:t] )
        """
        fwd_ret = klines["close"].pct_change(self.forward_bars).shift(-self.forward_bars)

        ic_records: dict[str, pd.Series] = {}
        for col in factor_df.columns:
            f_series = factor_df[col]
            ic_vals = []
            for i in range(len(factor_df)):
                if i < self.ic_window:
                    ic_vals.append(np.nan)
                    continue
                window_f = f_series.iloc[i - self.ic_window: i]
                window_r = fwd_ret.iloc[i - self.ic_window: i]
                mask = window_f.notna() & window_r.notna()
                if mask.sum() < 10:
                    ic_vals.append(np.nan)
                    continue
                rho, _ = stats.spearmanr(window_f[mask], window_r[mask])
                ic_vals.append(rho if not np.isnan(rho) else 0.0)
            ic_records[col] = pd.Series(ic_vals, index=factor_df.index)

        return pd.DataFrame(ic_records, index=factor_df.index)

    def _combine(
        self, factor_df: pd.DataFrame, ic_df: pd.DataFrame
    ) -> pd.Series:
        """IC 加权求和：signal_t = Σ( IC_i(t) × z_factor_i(t) ) / Σ|IC_i(t)|"""
        # 低 IC 因子权重置零
        filtered_ic = ic_df.where(ic_df.abs() >= self.min_ic_abs, other=0.0)
        ic_abs_sum = filtered_ic.abs().sum(axis=1).replace(0, np.nan)
        weights = filtered_ic.div(ic_abs_sum, axis=0)    # 行归一化

        # 逐元素乘法后按行求和
        weighted = factor_df.mul(weights)
        return weighted.sum(axis=1, min_count=1).rename("raw_signal")
