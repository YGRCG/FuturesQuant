"""
自动化特征筛选 — 适配 regression / classification / triple_barrier

用法:
    python -m ml.select_features                      # 使用默认 config.yaml
    python -m ml.select_features --config ml/config.yaml

四步筛选流水线:
    1. IC 过滤 — 去除 |IC| < threshold 且 p > p_threshold 的噪声特征
    2. 相关性过滤 — 高相关组内只保留 IC 最高的
    3. LightGBM 重要性过滤 — 去除 Gain 低于中位数的特征
    4. Walk-Forward 稳定性过滤 — 去除 IC 符号一致性 < 50% 的不稳定特征

产出:
    ml/selected_features_{freq}_forward_{forward_bars}.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.evaluate import rank_ic, oof_metrics, freq_to_annual_factor
from ml.features import build_features
from ml.labels import build_labels
from ml.resample import session_resample_last
from ml.train import load_config, load_data


def _ts() -> str:
    return time.strftime('%H:%M:%S')


# ---------------------------------------------------------------------------
# Step 1: IC 分析
# ---------------------------------------------------------------------------

def calc_feature_ic(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    aligned = pd.concat([X, y.rename('__y__')], axis=1).dropna(subset=['__y__'])

    records = {}
    for col in X.columns:
        valid = aligned[[col, '__y__']].dropna()
        if len(valid) < 10:
            continue
        rho, pval = stats.spearmanr(valid[col], valid['__y__'])
        records[col] = {'IC': rho, 'p_value': pval, '|IC|': abs(rho)}

    return pd.DataFrame(records).T.sort_values('|IC|', ascending=False)


# ---------------------------------------------------------------------------
# Step 2: 相关性过滤
# ---------------------------------------------------------------------------

def correlation_filter(
    X: pd.DataFrame,
    ic_df: pd.DataFrame,
    corr_threshold: float = 0.7,
) -> set[str]:
    corr_full = X.corr(method='spearman')
    to_drop: set[str] = set()
    for i, c1 in enumerate(corr_full.columns):
        if c1 in to_drop:
            continue
        for j, c2 in enumerate(corr_full.columns):
            if i >= j or c2 in to_drop:
                continue
            r = corr_full.iloc[i, j]
            if pd.notna(r) and abs(r) >= corr_threshold:
                ic1 = ic_df.loc[c1, '|IC|'] if c1 in ic_df.index else 0
                ic2 = ic_df.loc[c2, '|IC|'] if c2 in ic_df.index else 0
                to_drop.add(c2 if ic1 >= ic2 else c1)
    return to_drop


# ---------------------------------------------------------------------------
# Step 3: LightGBM 重要性
# ---------------------------------------------------------------------------

def lgb_importance(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict,
) -> pd.Series:
    mcfg = cfg['model']
    tcfg = cfg['training']
    params = dict(mcfg['params'])
    label_type = cfg['labels'].get('label_type', 'regression')

    if label_type == 'triple_barrier':
        n_classes = int(y.nunique())
        if params.get('objective') == 'regression':
            params['objective'] = 'multiclass'
            params['num_class'] = n_classes
            params['metric'] = 'multi_logloss'
        label_map = {v: i for i, v in enumerate(sorted(y.unique()))}
        y_train = y.map(label_map)
    else:
        y_train = y

    tscv = TimeSeriesSplit(
        n_splits=tcfg['n_splits'],
        gap=tcfg.get('gap', 1),
    )

    gain_list = []
    for train_idx, val_idx in tscv.split(X):
        dtrain = lgb.Dataset(X.iloc[train_idx], label=y_train.iloc[train_idx], free_raw_data=False)
        dval = lgb.Dataset(X.iloc[val_idx], label=y_train.iloc[val_idx], reference=dtrain, free_raw_data=False)

        model = lgb.train(
            params, dtrain,
            num_boost_round=mcfg.get('num_boost_round', 500),
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(mcfg.get('early_stopping_rounds', 50), verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        gain_list.append(pd.Series(model.feature_importance('gain'), index=X.columns))

    return pd.concat(gain_list, axis=1).mean(axis=1).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Step 4: Walk-Forward 稳定性
# ---------------------------------------------------------------------------

def walk_forward_stability(
    X: pd.DataFrame,
    y: pd.Series,
    n_windows: int = 6,
) -> pd.Series:
    idx = X.dropna(how='all').index.intersection(y.dropna().index)
    splits = np.array_split(idx, n_windows)

    records: dict[str, list[float]] = {}
    for split_idx in splits:
        x_w = X.loc[split_idx]
        y_w = y.loc[split_idx]
        for col in X.columns:
            aligned = pd.concat([x_w[col], y_w], axis=1).dropna()
            if len(aligned) < 10:
                continue
            rho, _ = stats.spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
            records.setdefault(col, []).append(rho)

    consistency = {}
    for col, ics in records.items():
        s = pd.Series(ics)
        pos_pct = (s > 0).mean()
        neg_pct = (s < 0).mean()
        consistency[col] = max(pos_pct, neg_pct)

    return pd.Series(consistency).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def select_features(
    cfg: dict,
    ic_threshold: float = 0.005,
    p_threshold: float = 0.2,
    corr_threshold: float = 0.85,
    importance_quantile: float = 0.4,
    stability_threshold: float = 0.4,
) -> dict:
    klines, klines_clean = load_data(cfg)
    freq = cfg['data'].get('freq', '1D')

    print(f'[{_ts()}] 构建特征 …')
    X = build_features(klines, klines_clean, cfg['features'], freq=freq)

    print(f'[{_ts()}] 构建标签 …')
    y = build_labels(klines, cfg['labels'], freq=freq)

    valid_mask = X.notna().sum(axis=1) > (X.shape[1] // 2)
    common = X[valid_mask].index.intersection(y.dropna().index)
    X, y = X.loc[common], y.loc[common]
    print(f'[{_ts()}] 有效样本: {len(X)}  特征数: {X.shape[1]}')

    total = set(X.columns)

    # Step 1: IC
    print(f'[{_ts()}] Step 1: IC 过滤 (|IC| >= {ic_threshold}, p < {p_threshold}) …')
    ic_df = calc_feature_ic(X, y)
    noise = ic_df[(ic_df['|IC|'] < ic_threshold) & (ic_df['p_value'] > p_threshold)].index
    keep = total - set(noise)
    print(f'  {len(total)} → {len(keep)}  (-{len(noise)})')

    # Step 2: 相关性
    print(f'[{_ts()}] Step 2: 相关性过滤 (|r| < {corr_threshold}) …')
    X_sub = X[[c for c in X.columns if c in keep]]
    ic_sub = ic_df.loc[ic_df.index.isin(keep)]
    to_drop_corr = correlation_filter(X_sub, ic_sub, corr_threshold)
    keep = keep - to_drop_corr
    print(f'  → {len(keep)}  (-{len(to_drop_corr)})')

    # Step 3: 重要性
    print(f'[{_ts()}] Step 3: LightGBM 重要性过滤 (top {1 - importance_quantile:.0%}) …')
    X_sub = X[[c for c in X.columns if c in keep]]
    gain = lgb_importance(X_sub, y, cfg)
    threshold = gain.quantile(importance_quantile)
    low_imp = set(gain[gain < threshold].index)
    keep = keep - low_imp
    print(f'  → {len(keep)}  (-{len(low_imp)})')

    # Step 4: 稳定性
    print(f'[{_ts()}] Step 4: Walk-Forward 稳定性过滤 (consistency >= {stability_threshold:.0%}) …')
    X_sub = X[[c for c in X.columns if c in keep]]
    stability = walk_forward_stability(X_sub, y)
    unstable = set(stability[stability < stability_threshold].index)
    keep = keep - unstable
    print(f'  → {len(keep)}  (-{len(unstable)})')

    selected = sorted(keep)
    print(f'\n[{_ts()}] 最终保留 {len(selected)} 个特征')

    # OOF 对比
    bars_per_year = freq_to_annual_factor(freq)
    forward_bars = cfg['labels'].get('forward_bars', 1)
    close = session_resample_last(klines[['close']], freq)['close']
    y_ret = close.pct_change(forward_bars).shift(-forward_bars).rename('ret')
    y_ret = y_ret.loc[y_ret.index.isin(common)]

    label_type = cfg['labels'].get('label_type', 'regression')
    mcfg = cfg['model']
    tcfg = cfg['training']
    params = dict(mcfg['params'])

    if label_type == 'triple_barrier':
        n_classes = int(y.nunique())
        if params.get('objective') == 'regression':
            params['objective'] = 'multiclass'
            params['num_class'] = n_classes
            params['metric'] = 'multi_logloss'
        label_map = {v: i for i, v in enumerate(sorted(y.unique()))}
        y_train = y.map(label_map)
    else:
        y_train = y
        label_map = None

    tscv = TimeSeriesSplit(n_splits=tcfg['n_splits'], gap=tcfg.get('gap', 1))

    def _eval_features(cols):
        Xc = X[cols]
        oof = pd.Series(np.nan, index=y.index)
        fi = []
        for train_idx, val_idx in tscv.split(Xc):
            dt = lgb.Dataset(Xc.iloc[train_idx], label=y_train.iloc[train_idx], free_raw_data=False)
            dv = lgb.Dataset(Xc.iloc[val_idx], label=y_train.iloc[val_idx], reference=dt, free_raw_data=False)
            m = lgb.train(params, dt, num_boost_round=mcfg.get('num_boost_round', 500),
                          valid_sets=[dv], callbacks=[
                              lgb.early_stopping(mcfg.get('early_stopping_rounds', 50), verbose=False),
                              lgb.log_evaluation(0)])
            pr = m.predict(Xc.iloc[val_idx])
            if label_type == 'triple_barrier' and pr.ndim == 2:
                pr = pr[:, label_map[1]] - pr[:, label_map[-1]]
            oof.iloc[val_idx] = pr
            fi.append((train_idx, val_idx))
        return oof_metrics(y, oof, fi, bars_per_year=bars_per_year, y_ret=y_ret, forward_bars=forward_bars)

    print(f'[{_ts()}] 对比评估: 全特征 vs 筛选后 …')
    metrics_full = _eval_features(list(X.columns))
    metrics_sel = _eval_features(selected)

    print(f'  全特征 ({X.shape[1]}): IC={metrics_full["IC"]}  ICIR={metrics_full["ICIR"]}  Sharpe={metrics_full["Sharpe"]}')
    print(f'  筛选后 ({len(selected)}): IC={metrics_sel["IC"]}  ICIR={metrics_sel["ICIR"]}  Sharpe={metrics_sel["Sharpe"]}')

    return {
        'selected_features': selected,
        'n_features': len(selected),
        'pipeline_config': {
            'ic_threshold': ic_threshold,
            'p_threshold': p_threshold,
            'corr_threshold': corr_threshold,
            'importance_quantile': importance_quantile,
            'stability_threshold': stability_threshold,
        },
        'metrics_comparison': {
            'full': {
                'n_features': int(X.shape[1]),
                'IC': float(metrics_full['IC']),
                'ICIR': float(metrics_full['ICIR']),
                'Sharpe': float(metrics_full['Sharpe']),
            },
            'selected': {
                'n_features': len(selected),
                'IC': float(metrics_sel['IC']),
                'ICIR': float(metrics_sel['ICIR']),
                'Sharpe': float(metrics_sel['Sharpe']),
            },
        },
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FuturesQuant 特征筛选')
    parser.add_argument('--config', default='ml/config.yaml')
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    result = select_features(cfg)

    freq = cfg['data'].get('freq', '1D').replace('/', '')
    forward_bars = cfg['labels'].get('forward_bars', 1)
    fname = f'selected_features_{freq}_forward_{forward_bars}.yaml'
    output_path = ROOT / 'ml' / fname

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(result, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f'\n[{_ts()}] 已导出: {output_path}')
    print(f'在 config.yaml 中配置:')
    print(f'  selected_features_file: ml/{fname}')
