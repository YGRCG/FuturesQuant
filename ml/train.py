"""
ML 训练主入口

用法:
    # 从项目根目录运行
    python -m ml.train                        # 使用默认 config.yaml
    python -m ml.train --config ml/config.yaml

产出（ml/artifacts/）:
    model_<timestamp>.pkl              LightGBM 模型
    oof_preds_<timestamp>.parquet      OOF 预测值
    feature_importance_<timestamp>.csv 特征重要性
    metrics_<timestamp>.json           汇总指标
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import TimeSeriesSplit

# 确保项目根目录在 sys.path
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from futuresquant.data.loader import FuturesDataLoader
from futuresquant.data.universe import ContinuousContract
from ml.evaluate import oof_metrics, plot_fold_ic, plot_importance, plot_oof_nav
from ml.features import build_features
from ml.labels import build_labels


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_data(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (klines, klines_clean)。"""
    dcfg = cfg['data']
    loader = FuturesDataLoader(
        ROOT / dcfg['data_dir'],
        cache_dir=ROOT / dcfg['cache_dir'],
    )

    if dcfg.get('use_continuous', True):
        cc = ContinuousContract(loader, product=dcfg['product'],
                                adjust='back', roll_n_days_before_expiry=5)
        print(f'[{_ts()}] 构建连续主力合约 {dcfg["start_date"]} ~ {dcfg["end_date"]} …')
        klines = cc.build(start=dcfg['start_date'], end=dcfg['end_date'])

        buf = dcfg.get('roll_buffer', 30)
        roll_mask = klines['contract'] != klines['contract'].shift(1)
        locs = [klines.index.get_loc(ts) for ts in klines.index[roll_mask]]
        bad_pos: set[int] = set()
        for loc in locs:
            bad_pos.update(range(max(0, loc - buf), min(len(klines), loc + buf + 1)))
        klines_clean = klines.copy()
        klines_clean.loc[klines.index[sorted(bad_pos)], ['volume', 'open_interest']] = np.nan
        print(f'[{_ts()}] K线: {len(klines):,}  换月次数: {roll_mask.sum()}')
    else:
        contract = dcfg['single_contract']
        print(f'[{_ts()}] 加载单合约 {contract} …')
        klines = loader.load(contract, start=dcfg['start_date'], end=dcfg['end_date'])
        klines_clean = klines.copy()
        print(f'[{_ts()}] K线: {len(klines):,}')

    return klines, klines_clean


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

def train(cfg: dict) -> dict:
    """完整训练流程，返回汇总指标 dict。"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    artifacts = ROOT / cfg['output']['artifacts_dir']
    artifacts.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    klines, klines_clean = load_data(cfg)

    # 2. 特征 & 标签
    print(f'[{_ts()}] 构建特征 …')
    X = build_features(klines, klines_clean, cfg['features'])

    print(f'[{_ts()}] 构建标签 …')
    y = build_labels(klines, cfg['labels'])

    # 3. 对齐，去掉无法使用的行
    common = X.dropna(how='all').index.intersection(y.dropna().index)
    X, y = X.loc[common], y.loc[common]
    print(f'[{_ts()}] 有效样本: {len(X)} 天  特征数: {X.shape[1]}')

    # 4. 时序交叉验证
    tcfg = cfg['training']
    tscv = TimeSeriesSplit(
        n_splits=tcfg['n_splits'],
        gap=tcfg.get('gap', 1),
        test_size=tcfg.get('test_size', None),
        max_train_size=tcfg.get('max_train_size', None),
    )

    oof_preds = pd.Series(np.nan, index=y.index, name='oof_pred')
    fold_indices = []
    importance_list = []
    models = []

    mcfg = cfg['model']
    params = mcfg['params']

    print(f'[{_ts()}] 开始 {tcfg["n_splits"]} 折时序交叉验证 …')
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
        dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=mcfg.get('num_boost_round', 500),
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(mcfg.get('early_stopping_rounds', 50), verbose=False),
                lgb.log_evaluation(0),   # 静默
            ],
        )

        preds = model.predict(X_val)
        oof_preds.iloc[val_idx] = preds
        fold_indices.append((train_idx, val_idx))

        imp = pd.Series(model.feature_importance(importance_type='gain'),
                        index=X.columns)
        importance_list.append(imp)
        models.append(model)

        from ml.evaluate import rank_ic
        fold_ic = rank_ic(y_val, pd.Series(preds, index=y_val.index))
        print(f'  Fold {fold}  '
              f'train={len(train_idx)}d  val={len(val_idx)}d  '
              f'IC={fold_ic:.4f}  '
              f'trees={model.best_iteration}')

    # 5. 汇总指标
    metrics = oof_metrics(y, oof_preds, fold_indices)
    print(f'\n[{_ts()}] OOF 汇总:')
    for k, v in metrics.items():
        if k != 'fold_ICs':
            print(f'  {k:15s}: {v}')

    # 6. 保存产出
    # 使用最后一折模型作为代表（生产环境可改为全量重训）
    final_model = models[-1]
    model_path = artifacts / f'model_{timestamp}.pkl'
    joblib.dump(final_model, model_path)
    print(f'[{_ts()}] 模型已保存: {model_path}')

    oof_path = artifacts / f'oof_preds_{timestamp}.parquet'
    pd.concat([y.rename('y_true'), oof_preds], axis=1).to_parquet(oof_path)

    imp_mean = pd.concat(importance_list, axis=1).mean(axis=1).sort_values(ascending=False)
    imp_path = artifacts / f'feature_importance_{timestamp}.csv'
    imp_mean.to_csv(imp_path, header=['importance'])

    metrics_path = artifacts / f'metrics_{timestamp}.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)

    # 7. 可视化（保存 HTML）
    plot_fold_ic(metrics['fold_ICs']).write_html(
        artifacts / f'fold_ic_{timestamp}.html')
    plot_importance(imp_mean).write_html(
        artifacts / f'importance_{timestamp}.html')
    plot_oof_nav(y, oof_preds).write_html(
        artifacts / f'oof_nav_{timestamp}.html')

    print(f'[{_ts()}] 产出已保存至 {artifacts}/')
    return metrics


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _ts() -> str:
    return time.strftime('%H:%M:%S')


def load_config(path: str | Path) -> dict:
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FuturesQuant ML 训练')
    parser.add_argument('--config', default='ml/config.yaml',
                        help='配置文件路径（默认 ml/config.yaml）')
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    train(cfg)
