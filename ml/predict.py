"""
推理脚本 — 加载最新模型，生成当日信号

用法:
    python -m ml.predict                          # 用最新模型预测最新数据
    python -m ml.predict --model ml/artifacts/model_20250101_120000.pkl

输出:
    标准输出打印当日信号（-1 做空 / 0 观望 / 1 做多）及置信度
    也可作为库导入：from ml.predict import predict_latest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from futuresquant.data.loader import FuturesDataLoader
from futuresquant.data.universe import ContinuousContract
from ml.features import build_features
from ml.train import load_config


def _latest_artifact(artifacts_dir: Path, prefix: str) -> Path:
    """返回 artifacts_dir 下最新的以 prefix 开头的文件。"""
    files = sorted(artifacts_dir.glob(f'{prefix}*.pkl'))
    if not files:
        raise FileNotFoundError(
            f'{artifacts_dir} 下未找到 {prefix}*.pkl，请先运行 train.py')
    return files[-1]


def predict_latest(
    cfg: dict,
    model_path: Path | None = None,
    n_recent_days: int = 5,
) -> pd.DataFrame:
    """
    加载最新数据，使用已训练模型生成信号。

    Parameters
    ----------
    cfg           : 完整 config dict
    model_path    : 指定模型路径，None 则自动取最新
    n_recent_days : 返回最近 N 个交易日的预测结果

    Returns
    -------
    pd.DataFrame  columns: ['pred', 'signal']
                  signal: 1=做多  0=观望  -1=做空（基于 20%/80% 分位）
    """
    artifacts = ROOT / cfg['output']['artifacts_dir']
    if model_path is None:
        model_path = _latest_artifact(artifacts, 'model_')

    model = joblib.load(model_path)
    print(f'模型: {model_path.name}')

    # 加载最新数据
    dcfg = cfg['data']
    loader = FuturesDataLoader(
        ROOT / dcfg['data_dir'],
        cache_dir=ROOT / dcfg['cache_dir'],
    )
    if dcfg.get('use_continuous', True):
        cc = ContinuousContract(loader, product=dcfg['product'],
                                adjust='back', roll_n_days_before_expiry=5)
        klines = cc.build(start=dcfg['start_date'], end=dcfg['end_date'])
        buf = dcfg.get('roll_buffer', 30)
        roll_mask = klines['contract'] != klines['contract'].shift(1)
        locs = [klines.index.get_loc(ts) for ts in klines.index[roll_mask]]
        bad_pos: set[int] = set()
        for loc in locs:
            bad_pos.update(range(max(0, loc - buf), min(len(klines), loc + buf + 1)))
        klines_clean = klines.copy()
        klines_clean.loc[klines.index[sorted(bad_pos)], ['volume', 'open_interest']] = np.nan
    else:
        klines = loader.load(dcfg['single_contract'],
                             start=dcfg['start_date'], end=dcfg['end_date'])
        klines_clean = klines.copy()

    # 构建特征
    freq = cfg['data'].get('freq', '1D')
    X = build_features(klines, klines_clean, cfg['features'], freq=freq)

    # 加载筛选特征（如有配置）
    sel_file = cfg['features'].get('selected_features_file')
    if sel_file:
        import yaml as _yaml
        with open(ROOT / sel_file, encoding='utf-8') as f:
            sel_cols = _yaml.unsafe_load(f)['selected_features']
        X = X[[c for c in sel_cols if c in X.columns]]

    X_clean = X.dropna(how='all')

    # 预测
    preds = pd.Series(
        model.predict(X_clean),
        index=X_clean.index,
        name='pred',
    )

    # 信号：基于全历史分位确定多空阈值（避免未来函数）
    thr = 0.20
    q_lo = preds.quantile(thr)
    q_hi = preds.quantile(1 - thr)
    signals = pd.Series(0, index=preds.index, name='signal')
    signals[preds >= q_hi] = 1
    signals[preds <= q_lo] = -1

    result = pd.concat([preds, signals], axis=1).tail(n_recent_days)
    return result


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FuturesQuant ML 推理')
    parser.add_argument('--config', default='ml/config.yaml')
    parser.add_argument('--model',  default=None,
                        help='指定模型 .pkl 路径，默认取最新')
    parser.add_argument('--days',   type=int, default=5,
                        help='打印最近 N 个交易日的信号（默认 5）')
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    model_path = Path(args.model) if args.model else None

    result = predict_latest(cfg, model_path=model_path, n_recent_days=args.days)

    signal_map = {1: '做多 ▲', 0: '观望 —', -1: '做空 ▼'}
    print(f'\n最近 {args.days} 个交易日信号:')
    print('-' * 40)
    for date, row in result.iterrows():
        label = signal_map.get(int(row['signal']), '—')
        print(f'  {str(date)[:10]}  pred={row["pred"]:+.5f}  {label}')
