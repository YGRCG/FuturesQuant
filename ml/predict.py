"""
推理脚本 — 加载最新模型，生成交易信号

用法:
    python -m ml.predict                          # 最新模型，最近 20 根 bar
    python -m ml.predict --bars 50                # 最近 50 根 bar
    python -m ml.predict --model ml/artifacts/model_xxx.pkl

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
    n_recent_bars: int = 20,
) -> pd.DataFrame:
    """
    加载最新数据，使用已训练模型生成信号。

    Parameters
    ----------
    cfg            : 完整 config dict
    model_path     : 指定模型路径，None 则自动取最新
    n_recent_bars  : 返回最近 N 根 bar 的预测结果

    Returns
    -------
    pd.DataFrame  columns: ['pred', 'signal', 'strength']
                  signal  : 1=做多  0=观望  -1=做空
                  strength: 信号强度百分位 0~100（50=中性）
    """
    artifacts = ROOT / cfg['output']['artifacts_dir']
    if model_path is None:
        model_path = _latest_artifact(artifacts, 'model_')

    model = joblib.load(model_path)

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
    preds = pd.Series(model.predict(X_clean), index=X_clean.index, name='pred')

    # 信号强度：在全历史中的百分位排名（0~100）
    strength = preds.rank(pct=True) * 100

    # 信号：基于全历史 20%/80% 分位
    thr = 0.20
    q_lo = preds.quantile(thr)
    q_hi = preds.quantile(1 - thr)
    signals = pd.Series(0, index=preds.index, name='signal')
    signals[preds >= q_hi] = 1
    signals[preds <= q_lo] = -1

    result = pd.concat([preds, signals, strength.rename('strength')], axis=1)
    return result.tail(n_recent_bars)


def _format_output(result: pd.DataFrame, cfg: dict, model_path: Path) -> None:
    """格式化打印预测结果。"""
    freq = cfg['data'].get('freq', '1D')
    product = cfg['data']['product']
    forward = cfg['labels'].get('forward_bars', cfg['labels'].get('forward_days', 1))

    print(f'\n{"=" * 60}')
    print(f'  {product} ML 预测信号')
    print(f'{"=" * 60}')
    print(f'  模型:     {model_path.name}')
    print(f'  频率:     {freq}')
    print(f'  前瞻:     {forward} bars')
    print(f'  信号范围: 底部20%=做空  顶部20%=做多  中间=观望')
    print(f'{"=" * 60}')

    # 最新信号 — 醒目显示
    latest = result.iloc[-1]
    sig = int(latest['signal'])
    sig_map = {1: '做多 ▲', 0: '观望 —', -1: '做空 ▼'}
    strength = latest['strength']

    print(f'\n  >>> 最新信号: {sig_map[sig]}  '
          f'(强度 {strength:.0f}/100, 预测值 {latest["pred"]:+.6f})')
    print(f'  >>> 时间:     {result.index[-1]}')

    # 近期信号表
    print(f'\n  {"时间":<22} {"信号":<8} {"强度":>6} {"预测值":>12}')
    print(f'  {"-"*52}')

    for ts, row in result.iterrows():
        s = int(row['signal'])
        label = sig_map[s]
        ts_str = str(ts)[:16] if freq != '1D' else str(ts)[:10]
        print(f'  {ts_str:<22} {label:<8} {row["strength"]:>5.0f}  {row["pred"]:>+12.6f}')

    # 信号统计
    total = len(result)
    n_long = (result['signal'] == 1).sum()
    n_short = (result['signal'] == -1).sum()
    n_flat = (result['signal'] == 0).sum()
    print(f'\n  近 {total} 根 bar: '
          f'做多 {n_long} | 观望 {n_flat} | 做空 {n_short}')
    print()


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FuturesQuant ML 推理')
    parser.add_argument('--config', default='ml/config.yaml')
    parser.add_argument('--model',  default=None,
                        help='指定模型 .pkl 路径，默认取最新')
    parser.add_argument('--bars',   type=int, default=20,
                        help='打印最近 N 根 bar 的信号（默认 20）')
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    model_path = Path(args.model) if args.model else None

    artifacts = ROOT / cfg['output']['artifacts_dir']
    if model_path is None:
        model_path = _latest_artifact(artifacts, 'model_')

    result = predict_latest(cfg, model_path=model_path, n_recent_bars=args.bars)
    _format_output(result, cfg, model_path)
