"""
ML 回测闭环 — 训练好的模型 → 生成信号 → 接入 BacktestEngine → 绩效报告

用法:
    python -m ml.backtest                                    # 最新模型 + 默认配置
    python -m ml.backtest --model ml/artifacts/model_xxx.pkl
    python -m ml.backtest --config ml/config.yaml

也可作为库导入:
    from ml.backtest import run_ml_backtest
    result = run_ml_backtest(cfg)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from futuresquant.backtest.engine import BacktestEngine, BacktestConfig
from futuresquant.strategy.examples.ml_strategy import MLStrategy
from ml.features import build_features
from ml.resample import session_resample_last, session_resample_ohlc
from ml.train import load_data, load_config


def _ts() -> str:
    return time.strftime('%H:%M:%S')


def _latest_model(artifacts_dir: Path) -> Path:
    files = sorted(artifacts_dir.glob('model_*.pkl'))
    if not files:
        raise FileNotFoundError(f'{artifacts_dir} 下未找到 model_*.pkl，请先运行 train.py')
    return files[-1]


def run_ml_backtest(
    cfg: dict,
    model_path: Path | None = None,
    entry_quantile: float = 0.8,
    exit_quantile: float = 0.5,
    rolling_window: int = 500,
    signal_ema_span: int | None = None,
):
    """
    完整 ML 回测闭环。

    Returns
    -------
    BacktestResult, signal_series
    """
    artifacts = ROOT / cfg['output']['artifacts_dir']
    if model_path is None:
        model_path = _latest_model(artifacts)

    model_or_models = joblib.load(model_path)
    if isinstance(model_or_models, list):
        models = model_or_models
        print(f'[{_ts()}] 集成模型: {model_path.name} ({len(models)} 折)')
    else:
        models = [model_or_models]
        print(f'[{_ts()}] 单模型: {model_path.name}')

    # 1. 加载数据（回测时段优先用 backtest 节，避免与训练期混用）
    bt_section = cfg.get('backtest', {})
    if bt_section:
        data_cfg = dict(cfg['data'])
        data_cfg['start_date'] = bt_section.get('start_date', data_cfg['start_date'])
        data_cfg['end_date'] = bt_section.get('end_date', data_cfg['end_date'])
        cfg = {**cfg, 'data': data_cfg}
        print(f'[{_ts()}] 回测时段: {data_cfg["start_date"]} ~ {data_cfg["end_date"]}')
    klines, klines_clean = load_data(cfg)
    freq = cfg['data'].get('freq', '1D')

    # 2. 构建特征
    print(f'[{_ts()}] 构建特征 (freq={freq}) …')
    X = build_features(klines, klines_clean, cfg['features'], freq=freq)

    # 加载筛选特征
    sel_file = cfg['features'].get('selected_features_file')
    if sel_file:
        with open(ROOT / sel_file, encoding='utf-8') as f:
            sel_cols = yaml.safe_load(f)['selected_features']
        X = X[[c for c in sel_cols if c in X.columns]]
        print(f'[{_ts()}] 筛选特征: {len(X.columns)} 个')

    valid_mask = X.notna().sum(axis=1) > (X.shape[1] // 2)
    X_clean = X[valid_mask]

    # 3. 生成全历史预测信号（多折集成：平均所有模型预测）
    print(f'[{_ts()}] 生成预测信号 ({len(X_clean)} bars, {len(models)} 模型集成) …')
    all_preds = [m.predict(X_clean) for m in models]
    preds_raw = np.mean(all_preds, axis=0)

    # multiclass 模型返回 (n, n_classes) 概率矩阵，转为连续信号
    if preds_raw.ndim == 2:
        signal_agg = pd.Series(
            preds_raw[:, -1] - preds_raw[:, 0],
            index=X_clean.index, name='ml_signal',
        )
    else:
        signal_agg = pd.Series(
            preds_raw, index=X_clean.index, name='ml_signal',
        )

    # 4. 回测在聚合频率的 K 线上运行（session-aware，不跨交易时段）
    if freq == '1min':
        klines_bt = klines
        signal_bt = signal_agg
    else:
        klines_bt = session_resample_ohlc(klines, freq)
        # 确保 signal 与 klines_bt 的 index 完全对齐
        signal_bt = signal_agg.reindex(klines_bt.index)

    print(f'[{_ts()}] 回测 K 线: {len(klines_bt):,} bars (freq={freq})')

    # 5. 构造策略 & 运行回测
    if signal_ema_span:
        print(f'[{_ts()}] 信号 EMA 平滑: span={signal_ema_span}')

    strategy = MLStrategy(
        signal=signal_bt,
        entry_quantile=entry_quantile,
        exit_quantile=exit_quantile,
        rolling_window=rolling_window,
        signal_ema_span=signal_ema_span,
    )

    symbol = cfg['data']['product']
    config = BacktestConfig(
        symbol=symbol,
        initial_capital=1_000_000,
        slippage_ticks=1,
    )

    print(f'[{_ts()}] 运行回测 …')
    result = BacktestEngine(strategy, config).run(klines_bt)
    result.print_summary()

    return result, signal_agg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FuturesQuant ML 回测')
    parser.add_argument('--config', default='ml/config.yaml')
    parser.add_argument('--model', default=None, help='模型 .pkl 路径，默认取最新')
    parser.add_argument('--entry-q', type=float, default=None, help='开仓信号分位 (覆盖 config)')
    parser.add_argument('--exit-q', type=float, default=None, help='平仓信号分位 (覆盖 config)')
    parser.add_argument('--rolling', type=int, default=None, help='滚动窗口 (覆盖 config)')
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    mp = Path(args.model) if args.model else None

    bt_cfg = cfg.get('backtest', {})
    run_ml_backtest(
        cfg, model_path=mp,
        entry_quantile=args.entry_q if args.entry_q is not None else bt_cfg.get('entry_quantile', 0.8),
        exit_quantile=args.exit_q if args.exit_q is not None else bt_cfg.get('exit_quantile', 0.5),
        rolling_window=args.rolling if args.rolling is not None else bt_cfg.get('rolling_window', 500),
        signal_ema_span=bt_cfg.get('signal_ema_span'),
    )
