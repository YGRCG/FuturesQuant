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

    model = joblib.load(model_path)
    print(f'[{_ts()}] 模型: {model_path.name}')

    # 1. 加载数据
    klines, klines_clean = load_data(cfg)
    freq = cfg['data'].get('freq', '1D')

    # 2. 构建特征
    print(f'[{_ts()}] 构建特征 (freq={freq}) …')
    X = build_features(klines, klines_clean, cfg['features'], freq=freq)

    # 加载筛选特征
    sel_file = cfg['features'].get('selected_features_file')
    if sel_file:
        with open(ROOT / sel_file, encoding='utf-8') as f:
            sel_cols = yaml.unsafe_load(f)['selected_features']
        X = X[[c for c in sel_cols if c in X.columns]]
        print(f'[{_ts()}] 筛选特征: {len(X.columns)} 个')

    X_clean = X.dropna(how='all')

    # 3. 生成全历史预测信号
    print(f'[{_ts()}] 生成预测信号 ({len(X_clean)} bars) …')
    signal_agg = pd.Series(
        model.predict(X_clean),
        index=X_clean.index,
        name='ml_signal',
    )

    # 4. 将聚合频率信号映射到 1min K 线索引
    if freq != '1min':
        signal_1min = signal_agg.reindex(klines.index).ffill()
    else:
        signal_1min = signal_agg

    print(f'[{_ts()}] 信号映射到 1min: {signal_1min.notna().sum():,} bars')

    # 5. 构造策略 & 运行回测
    strategy = MLStrategy(
        signal=signal_1min,
        entry_quantile=entry_quantile,
        exit_quantile=exit_quantile,
        rolling_window=rolling_window,
    )

    symbol = cfg['data']['product']
    config = BacktestConfig(
        symbol=symbol,
        initial_capital=1_000_000,
        slippage_ticks=1,
    )

    print(f'[{_ts()}] 运行回测 …')
    result = BacktestEngine(strategy, config).run(klines)
    result.print_summary()

    return result, signal_agg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FuturesQuant ML 回测')
    parser.add_argument('--config', default='ml/config.yaml')
    parser.add_argument('--model', default=None, help='模型 .pkl 路径，默认取最新')
    parser.add_argument('--entry-q', type=float, default=0.8, help='开仓信号分位 (默认 0.8)')
    parser.add_argument('--exit-q', type=float, default=0.5, help='平仓信号分位 (默认 0.5)')
    parser.add_argument('--rolling', type=int, default=500, help='滚动窗口 (默认 500)')
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    mp = Path(args.model) if args.model else None

    run_ml_backtest(
        cfg, model_path=mp,
        entry_quantile=args.entry_q,
        exit_quantile=args.exit_q,
        rolling_window=args.rolling,
    )
