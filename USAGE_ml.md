# ML 流水线使用指南

## 目录结构

```
ml/
├── config.yaml        # 所有参数配置入口
├── features.py        # 特征构建（因子 + 滞后/滚动统计）
├── labels.py          # 标签定义（回归 / 分类 / Triple Barrier）
├── train.py           # 训练主入口
├── evaluate.py        # 评估工具（IC/ICIR/Sharpe/图表）
├── predict.py         # 推理，输出交易信号
├── backtest.py        # 回测闭环（模型 → 信号 → BacktestEngine）
├── selected_features_*.yaml  # 特征筛选结果（notebook 06 导出）
└── artifacts/         # 训练产出（自动生成，不入库）
    ├── model_<时间戳>.pkl
    ├── oof_preds_<时间戳>.parquet
    ├── feature_importance_<时间戳>.csv
    ├── metrics_<时间戳>.json
    └── *.html

research/
├── 06_ml_factor_analysis.ipynb   # ML 因子筛选 & A/B 标签对比
└── 07_ml_backtest.ipynb          # ML 回测分析（权益曲线/回撤/成交）
```

---

## 完整工作流

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: 因子分析 & 特征筛选                                │
│    打开 research/06_ml_factor_analysis.ipynb                │
│    → IC 分析 → 相关性过滤 → LightGBM 重要性 → SHAP        │
│    → 自动三步筛选 → 导出 ml/selected_features_*.yaml       │
│    → Regression vs Triple Barrier A/B 对比                 │
├─────────────────────────────────────────────────────────────┤
│  Step 2: 配置                                               │
│    编辑 ml/config.yaml（freq / 标签 / 筛选特征文件）        │
├─────────────────────────────────────────────────────────────┤
│  Step 3: 训练                                               │
│    python -m ml.train                                       │
├─────────────────────────────────────────────────────────────┤
│  Step 4: 回测                                               │
│    python -m ml.backtest                                    │
│    或打开 research/07_ml_backtest.ipynb 交互分析            │
├─────────────────────────────────────────────────────────────┤
│  Step 5: 推理（生成最新信号）                               │
│    python -m ml.predict                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Step 1: 因子分析 & 特征筛选

打开 `research/06_ml_factor_analysis.ipynb`，依次运行：

1. **IC 分析** — 对齐 ML 标签窗口的特征预测力排名
2. **相关性过滤** — 去除 |r| > 0.7 的冗余特征
3. **LightGBM 重要性** — Gain / Split 两种度量
4. **SHAP** — 理解模型如何使用因子
5. **Walk-Forward IC** — 检查因子时间稳定性
6. **三步自动筛选** — IC → 相关性 → 重要性，输出精选特征列表
7. **A/B 标签对比** — Regression vs Triple Barrier 净值对比

运行到最后，会导出 `ml/selected_features_*.yaml`。

---

## Step 2: 配置 ml/config.yaml

### 核心参数

```yaml
data:
  freq: "15min"                 # 预测频率: 1min / 5min / 15min / 30min / 1H / 4H / 1D

features:
  lags: [1, 5, 20]
  rolling_windows: [20]
  selected_features_file: ml/selected_features_15min_1.yaml  # 筛选特征文件，null=全量

labels:
  forward_bars: 4               # 预测未来几根 bar 的收益
  label_type: regression        # regression | classification | triple_barrier
  # ── triple_barrier 专用 ──
  # atr_period: 14
  # atr_multiplier: 1.5

training:
  n_splits: 5
  gap: 4                        # 必须 >= forward_bars，防止标签泄漏
```

### 常用配置组合

| 场景 | freq | forward_bars | label_type | gap |
|------|------|-------------|------------|-----|
| 日频回归 | 1D | 1 | regression | 1 |
| 小时频回归 | 1H | 4 | regression | 4 |
| 15分钟 Triple Barrier | 15min | 20 | triple_barrier | 20 |
| 5分钟短线 | 5min | 12 | regression | 12 |

### 切换单合约 / 连续合约

```yaml
data:
  use_continuous: false
  single_contract: FU2505
```

### 模型超参数

```yaml
model:
  params:
    num_leaves: 15              # 样本少时压低复杂度
    learning_rate: 0.02
    min_child_samples: 50
  num_boost_round: 1000
  early_stopping_rounds: 100
```

---

## Step 3: 训练

```bash
python -m ml.train
python -m ml.train --config ml/config.yaml   # 指定配置
```

输出示例：

```
[09:12:01] 构建连续主力合约 2021-01-01 ~ 2025-04-30 …
[09:12:06] K线: 357,611  换月次数: 85
[09:12:06] 频率: 15min  年化系数: 4032
[09:12:08] 构建特征 …
[09:12:08] 已加载筛选特征: selected_features_15min_1.yaml  (58 个)
[09:12:15] 构建标签 …
[09:12:15] 有效样本: 15680 bars  特征数: 58
[09:12:15] 开始 5 折时序交叉验证 …
  Fold 1  train=2560  val=2612  IC=0.0321  trees=312
  ...

OOF 汇总:
  IC             : 0.0412
  ICIR           : 0.3817
  Sharpe         : 0.721
  MaxDrawdown    : 0.1843

[09:12:38] 产出已保存至 ml/artifacts/
```

产出文件：

| 文件 | 用途 |
|------|------|
| `model_<ts>.pkl` | 训练好的模型，供 predict / backtest 使用 |
| `oof_preds_<ts>.parquet` | OOF 预测值，供进一步分析 |
| `feature_importance_<ts>.csv` | 特征重要性排名 |
| `metrics_<ts>.json` | IC / ICIR / Sharpe 等指标 |
| `fold_ic_<ts>.html` | 逐折 IC 柱状图 |
| `importance_<ts>.html` | 特征重要性图 |
| `oof_nav_<ts>.html` | OOF 多空净值曲线 |

---

## Step 4: 回测

### 命令行

```bash
# 使用最新模型 + 默认参数
python -m ml.backtest

# 指定参数
python -m ml.backtest --model ml/artifacts/model_xxx.pkl --entry-q 0.8 --exit-q 0.5 --rolling 500
```

参数说明：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--entry-q` | 0.8 | 信号进入 top 20% 开多，bottom 20% 开空 |
| `--exit-q` | 0.5 | 信号回到中位以内平仓 |
| `--rolling` | 500 | 滚动窗口（bar 数），用于计算分位阈值 |

### Notebook 交互分析

打开 `research/07_ml_backtest.ipynb`，包含：

- 预测信号分布
- 权益曲线 + 回撤 + 标的价格（三图联动）
- 成交点标注
- 月度收益热图
- 滚动 Sharpe
- 绩效汇总

---

## Step 5: 推理

```bash
python -m ml.predict              # 最新模型，最近 5 个交易日
python -m ml.predict --days 10    # 最近 10 个
python -m ml.predict --model ml/artifacts/model_xxx.pkl
```

输出示例：

```
模型: model_20250501_120000.pkl

最近 5 个交易日信号:
----------------------------------------
  2025-04-24  pred=+0.00312  做多 ▲
  2025-04-25  pred=-0.00128  观望 —
  2025-04-28  pred=+0.00287  做多 ▲
  2025-04-29  pred=-0.00341  做空 ▼
  2025-04-30  pred=+0.00095  观望 —
```

---

## 作为库导入

### 在 Notebook 中使用

```python
import sys
sys.path.insert(0, '..')

from ml.train import load_config, load_data
from ml.features import build_features
from ml.labels import build_labels
from ml.evaluate import rank_ic, oof_metrics, freq_to_annual_factor
from ml.backtest import run_ml_backtest

cfg = load_config('../ml/config.yaml')
freq = cfg['data'].get('freq', '1D')

# 准备数据
klines, klines_clean = load_data(cfg)
X = build_features(klines, klines_clean, cfg['features'], freq=freq)
y = build_labels(klines, cfg['labels'], freq=freq)

# 运行回测
result, signal = run_ml_backtest(cfg)
```

### 评估工具

```python
from ml.evaluate import rank_ic, sharpe_from_pred, freq_to_annual_factor

ic = rank_ic(y_true, y_pred)
bars_yr = freq_to_annual_factor('15min')  # → 4032
sharpe, mdd = sharpe_from_pred(y_true, y_pred, bars_per_year=bars_yr)
```

---

## 标签类型详解

### regression（默认）

预测未来 `forward_bars` 根 bar 的收益率，连续值。

```yaml
labels:
  label_type: regression
  forward_bars: 4
```

### classification

三分类：收益 > threshold → +1，< -threshold → -1，中间 → 0。

```yaml
labels:
  label_type: classification
  forward_bars: 4
  threshold: 0.003
```

### triple_barrier

三重障碍标签：设 ATR 自适应止盈止损 + 时间障碍，先触碰哪个就标为 +1（止盈）/ -1（止损）/ 0（超时）。

```yaml
labels:
  label_type: triple_barrier
  forward_bars: 20          # 时间障碍（最长持仓 bar 数）
  atr_period: 14
  atr_multiplier: 1.5       # 止盈止损 = close ± 1.5 × ATR
```

---

## 频率支持

| freq | 含义 | FU 年化系数 | 适用场景 |
|------|------|-------------|----------|
| `1min` | 不聚合 | 60480 | 超短线（不推荐，噪声大） |
| `5min` | 5分钟 | 12096 | 短线 |
| `15min` | 15分钟 | 4032 | 短中线 |
| `30min` | 30分钟 | 2016 | 中线 |
| `1H` | 1小时 | 1008 | 中线 |
| `4H` | 4小时 | 252 | 日级别（FU ~1根/天） |
| `1D` | 日频 | 252 | 日级别 |

---

## 注意事项

| 事项 | 说明 |
|------|------|
| **前视偏差** | 所有特征基于当前时刻及之前数据；`shift(-n)` 只用于标签 |
| **gap >= forward_bars** | 训练 gap 必须 >= 标签窗口，否则标签重叠导致 IC 虚高 |
| **交叉验证** | 使用 `TimeSeriesSplit`，训练集始终在验证集之前 |
| **换月清洗** | `klines_clean` 已在加载阶段处理 OBV/OIChange 换月跳变 |
| **回测信号映射** | 聚合频率信号通过 `ffill` 映射到 1min K 线，同一窗口内信号不变 |
| **滚动分位阈值** | 回测策略使用滚动窗口计算开平仓阈值，不存在未来函数 |
| **artifacts 不入库** | `ml/artifacts/` 已加入 `.gitignore` |
| **config.yaml 入库** | 配置文件应提交，便于复现实验 |
