# ML 流水线使用指南

## 目录结构

```
ml/
├── config.yaml        # 所有参数配置入口
├── features.py        # 特征构建（因子 + 滞后/滚动统计）
├── labels.py          # 标签定义（回归 / 分类）
├── train.py           # 训练主入口
├── evaluate.py        # 评估工具（IC/ICIR/Sharpe/图表）
├── predict.py         # 推理，输出交易信号
└── artifacts/         # 训练产出（自动生成，不入库）
    ├── model_<时间戳>.pkl
    ├── oof_preds_<时间戳>.parquet
    ├── feature_importance_<时间戳>.csv
    ├── metrics_<时间戳>.json
    └── *.html
```

---

## 快速开始

### 1. 安装 ML 依赖

```bash
# 从项目根目录执行
uv pip install -e ".[ml]"
```

### 2. 训练

```bash
# 使用默认配置（ml/config.yaml）
python -m ml.train

# 指定配置文件
python -m ml.train --config ml/config.yaml
```

训练结束后终端输出示例：

```
[09:12:01] 构建连续主力合约 2021-01-01 ~ 2025-04-30 …
[09:12:06] K线: 357,611  换月次数: 85
[09:12:08] 构建特征 …
[09:12:15] 构建标签 …
[09:12:15] 有效样本: 980 天  特征数: 168
[09:12:15] 开始 5 折时序交叉验证 …
  Fold 1  train=120d  val=172d  IC=0.0421  trees=312
  Fold 2  train=292d  val=172d  IC=0.0389  trees=287
  ...

OOF 汇总:
  IC             : 0.0412
  ICIR           : 0.3817
  IC_pos_pct     : 0.800
  Sharpe         : 0.721
  MaxDrawdown    : 0.1843

[09:12:38] 产出已保存至 ml/artifacts/
```

### 3. 推理（生成信号）

```bash
# 使用最新模型，打印最近 5 个交易日信号
python -m ml.predict

# 指定模型和天数
python -m ml.predict --model ml/artifacts/model_20250501_120000.pkl --days 10
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

## 调整实验参数

**只需修改 `ml/config.yaml`，代码无需改动。**

### 切换数据范围

```yaml
data:
  start_date: "2019-01-01"   # 延长历史
  end_date:   "2025-04-30"
```

### 切换单合约 / 连续合约

```yaml
data:
  use_continuous: false        # 切回单合约
  single_contract: FU2505
```

### 调整预测窗口

```yaml
labels:
  forward_days: 5              # 预测未来 5 日收益（建议配合 IC 衰减曲线选择）
```

### 切换回归 / 分类标签

```yaml
labels:
  label_type: classification   # regression | classification
  threshold: 0.003             # ±0.3% 以内视为噪声，归入 0 类
```

### 调整滞后和滚动窗口

```yaml
features:
  lags: [1, 5, 20]             # 对所有因子添加这几阶滞后
  rolling_windows: [10, 20]    # 添加 10 日和 20 日的滚动均值/标准差
```

### 调整模型超参数

```yaml
model:
  params:
    num_leaves: 63             # 增大树复杂度
    learning_rate: 0.03
    feature_fraction: 0.7
  num_boost_round: 1000
  early_stopping_rounds: 100
```

### 调整交叉验证折数

```yaml
training:
  n_splits: 8                  # 更多折，评估更稳定
  gap: 5                       # 训练集与验证集间隔 5 天（减少标签重叠）
  min_train_size: 240          # 最少 240 天训练数据
```

---

## 作为库导入

### 在 Notebook 中使用

```python
import sys
sys.path.insert(0, '..')

import yaml
from ml.train import load_config, load_data
from ml.features import build_features
from ml.labels import build_labels
from ml.evaluate import oof_metrics, plot_oof_nav, plot_importance

# 加载配置
cfg = load_config('../ml/config.yaml')

# 准备数据
klines, klines_clean = load_data(cfg)
X = build_features(klines, klines_clean, cfg['features'])
y = build_labels(klines, cfg['labels'])

# 加载已有 OOF 预测，做进一步分析
import pandas as pd
oof = pd.read_parquet('../ml/artifacts/oof_preds_<时间戳>.parquet')
plot_oof_nav(oof['y_true'], oof['oof_pred']).show()
```

### 单独使用评估工具

```python
from ml.evaluate import rank_ic, sharpe_from_pred, plot_importance
import pandas as pd

# 计算任意预测的 IC
ic = rank_ic(y_true, y_pred)

# 计算多空 Sharpe
sharpe, mdd = sharpe_from_pred(y_true, y_pred)
```

---

## 典型工作流

```
因子分析（02_factor_analysis.ipynb）
    ↓ 观察 IC 衰减曲线 → 确定 forward_days
    ↓ 观察 T 检验     → 过滤噪声因子
    ↓ 观察相关性矩阵  → 了解特征冗余情况
修改 ml/config.yaml
    ↓
python -m ml.train
    ↓ 查看 ml/artifacts/*.html
    ↓ 查看 OOF IC / Sharpe / 特征重要性
根据特征重要性反馈，在 features.py 中裁剪低价值特征
    ↓
python -m ml.train   （迭代）
    ↓
python -m ml.predict  （生成信号）
```

---

## 注意事项

| 事项 | 说明 |
|------|------|
| **前视偏差** | 所有特征基于当前时刻及之前数据；`shift(-n)` 只用于标签 |
| **交叉验证** | 使用 `TimeSeriesSplit`，训练集始终在验证集之前，不允许随机打乱 |
| **换月清洗** | `klines_clean` 已在加载阶段处理 OBV/OIChange 换月跳变 |
| **artifacts 不入库** | `ml/artifacts/` 已加入 `.gitignore`，模型文件需单独管理 |
| **config.yaml 入库** | 配置文件应提交，便于复现实验 |
