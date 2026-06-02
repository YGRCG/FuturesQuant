# 因子使用文档

> 适用项目：FuturesQuant — 中国期货量化研究框架  
> 最后更新：2026-06

---

## 目录

1. [因子体系概览](#1-因子体系概览)
2. [因子清单](#2-因子清单)
3. [基础用法](#3-基础用法)
4. [传统因子分析](#4-传统因子分析)
5. [机器学习使用指南](#5-机器学习使用指南)
6. [注意事项](#6-注意事项)

---

## 1. 因子体系概览

```
futuresquant/factors/
├── base.py                  # Factor 抽象基类，算术组合，zscore/rank 工具
├── engine.py                # FactorEngine — 批量计算 + 标准化
└── technical/
    ├── momentum.py          # 动量因子（5个）
    ├── trend.py             # 趋势因子（4个）
    ├── volatility.py        # 波动率因子（4个）
    ├── volume.py            # 成交量因子（4个）
    └── futures_structure.py # 期货结构因子（4个）+ Lagged 包装器
```

所有因子共享统一接口：

```python
factor.compute(klines: pd.DataFrame) -> pd.Series
# 输入：含 open/high/low/close/volume/amount/open_interest 列的 1-min K 线 DataFrame
# 输出：与输入同索引的 pd.Series，name = factor.name
```

---

## 2. 因子清单

### 2.1 动量因子

| 因子类 | 参数 | 输出含义 | 信号方向 |
|--------|------|----------|----------|
| `ROC(period=20)` | period: 回看周期 | 收益率 `close/close[-n]-1` | 正=动量向上 |
| `MOM(period=20)` | period: 回看周期 | 绝对价差 `close-close[-n]` | 正=价格上涨 |
| `RSI(period=14)` | period: 平滑周期 | Wilder RSI，范围 0–100 | >70 超买，<30 超卖 |
| `BollingerBand(period=20, n_std=2.0)` | period: 均线周期 | 价格在布林带中的分位 0–1 | 0=下轨，1=上轨 |
| `TSMomentum(fast=5, slow=20)` | fast/slow: 均线周期 | 慢均值-快均值（日收益） | 正=上升趋势 |

### 2.2 趋势因子

| 因子类 | 参数 | 输出含义 | 信号方向 |
|--------|------|----------|----------|
| `MACross(fast=5, slow=20, ma_type="ema")` | fast/slow: EMA 周期 | `fast_MA/slow_MA - 1` | 正=多头排列 |
| `MACD(fast=12, slow=26, signal=9)` | 标准 MACD 参数 | 柱状图（价格归一化） | 正=做多信号 |
| `ADX(period=14)` | period: 平滑周期 | 趋势强度 0–100 | >25 为强趋势 |
| `PriceChannel(period=20)` | period: 通道周期 | 唐奇安通道分位 0–1 | 0=通道底，1=通道顶 |

### 2.3 波动率因子

| 因子类 | 参数 | 输出含义 | 信号方向 |
|--------|------|----------|----------|
| `ATR(period=14)` | period: 平滑周期 | 原始 ATR（价格单位） | 越大波动越高 |
| `NormATR(period=14)` | period: 平滑周期 | `ATR/close`，可跨合约比较 | 越大波动越高 |
| `HistoricalVolatility(window=240)` | window: 滚动窗口（bar） | 年化历史波动率 | 越大越不稳定 |
| `VolatilityRatio(fast=20, slow=120)` | fast/slow: 波动窗口 | `短期波动/长期波动` | >1=波动扩张（突破） |

### 2.4 成交量因子

| 因子类 | 参数 | 输出含义 | 信号方向 |
|--------|------|----------|----------|
| `VolumeRatio(period=20)` | period: 均量窗口 | `成交量/均量` | >1=放量 |
| `OBV(norm_period=20)` | norm_period: 归一化周期 | 累计 OBV 的变化率 | 正=资金净流入 |
| `VWAP(period=60, atr_period=14)` | period: VWAP 窗口 | `(close-VWAP)/ATR` | 正=强于均价 |
| `OpenInterestChange(period=20)` | period: 变化周期 | 持仓量变化率 | 正+价涨=趋势确认 |

### 2.5 期货结构因子（期货特有）

| 因子类 | 参数 | 输出含义 | 备注 |
|--------|------|----------|------|
| `DayOfWeek()` | — | 星期几 0–4（Mon–Fri） | 捕捉周内效应 |
| `MinuteOfDay()` | — | 日内分钟归一化 [0, 1] | 连续日内位置编码 |
| `SessionCode(sessions=None)` | sessions: 自定义时段字典 | 0=夜盘 1=早盘 2=午前 3=午后 NaN=非交易 | 默认上期所 FU 时段 |
| `DaysToExpiry()` | — | 距交割月首日的自然日数 | 需连续合约 `contract` 列 |
| `Lagged(factor, lag)` | factor: 任意因子；lag: 滞后 bar 数 | 将任意因子输出向后移 N bar | 为树模型提供时序记忆 |

**SessionCode 自定义示例（大商所无夜盘品种）：**

```python
import datetime
SessionCode(sessions={
    1: (datetime.time(9, 0),  datetime.time(11, 30)),
    3: (datetime.time(13, 30), datetime.time(15, 0)),
})
```

---

## 3. 基础用法

### 3.1 单因子计算

```python
from futuresquant.factors.technical import ROC, ADX

roc = ROC(20)
series = roc.compute(klines)   # pd.Series，name="ROC_20"
```

### 3.2 因子算术组合

```python
from futuresquant.factors.technical import ROC, NormATR

# 动量 / 波动率 = 风险调整动量
risk_adj_mom = ROC(20) / NormATR(14)
series = risk_adj_mom.compute(klines)
```

### 3.3 批量计算（推荐）

```python
from futuresquant.factors.engine import FactorEngine
from futuresquant.factors.technical import (
    ROC, RSI, MACD, ADX, NormATR, VolatilityRatio,
    VolumeRatio, OBV, DayOfWeek, MinuteOfDay,
    SessionCode, DaysToExpiry, Lagged,
)

FACTORS = [
    ROC(20), RSI(14), MACD(12, 26, 9), ADX(14),
    NormATR(14), VolatilityRatio(20, 120),
    VolumeRatio(20), OBV(20),
    DayOfWeek(), MinuteOfDay(), SessionCode(), DaysToExpiry(),
    Lagged(ROC(20), lag=1), Lagged(ROC(20), lag=5),
]

engine = FactorEngine(FACTORS)
factor_df = engine.compute(klines)
# 返回 DataFrame：index=DatetimeIndex，columns=因子名称
```

### 3.4 连续主力合约加载

```python
from futuresquant.data.loader import FuturesDataLoader
from futuresquant.data.universe import ContinuousContract

loader = FuturesDataLoader("raw_data/1min_FU", cache_dir="cache")
cc = ContinuousContract(loader, product="FU", adjust="back",
                         roll_n_days_before_expiry=5)
klines = cc.build(start="2021-01-01", end="2025-04-30")
# klines 含 contract 列，DaysToExpiry 因子需要此列
```

**换月节点清洗（OBV / OIChange 必须处理）：**

```python
import numpy as np

ROLL_BUFFER = 30
roll_mask = klines['contract'] != klines['contract'].shift(1)
locs = [klines.index.get_loc(ts) for ts in klines.index[roll_mask]]
bad_pos = set()
for loc in locs:
    bad_pos.update(range(max(0, loc - ROLL_BUFFER),
                         min(len(klines), loc + ROLL_BUFFER + 1)))
klines_clean = klines.copy()
klines_clean.loc[klines.index[sorted(bad_pos)], ['volume', 'open_interest']] = np.nan

factor_df = engine.compute(klines_clean)  # 传入清洗后的数据
```

---

## 4. 传统因子分析

### 4.1 Rank IC / ICIR

```python
from scipy import stats

def calc_ic(factor_df, klines, forward_bars=60):
    fwd_ret = klines['close'].pct_change(forward_bars).shift(-forward_bars)
    daily_f = factor_df.resample('1D').last()
    daily_r = fwd_ret.resample('1D').last()
    records = {}
    for col in daily_f.columns:
        aligned = pd.concat([daily_f[col], daily_r], axis=1).dropna()
        if len(aligned) < 5:
            continue
        rho, pval = stats.spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
        records[col] = {'IC': rho, 'p_value': pval}
    return pd.DataFrame(records).T

ic_summary = calc_ic(factor_df, klines, forward_bars=60)
# |IC| > 0.02 视为有信号，ICIR > 0.5 视为稳定
```

### 4.2 判断标准

| 指标 | 有效阈值 | 说明 |
|------|----------|------|
| \|IC\| | > 0.02 | 因子有方向性预测力 |
| ICIR | > 0.5 | 因子稳定，不依赖少数时段 |
| IC 衰减 | 观察峰值所在窗口 | 决定最优持仓周期 |
| 换手率 | 越低越好 | 高换手率 = 高交易成本 |
| 盈亏平衡成本 | 需 > 实际手续费 | `\|IC\| × σ / (2 × 换手率)` |

---

## 5. 机器学习使用指南

### 5.1 整体流程

```
原始 K 线
    ↓ engine.compute()
因子矩阵 (n_bars × n_factors)
    ↓ 特征工程（滞后、滚动统计）
ML 特征集 X
    ↓ 定义预测目标 Y（IC 衰减曲线确定窗口）
标签 Y
    ↓ 时序交叉验证（PurgedKFold）
训练 → 验证 → 测试
    ↓ 特征重要性反馈
筛选因子 → 迭代
```

### 5.2 特征准备

```python
import pandas as pd
import numpy as np

# 步骤 1：计算所有因子
factor_df = engine.compute(klines_clean)   # shape: (n_bars, n_factors)

# 步骤 2：日频聚合（以日为预测单位时）
daily_f = factor_df.resample('1D').last()

# 步骤 3：添加滞后特征（树模型不感知时序，手动添加）
lags = [1, 5, 20]
lag_feats = pd.concat(
    [daily_f.shift(lag).add_suffix(f'_lag{lag}') for lag in lags],
    axis=1
)

# 步骤 4：添加滚动统计特征（捕捉均值、波动率的近期变化）
roll_mean = daily_f.rolling(20).mean().add_suffix('_rmean20')
roll_std  = daily_f.rolling(20).std().add_suffix('_rstd20')

# 步骤 5：合并所有特征
X_raw = pd.concat([daily_f, lag_feats, roll_mean, roll_std], axis=1)
X_raw = X_raw.replace([np.inf, -np.inf], np.nan)
```

### 5.3 标签定义

**根据 IC 衰减曲线选择预测窗口（见 Section 7）。**

```python
# 示例：预测未来 1 日收益率（分类标签）
daily_close = klines['close'].resample('1D').last()
forward_ret = daily_close.pct_change(1).shift(-1)          # 未来 1 日收益

# 回归标签：直接使用收益率
y_reg = forward_ret

# 分类标签：三分类（做多/观望/做空）
THRESHOLD = 0.003   # 0.3%，低于此视为噪声
y_clf = pd.cut(forward_ret,
               bins=[-np.inf, -THRESHOLD, THRESHOLD, np.inf],
               labels=[-1, 0, 1]).astype(float)
```

### 5.4 时序交叉验证（核心：防止未来数据泄漏）

**期货数据有自相关性，不能用随机 KFold，必须用时序分割。**

```python
from sklearn.model_selection import TimeSeriesSplit

# 对齐 X 和 y
common_idx = X_raw.dropna().index.intersection(forward_ret.dropna().index)
X = X_raw.loc[common_idx]
y = forward_ret.loc[common_idx]

tscv = TimeSeriesSplit(n_splits=5, gap=1)   # gap=1 避免信息泄漏

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    # 训练模型 ...
```

**推荐用 PurgedKFold（处理标签重叠问题）：**

```python
# pip install mlfinlab
from mlfinlab.cross_validation import PurgedKFold

pkf = PurgedKFold(n_splits=5, n_jobs=1, pct_embargo=0.01)
```

### 5.5 模型训练示例（LightGBM）

```python
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import numpy as np

params = {
    'objective':    'regression',
    'metric':       'rmse',
    'num_leaves':   31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq':  5,
    'verbose':      -1,
}

tscv = TimeSeriesSplit(n_splits=5, gap=1)
oof_preds = np.zeros(len(y))

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model = lgb.train(
        params, dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )
    oof_preds[val_idx] = model.predict(X_val)
    rmse = mean_squared_error(y_val, oof_preds[val_idx], squared=False)
    print(f'Fold {fold+1}  RMSE={rmse:.6f}')

# IC of OOF predictions（比 RMSE 更有意义的金融指标）
oof_ic, _ = stats.spearmanr(oof_preds, y.values)
print(f'OOF Rank IC: {oof_ic:.4f}')
```

### 5.6 特征重要性与因子筛选

```python
import plotly.express as px

# 获取特征重要性（最后一个 fold 的模型）
importance = pd.Series(
    model.feature_importance(importance_type='gain'),
    index=X.columns
).sort_values(ascending=False)

# 可视化 Top 30
fig = px.bar(importance.head(30).reset_index(),
             x='index', y=0,
             title='LightGBM 特征重要性（Gain）')
fig.show()

# 筛选：只保留重要性 > 阈值的因子
IMPORTANCE_THRESHOLD = importance.quantile(0.5)  # 保留前 50%
selected_features = importance[importance > IMPORTANCE_THRESHOLD].index.tolist()
print(f'筛选后特征数: {len(selected_features)} / {len(X.columns)}')
```

### 5.7 注意事项汇总

| 问题 | 处理方式 |
|------|----------|
| **前视偏差** | 所有特征必须基于当前时刻及之前的数据；`shift(-n)` 只能用于标签 |
| **数据泄漏** | 验证集和测试集的 scaler、encoder 必须在训练集上 fit |
| **标签重叠** | 使用多日前瞻标签（如 5 日收益）时，相邻样本标签重叠，用 PurgedKFold 或加 gap |
| **换月跳变** | OBV、OIChange 在换月节点置 NaN（已在加载阶段处理）|
| **因子量纲** | 树模型不需要归一化；线性模型（Lasso、Ridge）需要 StandardScaler |
| **过拟合检验** | 对比 IS/OOS IC 比值（walk-forward 分析，见 `optimize/walk_forward.py`） |
| **非平稳性** | 使用收益率（`pct_change`）而非价格水平作为特征和标签 |

### 5.8 从因子分析到 ML 的决策路径

```
1. IC 衰减（Section 7）
       → 确定预测窗口（Y 标签的 forward_bars）

2. T 检验（Section 14）+ IC 大小
       → 过滤 p > 0.2 且 |IC| < 0.01 的纯噪声因子，不进入 ML

3. 相关性矩阵（Section 4）
       → 高相关组（> 0.7）内只保留 IC 最高的代表

4. 换手率（Section 9）
       → 高换手率因子（> 1.5）作为特征时，ML 模型可能学到假信号

5. 市场状态 IC（Section 11）
       → 若因子在不同状态下 IC 符号相反，考虑将市场状态作为元特征输入

6. ML 特征重要性（Section 5.6）
       → 反向验证：重要性低但 IC 高的因子，可能 ML 以非线性方式使用，保留观察
```

---

## 6. 注意事项

### 连续合约

- 必须使用后复权（`adjust="back"`）；前复权会使价格水平失真
- 换月节点 `volume` 和 `open_interest` 必须清洗（±30 bar），否则 OBV/OIChange 产生巨大跳变

### DaysToExpiry 因子

- 仅在 `klines` 含 `contract` 列时有效（即使用 `ContinuousContract.build()` 加载）
- 单合约 klines（`loader.load("FU2505")`）此因子返回全 NaN

### Lagged 因子

- 在 notebook 的因子列表里，`Lagged(ROC(20), 1)` 和 `Lagged(ROC(20), 5)` 是针对当前 IC Top 因子手动挑选的
- 完整流程：先跑一遍 IC 分析 → 看 Top3~5 因子 → 对这些因子加 lag=1,5,20 的滞后版本

### 时间特征对 IC 分析的意义

`DayOfWeek`、`MinuteOfDay`、`SessionCode` 是 ML 特征，不是传统意义上的 alpha 因子：
- 它们的 IC 几乎为 0（无直接线性预测力）
- 但在 ML 模型里，它们帮助模型"知道现在是什么时段"，从而对其他因子的信号给予不同权重
- 在 Section 12（分时效应）中已经可以看到这种状态依赖性
