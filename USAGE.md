# FuturesQuant 使用说明

基于 Python 的中国期货量化研究框架，覆盖数据 → 因子 → 回测 → 优化 → 实盘全流程。

---

## 目录

1. [环境搭建](#1-环境搭建)
2. [数据层](#2-数据层)
3. [因子层](#3-因子层)
4. [回测引擎](#4-回测引擎)
5. [策略开发](#5-策略开发)
6. [参数优化](#6-参数优化)
7. [实盘执行](#7-实盘执行)
8. [研究 Notebooks](#8-研究-notebooks)

---

## 1. 环境搭建

### 1.1 创建虚拟环境

```bash
cd I:\stock\FuturesQuant

# 创建 Python 3.10 虚拟环境
uv venv --python=3.10

# 激活（Windows）
.venv\Scripts\activate

# 安装项目及开发依赖
uv pip install -e ".[dev]"
```

> **规则**：所有包管理一律通过 `uv pip`，不直接使用 `pip`。

### 1.2 配置天勤账号

复制 `.env.example` 为 `.env` 并填入真实值：

```bash
copy .env.example .env
```

```ini
# .env
TQ_USERNAME=你的天勤用户名
TQ_PASSWORD=你的天勤密码
TQ_MODE=sim                  # sim=仿真  live=实盘

# 风控参数（可选调整）
RISK_MAX_POSITION_LOTS=10
RISK_MAX_DRAWDOWN_PCT=0.05
RISK_DAILY_LOSS_LIMIT=5000
```

### 1.3 数据目录结构

```
raw_data/
└── 1min_FU/          ← 命名规范：1min_{品种代码}
    ├── FU0503.csv
    ├── FU2210.csv
    └── ...
```

新增品种只需创建对应子目录，框架自动识别。

### 1.4 运行测试

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
# 预期：127 passed
```

---

## 2. 数据层

### 2.1 单品种加载（FuturesDataLoader）

```python
from futuresquant.data.loader import FuturesDataLoader

# 直接读 CSV（无缓存）
loader = FuturesDataLoader(r"I:\stock\FuturesQuant\raw_data\1min_FU")

# 启用 Parquet 缓存（首次建立后读取速度提升 ~9 倍）
loader = FuturesDataLoader(
    r"I:\stock\FuturesQuant\raw_data\1min_FU",
    cache_dir=r"I:\stock\FuturesQuant\cache",
)

# 加载单合约
df = loader.load("FU2210")
df = loader.load("FU2210", start="2022-01-01", end="2022-10-31")

# 查看所有合约
contracts = loader.list_contracts()
print(contracts[0].contract_id)    # "FU0503"
print(contracts[0].tqsdk_symbol)   # "SHFE.fu0503"

# 预热缓存（一次性转换所有 CSV）
loader.warm_up_cache()
```

返回的 DataFrame 列：`open / high / low / close / volume / amount / open_interest`，DatetimeIndex。

### 2.2 多品种根目录（DataRegistry）

```python
from futuresquant.data.loader import DataRegistry

registry = DataRegistry(
    raw_data_dir=r"I:\stock\FuturesQuant\raw_data",
    cache_dir=r"I:\stock\FuturesQuant\cache",
)

registry.list_products()           # ["CU", "FU", "RB", ...]
loader_fu = registry.get_loader("FU")
df = registry.load("FU2210", start="2022-01-01")

# 批量预热所有品种的缓存
registry.warm_up_cache()
```

### 2.3 连续合约

```python
from futuresquant.data.universe import ContinuousContract

cc = ContinuousContract(
    loader,
    product="FU",
    adjust="back",            # "back"=后复权  "ratio"=比例复权  "none"=不复权
    roll_n_days_before_expiry=5,
)

cont = cc.build(start="2020-01-01", end="2022-12-31")
# 额外列 "contract" 标记当前使用的合约代码
```

---

## 3. 因子层

### 3.1 内置因子（17 个）

| 类别 | 因子类 | 主要参数 |
|------|--------|---------|
| **动量** | `ROC(period)` | 价格变化率 |
| | `MOM(period)` | 绝对价格动量 |
| | `RSI(period)` | 相对强弱指数 |
| | `BollingerBand(period, n_std)` | 布林带 %B 位置 |
| | `TSMomentum(fast, slow)` | 时序动量 |
| **趋势** | `MACross(fast, slow, ma_type)` | 均线交叉 |
| | `MACD(fast, slow, signal)` | MACD 柱状 |
| | `ADX(period)` | 趋势强度 |
| | `PriceChannel(period)` | Donchian 通道位置 |
| **波动率** | `ATR(period)` | 平均真实波幅 |
| | `NormATR(period)` | 归一化 ATR |
| | `HistoricalVolatility(window)` | 年化历史波动率 |
| | `VolatilityRatio(fast, slow)` | 短/长期波动比 |
| **成交量** | `VolumeRatio(period)` | 成交量相对均值 |
| | `OBV(norm_period)` | 能量潮 |
| | `VWAP(period, atr_period)` | VWAP 偏离度 |
| | `OpenInterestChange(period)` | 持仓量变化率 |

### 3.2 单因子计算

```python
from futuresquant.factors.technical import ROC, ATR, MACross

roc = ROC(period=20)
series = roc.compute(klines)   # pd.Series，DatetimeIndex
```

### 3.3 因子算术组合

```python
# 因子支持 +  -  *  /  -(取反) 直接组合
momentum_adj = ROC(20) - NormATR(14)    # 风险调整动量
combined = ROC(20) * VolumeRatio(20)    # 成交量确认的动量
```

### 3.4 批量计算（FactorEngine）

```python
from futuresquant.factors.engine import FactorEngine
from futuresquant.factors.technical import ROC, RSI, ATR

engine = FactorEngine(
    factors=[ROC(20), RSI(14), ATR(14)],
    norm="zscore",      # "none" / "zscore" / "rank"
    norm_window=240,
)

# 单合约 → DataFrame
factor_df = engine.compute(klines)

# 多合约面板 → MultiIndex DataFrame
panel = engine.compute_panel({"FU2209": df1, "FU2210": df2})

# 截面标准化
panel_cs = FactorEngine.cross_section_zscore(panel)
```

### 3.5 多因子 IC 加权合成信号

```python
from futuresquant.factors.signal import MultiFactorSignal
from futuresquant.factors.technical import ROC, MACross, NormATR, RSI

signal_gen = MultiFactorSignal(
    factors=[ROC(20), MACross(5, 20), NormATR(14), RSI(14)],
    forward_bars=60,       # IC 计算的前瞻期（bar 数）
    ic_window=240 * 5,     # 滚动 IC 回看窗口（5 天）
    norm_window=240 * 20,  # 信号标准化窗口（20 天）
    min_ic_abs=0.005,      # 过滤低 IC 因子
)

signal = signal_gen.compute(klines)    # pd.Series，正值偏多，负值偏空
weights = signal_gen.factor_weights(klines)  # 各因子动态权重
```

---

## 4. 回测引擎

### 4.1 运行回测

```python
from futuresquant.data.loader import FuturesDataLoader
from futuresquant.backtest.engine import BacktestEngine, BacktestConfig
from futuresquant.strategy.examples import MACrossStrategy

loader = FuturesDataLoader(r"I:\stock\FuturesQuant\raw_data\1min_FU",
                           cache_dir=r"I:\stock\FuturesQuant\cache")
klines = loader.load("FU2210", start="2022-01-01", end="2022-10-31")

config = BacktestConfig(
    symbol="FU2210",
    initial_capital=1_000_000,
    slippage_ticks=1,        # 每笔成交的滑点（tick 数）
)

result = BacktestEngine(MACrossStrategy(fast=5, slow=20), config).run(klines)
```

### 4.2 查看结果

```python
# 控制台打印摘要
result.print_summary()

# 打开浏览器交互式 HTML 报告（权益曲线 + 回撤 + 价格）
result.report()

# 获取绩效指标字典
m = result.metrics
# 包含：total_return / annual_return / annual_vol /
#       sharpe / sortino / max_drawdown / calmar

# 获取权益曲线
equity = result.account.equity_curve()   # pd.Series

# 获取所有成交记录
fills = result.account.fills   # List[Fill]
```

### 4.3 执行模型

- 订单在 **bar T 提交**，在 **bar T+1 开盘成交**（T+1 成交，无前视偏差）
- 市价单以 `open ± slippage_ticks × tick_size` 成交
- 回测结束时自动平掉所有残余持仓

### 4.4 合约规格

内置常见品种参数（`backtest/contract.py`）：

| 品种 | 乘数 | 保证金率 | 手续费 |
|------|------|---------|--------|
| FU | 10 吨/手 | 12% | 0.4 元/手 |
| RB | 10 吨/手 | 10% | 1.0 元/手 |
| IF | 300 元/点 | 15% | 万分之 2.3 |

如需自定义：

```python
from futuresquant.backtest.contract import ContractSpec

config = BacktestConfig(
    symbol="XX2501",
    specs={"XX": ContractSpec("XX", multiplier=5, tick_size=1.0,
                               margin_ratio=0.10, commission_per_lot=2.0)},
)
```

---

## 5. 策略开发

### 5.1 继承 StrategyBase

```python
import pandas as pd
from futuresquant.strategy.base import Context, StrategyBase

class MyStrategy(StrategyBase):
    warmup_bars = 30   # 跳过前 N 根 bar（等待指标稳定）

    def on_start(self, klines: pd.DataFrame) -> None:
        """在第一根 bar 前调用一次，可在此做向量化预计算。"""
        self._ma = klines["close"].rolling(20).mean()

    def on_bar(self, ctx: Context) -> None:
        """每根 bar 调用一次，策略核心逻辑写在这里。"""
        ts = ctx.bar.name
        ma = self._ma.loc[ts]
        close = float(ctx.bar["close"])
        pos = ctx.position   # 当前净持仓（正=多，负=空）

        if close > ma and pos <= 0:
            if pos < 0:
                ctx.buy_close()    # 先平空
            ctx.buy_open(1)        # 再开多

        elif close < ma and pos >= 0:
            if pos > 0:
                ctx.sell_close()   # 先平多
            ctx.sell_open(1)       # 再开空

    def on_fill(self, fill) -> None:
        """每笔成交后回调（可选）。"""

    def on_end(self, account) -> None:
        """最后一根 bar 后回调（可选）。"""
```

### 5.2 Context 下单接口

| 方法 | 说明 |
|------|------|
| `ctx.buy_open(volume=1)` | 买入开仓（做多） |
| `ctx.sell_close(volume=None)` | 卖出平仓（平多，volume=None 平全部） |
| `ctx.sell_open(volume=1)` | 卖出开仓（做空） |
| `ctx.buy_close(volume=None)` | 买入平仓（平空） |
| `ctx.close_position()` | 平掉全部持仓（自动判断方向） |
| `ctx.position` | 当前净持仓手数 |
| `ctx.bar` | 当前 bar 的 pd.Series |
| `ctx.timestamp` | 当前 bar 的时间戳 |

### 5.3 内置示例策略

```python
from futuresquant.strategy.examples import (
    MACrossStrategy,          # MA 双均线
    ATRBreakoutStrategy,      # ATR 通道突破
    RSIMeanReversionStrategy, # RSI 均值回归
    BollingerBandStrategy,    # 布林带（回归/突破双模式）
    MultiFactorStrategy,      # 多因子信号驱动
)

# 布林带策略两种模式
strat_rv = BollingerBandStrategy(period=20, mode="reversion")   # 均值回归
strat_bk = BollingerBandStrategy(period=20, mode="breakout")    # 突破

# 多因子策略
from futuresquant.factors.signal import MultiFactorSignal
signal_gen = MultiFactorSignal([ROC(20), MACross(5, 20), NormATR(14)])
strat_mf = MultiFactorStrategy(signal_gen, entry_threshold=1.0, exit_threshold=0.3)
```

---

## 6. 参数优化

### 6.1 网格搜索

```python
from futuresquant.optimize import GridSearchOptimizer

optimizer = GridSearchOptimizer(
    strategy_factory=lambda fast, slow: MACrossStrategy(fast=fast, slow=slow),
    param_grid={"fast": [3, 5, 10, 15], "slow": [20, 30, 40, 60]},
    config=BacktestConfig(symbol="FU2210", initial_capital=1_000_000),
    metric="sharpe",                              # 优化目标
    constraints=[lambda p: p["fast"] < p["slow"]],  # 约束条件
)

result = optimizer.run(klines)
print(result.best_params)    # {"fast": 5, "slow": 30}
print(result.best_score)     # 最优 Sharpe

result.summary(top_n=10)     # 全部结果 DataFrame（按 Sharpe 降序）
result.heatmap_data("fast", "slow")   # 热图数据（用于可视化）
```

### 6.2 遗传算法（大参数空间）

```python
from futuresquant.optimize import GeneticOptimizer

optimizer = GeneticOptimizer(
    strategy_factory=factory,
    param_grid={"fast": list(range(3, 25, 2)), "slow": list(range(20, 100, 5))},
    config=config,
    metric="sharpe",
    constraints=constraints,
    population_size=30,   # 种群大小
    n_generations=20,     # 进化代数
    mutation_rate=0.15,   # 基因变异概率
    elitism_k=3,          # 每代保留精英数
    seed=42,              # 随机种子（可复现）
)

result = optimizer.run(klines)
```

### 6.3 Walk-Forward Analysis（防过拟合）

```python
from futuresquant.optimize import WalkForwardOptimizer

wf = WalkForwardOptimizer(
    strategy_factory=factory,
    param_grid={"fast": [3, 5, 10], "slow": [20, 30, 40]},
    config=config,
    metric="sharpe",
    constraints=constraints,
    n_splits=5,       # 窗口数量
    is_ratio=0.7,     # IS 占比（70% 优化，30% 验证）
    optimizer="grid", # "grid" 或 "genetic"
)

wf_result = wf.run(klines)

# 防过拟合判据
print(f"OOS/IS 衰减比: {wf_result.degradation_ratio:.3f}")
# > 0.5  → 参数稳健
# 0~0.5  → 轻度过拟合，谨慎使用
# < 0    → 严重过拟合，放弃

wf_result.summary()           # 各窗口 IS/OOS 对比
wf_result.param_stability()   # 各窗口最优参数（稳定性检验）
wf_result.oos_equity          # 拼接的 OOS 权益曲线（真实绩效估计）
wf_result.oos_metrics         # 基于 OOS 权益的绩效指标
```

---

## 7. 实盘执行

### 7.1 仿真（TqSim）

```python
from futuresquant.execution.runner import LiveRunner, LiveConfig
from futuresquant.strategy.examples import MACrossStrategy

runner = LiveRunner(
    strategy=MACrossStrategy(fast=5, slow=20),
    config=LiveConfig(
        symbol="SHFE.fu2509",   # tqsdk 格式，含交易所前缀
        kline_duration=60,      # K 线周期（秒）
        initial_capital=1_000_000,
    ),
)
runner.run()   # 阻塞运行，Ctrl-C 优雅退出
```

> 仿真模式不需要期货账户，`.env` 中 `TQ_MODE=sim` 即可。

### 7.2 实盘

在 `.env` 中设置：

```ini
TQ_MODE=live
TQ_BROKER_ID=你的期货公司代码
TQ_ACCOUNT_ID=你的账户号
TQ_ACCOUNT_PASSWORD=你的CTP密码
```

策略代码**不需要任何修改**，框架自动切换到实盘模式。

### 7.3 风控规则

所有开仓在提交前经过 `RiskManager` 检查，任一条件触发即拒绝或熔断：

| 规则 | 默认值 | 触发后行为 |
|------|--------|-----------|
| 持仓限额 | 10 手/品种 | 拒绝超额开仓 |
| 最大回撤 | 5% | 停止所有开仓 |
| 日亏损限额 | 5,000 元 | 停止所有开仓 |
| 单笔名义价值 | 50 万元 | 拒绝该笔订单 |

风控参数在 `.env` 中配置，无需改代码。

---

## 8. 研究 Notebooks

启动方式：

```bash
cd I:\stock\FuturesQuant
.venv\Scripts\jupyter lab research\
```

| 文件 | 内容 |
|------|------|
| `01_data_exploration.ipynb` | 合约列表、K 线可视化、连续合约换月标注、收益率分布 |
| `02_factor_analysis.ipynb` | IC/ICIR 排名、滚动 IC、因子相关性热图、五分位分层回测、自相关衰减 |
| `03_backtest_analysis.ipynb` | 端到端回测、权益曲线、成交点标注、参数敏感性热图、月度收益热图 |
| `04_multi_factor.ipynb` | 多因子信号可视化、因子权重演化、六策略横向对比、Sharpe-回撤散点图 |
| `05_optimization.ipynb` | 网格搜索热图、遗传算法收敛、WFA IS/OOS 对比、参数稳定性分析 |

---

## 常见问题

**Q: 如何添加新品种数据？**
在 `raw_data/` 下创建 `1min_{品种代码}/` 目录，放入同格式 CSV，`DataRegistry` 自动识别。

**Q: 缓存什么时候需要更新？**
无需手动操作。每次 `load()` 时自动比较 CSV 与 Parquet 的修改时间，CSV 更新后下次读取自动重建。

**Q: 如何在回测和实盘间切换？**
只替换入口类：`BacktestEngine → LiveRunner`。策略代码（`on_bar` 逻辑）完全不变。

**Q: 连续合约后复权会影响实际 PnL 计算吗？**
回测引擎使用**原始价格**计算 PnL（通过各合约的 CSV 直接加载）。连续合约仅用于因子计算和研究，不直接参与 PnL 核算。

**Q: 遗传算法每次结果是否一致？**
设置 `seed` 参数后完全可复现（`GeneticOptimizer(..., seed=42)`）。
