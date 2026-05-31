"""
LiveRunner — 实盘/仿真运行主循环。

与 BacktestEngine 的核心区别：
  - 使用 tqsdk TqApi 的 wait_update() 驱动（非 for 循环）
  - 订单同步提交到交易所（或 TqSim）
  - 每 bar 前执行 RiskManager 检查
  - 支持 Ctrl-C 优雅退出（平仓 + 撤单）

使用示例
--------
    from futuresquant.execution.runner import LiveRunner, LiveConfig
    from futuresquant.strategy.examples.ma_cross import MACrossStrategy

    runner = LiveRunner(MACrossStrategy(5, 20), LiveConfig(symbol='SHFE.fu2509'))
    runner.run()
"""

from __future__ import annotations

import logging
import signal
import sys
from dataclasses import dataclass, field
from datetime import date
from typing import Dict

import pandas as pd

from futuresquant.backtest.contract import ContractSpec
from futuresquant.backtest.order import Direction, Offset
from futuresquant.config.settings import settings
from futuresquant.execution.broker import LiveBroker
from futuresquant.execution.risk import RiskManager
from futuresquant.strategy.base import Context, StrategyBase

logger = logging.getLogger(__name__)


@dataclass
class LiveConfig:
    symbol: str                    # tqsdk 格式，如 'SHFE.fu2509'
    kline_duration: int = 60       # K 线周期（秒），默认 1 分钟
    kline_count: int = 500         # 订阅的历史 K 线根数（策略预热用）
    initial_capital: float = 1_000_000.0
    specs: Dict[str, ContractSpec] = field(default_factory=dict)
    # 风控覆盖（None = 使用 settings 中的值）
    max_position_lots: int | None = None
    max_drawdown_pct: float | None = None
    daily_loss_limit: float | None = None
    max_order_notional: float | None = None


class LiveRunner:
    """连接 tqsdk，驱动策略的实盘/仿真运行循环。"""

    def __init__(self, strategy: StrategyBase, config: LiveConfig):
        self.strategy = strategy
        self.config = config
        self._running = True

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def run(self) -> None:
        cfg = self.config

        # 构建 tqsdk API（根据模式选择账户类型）
        api = self._build_api()

        risk = RiskManager(
            max_position_lots=cfg.max_position_lots or settings.risk_max_position_lots,
            max_drawdown_pct=cfg.max_drawdown_pct or settings.risk_max_drawdown_pct,
            daily_loss_limit=cfg.daily_loss_limit or settings.risk_daily_loss_limit,
            max_order_notional=cfg.max_order_notional or settings.risk_max_order_notional,
        )
        broker = LiveBroker(api, cfg.initial_capital, specs=cfg.specs)
        risk.reset_daily(cfg.initial_capital)

        # 注册优雅退出
        signal.signal(signal.SIGINT, lambda *_: self._shutdown(api, broker))
        signal.signal(signal.SIGTERM, lambda *_: self._shutdown(api, broker))

        try:
            klines = api.get_kline_serial(cfg.symbol, cfg.kline_duration, cfg.kline_count)
            self.strategy.on_start(klines.copy())
            logger.info("LiveRunner started: %s  mode=%s", cfg.symbol, settings.tq_mode)

            last_bar_id: int | None = None   # 用于判断是否有新 bar 收盘

            while self._running and api.wait_update():
                ts = pd.Timestamp.now()

                # --- 收集成交回报 ---
                fills = broker.collect_fills(ts)
                for fill in fills:
                    self.strategy.on_fill(fill)

                # --- 盯市 & 风控更新 ---
                close_price = float(klines.iloc[-1]["close"])
                equity = broker.mark_to_market({cfg.symbol: close_price}, ts)
                risk.update(equity, cfg.initial_capital)

                # --- 判断是否有新 bar（只在新 bar 收盘时执行策略） ---
                current_bar_id = int(klines.iloc[-1]["id"]) if "id" in klines.columns else id(klines.iloc[-1].to_dict())
                if not api.is_changing(klines.iloc[-1], "close"):
                    continue
                if current_bar_id == last_bar_id:
                    continue
                last_bar_id = current_bar_id

                # --- 策略执行（带风控包装的 Context） ---
                raw_ctx = Context(
                    symbol=cfg.symbol,
                    bar=klines.iloc[-1],
                    account=broker,   # type: ignore[arg-type]  — duck typing
                    timestamp=ts,
                )
                safe_ctx = _RiskContext(raw_ctx, risk, broker, cfg)
                self.strategy.on_bar(safe_ctx)

        except Exception:
            logger.exception("LiveRunner encountered an error")
        finally:
            self._cleanup(api, broker)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _build_api(self):
        """根据 settings.tq_mode 构建 TqApi。"""
        from tqsdk import TqApi, TqAuth, TqSim, TqAccount

        auth = TqAuth(settings.tq_username, settings.tq_password)

        if settings.tq_mode == "sim":
            logger.info("Running in TqSim (paper trading) mode")
            return TqApi(TqSim(), auth=auth)
        else:
            logger.info("Running in LIVE mode — broker=%s account=%s",
                        settings.tq_broker_id, settings.tq_account_id)
            account = TqAccount(
                settings.tq_broker_id,
                settings.tq_account_id,
                settings.tq_account_password,
            )
            return TqApi(account, auth=auth)

    def _shutdown(self, api, broker: LiveBroker) -> None:
        logger.warning("Shutdown signal received — cancelling orders and closing positions")
        self._running = False

    def _cleanup(self, api, broker: LiveBroker) -> None:
        try:
            broker.cancel_all()
            self.strategy.on_end(broker)   # type: ignore[arg-type]
        finally:
            api.close()
            logger.info("LiveRunner stopped.")


class _RiskContext(Context):
    """
    Context 子类：在转发下单前插入风控检查。
    """

    def __init__(self, ctx: Context, risk: RiskManager,
                 broker: LiveBroker, cfg: LiveConfig):
        # 直接复用父类字段
        super().__init__(
            symbol=ctx.symbol, bar=ctx.bar,
            account=ctx.account, timestamp=ctx.timestamp
        )
        self._risk = risk
        self._broker = broker
        self._cfg = cfg

    def buy_open(self, volume: int = 1, limit_price=None):
        return self._guarded_open(Direction.LONG, volume, limit_price)

    def sell_open(self, volume: int = 1, limit_price=None):
        return self._guarded_open(Direction.SHORT, volume, limit_price)

    def sell_close(self, volume=None, limit_price=None):
        pos = self._broker.get_position(self.symbol)
        vol = volume if volume is not None else max(pos.net, 0)
        ok, reason = self._risk.check_close(vol, pos.net)
        if not ok:
            logger.warning("RISK REJECT close: %s", reason)
            return None
        return self.account.submit_order(
            self.symbol, Direction.SHORT, Offset.CLOSE, vol, self.timestamp, limit_price
        )

    def buy_close(self, volume=None, limit_price=None):
        pos = self._broker.get_position(self.symbol)
        vol = volume if volume is not None else abs(min(pos.net, 0))
        ok, reason = self._risk.check_close(vol, pos.net)
        if not ok:
            logger.warning("RISK REJECT close: %s", reason)
            return None
        return self.account.submit_order(
            self.symbol, Direction.LONG, Offset.CLOSE, vol, self.timestamp, limit_price
        )

    def _guarded_open(self, direction: Direction, volume: int, limit_price):
        from futuresquant.backtest.contract import get_spec
        product = "".join(c for c in self.symbol.split(".")[-1] if c.isalpha()).upper()
        spec = get_spec(product)
        price = limit_price or float(self.bar["close"])
        pos = self._broker.get_position(self.symbol)

        ok, reason = self._risk.check_open(
            self.symbol, volume, price, spec.multiplier, pos.net
        )
        if not ok:
            logger.warning("RISK REJECT open: %s", reason)
            return None
        return self.account.submit_order(
            self.symbol, direction, Offset.OPEN, volume, self.timestamp, limit_price
        )
